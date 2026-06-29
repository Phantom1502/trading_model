"""
book_and_output_pipeline.py
============================
Nhánh 3: BookPipeline
    Trích đoạn văn từ kho PDF sách trading (text-based),
    ghi ra parquet cùng schema với nhánh 1 và 2.

Nhánh 4: OutputPipeline
    Tổng hợp parquet lẻ từ 3 nhánh trên:
    - Merge thành file lớn
    - Shuffle
    - Split train/val
    - Ghi ra parquet sẵn sàng train

Schema chung (cả 4 nhánh):
    text          : string
    source        : string
    token_length  : int64   (0 nếu chưa tokenize)
    meta          : string  (JSON)

Cách dùng:
    from book_and_output_pipeline import BookPipeline, OutputPipeline

    # Nhánh 3
    book = BookPipeline(min_paragraph_len=80)
    book.build(input_dir="data/books", output_path="data/books.parquet")

    # Nhánh 4
    out = OutputPipeline()
    out.merge_and_split(
        input_paths = ["data/pretrain.parquet", "data/action.parquet", "data/books.parquet"],
        output_dir  = "data/final",
        val_ratio   = 0.05,
    )
"""

import glob
import json
import os
import random
import re
import statistics
from typing import List, Optional
from app.memlm.tokenizer import VietnameseTokenizer

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    _HAS_PYARROW = True
except ImportError:
    _HAS_PYARROW = False

try:
    import pdfplumber
    _HAS_PDFPLUMBER = True
except ImportError:
    _HAS_PDFPLUMBER = False


# ══════════════════════════════════════════════════════════════════════
# SCHEMA CHUNG
# ══════════════════════════════════════════════════════════════════════

def _schema():
    return pa.schema([
        ("text",         pa.string()),
        ("source",       pa.string()),
        ("token_length", pa.int64()),
        ("meta",         pa.string()),
    ])


# ══════════════════════════════════════════════════════════════════════
# NHÁNH 3: BOOK PIPELINE
# ══════════════════════════════════════════════════════════════════════

def _split_page_paragraphs(page, gap_ratio: float = 1.8) -> List[str]:
    """
    Tách 1 trang PDF thành đoạn văn dựa trên khoảng cách dọc giữa dòng.
    Dùng extract_text_lines() thay vì extract_text() để giữ được ranh giới đoạn.
    """
    lines = page.extract_text_lines()
    if not lines:
        return []
    if len(lines) == 1:
        t = lines[0]["text"].strip()
        return [t] if t else []

    gaps     = [lines[i]["top"] - lines[i - 1]["bottom"] for i in range(1, len(lines))]
    pos_gaps = [g for g in gaps if g > 0]
    median   = statistics.median(pos_gaps) if pos_gaps else 1.0
    threshold = max(median * gap_ratio, median + 2.0)

    paragraphs: List[str] = []
    current: List[str]    = [lines[0]["text"]]

    for i in range(1, len(lines)):
        gap = lines[i]["top"] - lines[i - 1]["bottom"]
        if gap > threshold:
            joined = re.sub(r"\s+", " ", " ".join(current)).strip()
            if joined:
                paragraphs.append(joined)
            current = [lines[i]["text"]]
        else:
            current.append(lines[i]["text"])

    joined = re.sub(r"\s+", " ", " ".join(current)).strip()
    if joined:
        paragraphs.append(joined)
    return paragraphs


def _merge_short_paragraphs(paragraphs: List[str], min_len: int) -> List[str]:
    """
    Gom các đoạn ngắn hơn min_len vào đoạn liền kề thay vì bỏ đi.
    Đoạn ngắn được nối vào đoạn tiếp theo, hoặc đoạn cuối nếu không còn gì.
    """
    merged = []
    buffer = ""

    for para in paragraphs:
        buffer = (buffer + " " + para).strip()
        if len(buffer) >= min_len:
            merged.append(buffer)
            buffer = ""

    # Còn thừa → gom vào đoạn cuối
    if buffer and merged:
        merged[-1] = (merged[-1] + " " + buffer).strip()
    elif buffer:
        merged.append(buffer)

    return merged


class BookPipeline:
    """
    Nhánh 3: Trích đoạn văn từ kho PDF sách trading.

    Mỗi PDF đọc xong → ghi ngay → không giữ nhiều sách trong RAM.
    Output cùng schema text/source/token_length/meta.
    """

    def __init__(self, tokenizer, min_paragraph_len: int = 80):
        if not _HAS_PDFPLUMBER:
            raise ImportError("Cần cài pdfplumber: pip install pdfplumber")
        if not _HAS_PYARROW:
            raise ImportError("Cần cài pyarrow: pip install pyarrow")
        self.tokenizer = tokenizer
        self.min_paragraph_len = min_paragraph_len

    def extract_file(self, pdf_path: str, source_name: str = None) -> List[dict]:
        """Trích đoạn văn từ 1 file PDF → list record."""
        source_name = source_name or os.path.basename(pdf_path)
        records: List[dict] = []
        para_idx = 0

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                raw_paras = _split_page_paragraphs(page)
                merged    = _merge_short_paragraphs(raw_paras, self.min_paragraph_len)
                for para in merged:
                    if not para.strip():
                        continue
                    meta = {
                        "page_number"    : page_num,
                        "paragraph_index": para_idx,
                        "char_length"    : len(para),
                    }
                    token_length = len(self.tokenizer.encode(para))
                    records.append({
                        "text"        : para,
                        "source"      : source_name,
                        "token_length": token_length,
                        "meta"        : json.dumps(meta, ensure_ascii=False),
                    })
                    para_idx += 1
        return records

    def build(
        self,
        input_dir  : str,
        output_path: str,
        pattern    : str = "*.pdf",
    ) -> int:
        """
        Xử lý toàn bộ kho PDF trong input_dir,
        ghi append ra output_path (parquet).
        Trả về tổng số đoạn văn.
        """
        schema = _schema()
        writer = None
        total  = 0

        pdf_paths = sorted(glob.glob(
            os.path.join(input_dir, "**", pattern), recursive=True
        ))
        if not pdf_paths:
            print(f"Không tìm thấy PDF nào trong: {input_dir}")
            return 0

        print(f"Tìm thấy {len(pdf_paths)} file PDF.")

        try:
            for path in pdf_paths:
                try:
                    records = self.extract_file(path)
                except Exception as e:
                    print(f"⚠️  Bỏ qua {path}: {e}")
                    continue

                if not records:
                    print(f"⚠️  {path}: không trích được đoạn văn (PDF scan ảnh?).")
                    continue

                table = pa.Table.from_pylist(records, schema=schema)
                if writer is None:
                    writer = pq.ParquetWriter(output_path, schema, compression="snappy")
                writer.write_table(table)
                total += len(records)
                print(f"[{os.path.basename(path)}] +{len(records)} đoạn | tổng: {total}")
        finally:
            if writer:
                writer.close()

        print(f"\n✅ Hoàn tất. {total} đoạn văn -> {output_path}")
        return total


# ══════════════════════════════════════════════════════════════════════
# NHÁNH 4: OUTPUT PIPELINE
# ══════════════════════════════════════════════════════════════════════

class OutputPipeline:
    """
    Nhánh 4: Tổng hợp parquet lẻ từ 3 nhánh trên.

    Các bước:
        1. Đọc nhiều parquet lẻ
        2. Merge + shuffle
        3. Split train/val
        4. Ghi ra parquet sẵn sàng train
    """

    def __init__(self, seed: int = 42):
        if not _HAS_PYARROW:
            raise ImportError("Cần cài pyarrow: pip install pyarrow")
        self.seed = seed

    # ── Đọc nhiều parquet → list records ──────────────────────────
    def _load_all(self, input_paths: List[str], text_column: str = "text") -> List[dict]:
        """Đọc tất cả parquet, trả về list record (dict)."""
        schema  = _schema()
        records = []

        for path in input_paths:
            if not os.path.exists(path):
                print(f"⚠️  Không tìm thấy: {path}, bỏ qua.")
                continue
            try:
                table = pq.read_table(path, schema=schema)
                batch = table.to_pydict()
                n     = len(batch["text"])
                for i in range(n):
                    records.append({
                        "text"        : batch["text"][i],
                        "source"      : batch["source"][i],
                        "token_length": batch["token_length"][i],
                        "meta"        : batch["meta"][i],
                    })
                print(f"[{os.path.basename(path)}] {n} records")
            except Exception as e:
                print(f"⚠️  Lỗi đọc {path}: {e}")

        return records

    # ── Merge nhiều file nhỏ trong thư mục ────────────────────────
    def merge_dir(
        self,
        input_dir  : str,
        output_path: str,
        pattern    : str = "*.parquet",
        compression: str = "snappy",
        target_size_mb: int = None,
    ) -> int:
        """
        Gom tất cả parquet trong input_dir thành 1 file (hoặc nhiều file
        ~target_size_mb nếu chỉ định).
        """
        paths = sorted(glob.glob(os.path.join(input_dir, pattern)))
        if not paths:
            print(f"Không tìm thấy parquet trong: {input_dir}")
            return 0

        schema = pq.ParquetFile(paths[0]).schema.to_arrow_schema()
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        if target_size_mb is None:
            # Gom thành 1 file
            writer = pq.ParquetWriter(output_path, schema, compression=compression)
            total  = 0
            try:
                for path in paths:
                    pf = pq.ParquetFile(path)
                    for rg in range(pf.num_row_groups):
                        writer.write_table(pf.read_row_group(rg))
                    total += pq.read_metadata(path).num_rows
            finally:
                writer.close()
            print(f"✅ Merged {len(paths)} files → {output_path} ({total} rows)")
            return total
        else:
            # Gom thành nhiều file ~target_size_mb
            target_bytes = target_size_mb * 1024 * 1024
            base, ext    = os.path.splitext(output_path)
            writer       = None
            cur_size     = 0
            file_idx     = 1
            total        = 0

            try:
                for path in paths:
                    size = os.path.getsize(path)
                    if writer and cur_size + size > target_bytes:
                        writer.close()
                        writer   = None
                        cur_size = 0

                    if writer is None:
                        out    = f"{base}_part{file_idx:04d}{ext}"
                        writer = pq.ParquetWriter(out, schema, compression=compression)
                        print(f"Tạo file: {out}")
                        file_idx += 1

                    pf = pq.ParquetFile(path)
                    for rg in range(pf.num_row_groups):
                        writer.write_table(pf.read_row_group(rg))
                    cur_size += size
                    total    += pq.read_metadata(path).num_rows
            finally:
                if writer:
                    writer.close()

            print(f"✅ Merged {len(paths)} files → {file_idx-1} parts ({total} rows)")
            return total

    # ── Merge + shuffle + split train/val ─────────────────────────
    def merge_and_split(
        self,
        input_paths: List[str],
        output_dir : str,
        val_ratio  : float = 0.05,
        compression: str   = "snappy",
        min_token_length: int = 0,
    ) -> dict:
        """
        Đọc nhiều parquet lẻ → shuffle → split train/val → ghi ra output_dir.

        Parameters
        ----------
        input_paths     : list đường dẫn parquet từ 3 nhánh
        output_dir      : thư mục output (tự tạo nếu chưa có)
        val_ratio       : tỉ lệ val (mặc định 5%)
        min_token_length: lọc bỏ record có token_length < N (0 = không lọc)

        Returns
        -------
        dict: {"train": N, "val": M, "total": N+M}
        """
        os.makedirs(output_dir, exist_ok=True)
        print(f"Đọc {len(input_paths)} file...")
        records = self._load_all(input_paths)

        if not records:
            print("Không có records nào.")
            return {"train": 0, "val": 0, "total": 0}

        # Filter token_length nếu cần
        if min_token_length > 0:
            before = len(records)
            records = [r for r in records if r["token_length"] >= min_token_length]
            print(f"Filter token_length >= {min_token_length}: {before} → {len(records)}")

        # Shuffle
        rng = random.Random(self.seed)
        rng.shuffle(records)
        print(f"Tổng: {len(records)} records, shuffle xong.")

        # Split
        n_val   = max(1, int(len(records) * val_ratio))
        n_train = len(records) - n_val
        val_records   = records[:n_val]
        train_records = records[n_val:]

        schema = _schema()

        # Ghi train
        train_path = os.path.join(output_dir, "train.parquet")
        pq.write_table(
            pa.Table.from_pylist(train_records, schema=schema),
            train_path, compression=compression,
        )

        # Ghi val
        val_path = os.path.join(output_dir, "val.parquet")
        pq.write_table(
            pa.Table.from_pylist(val_records, schema=schema),
            val_path, compression=compression,
        )

        print(f"\n✅ Train: {n_train} records → {train_path}")
        print(f"✅ Val  : {n_val}   records → {val_path}")

        # Ghi stats
        stats = self._compute_stats(train_records, val_records)
        stats_path = os.path.join(output_dir, "stats.json")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"✅ Stats → {stats_path}")

        return {"train": n_train, "val": n_val, "total": len(records)}

    # ── Stats ──────────────────────────────────────────────────────
    @staticmethod
    def _compute_stats(train: List[dict], val: List[dict]) -> dict:
        """Tính thống kê cơ bản về dataset."""
        from collections import Counter

        def _source_dist(records):
            return dict(Counter(r["source"] for r in records))

        def _token_stats(records):
            lengths = [r["token_length"] for r in records if r["token_length"] > 0]
            if not lengths:
                return {}
            return {
                "mean"  : round(sum(lengths) / len(lengths), 1),
                "min"   : min(lengths),
                "max"   : max(lengths),
                "total" : sum(lengths),
            }

        return {
            "train": {
                "count"       : len(train),
                "source_dist" : _source_dist(train),
                "token_stats" : _token_stats(train),
            },
            "val": {
                "count"       : len(val),
                "source_dist" : _source_dist(val),
                "token_stats" : _token_stats(val),
            },
        }

    # ── Preview ────────────────────────────────────────────────────
    @staticmethod
    def preview(parquet_path: str, n: int = 3) -> None:
        """In nhanh N records đầu tiên từ parquet."""
        table = pq.read_table(parquet_path)
        df    = table.to_pandas()
        print(f"\n── {parquet_path} ({len(df)} rows) ──")
        for i, row in df.head(n).iterrows():
            print(f"\n[{i}] source={row['source']} | token_length={row['token_length']}")
            print(f"     text: {str(row['text'])[:200]}...")
            print(f"     meta: {row['meta']}")


# ══════════════════════════════════════════════════════════════════════
# DEMO
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import tempfile
    import os

    print("=" * 60)
    print("Demo book_and_output_pipeline.py")
    print("=" * 60)

    # ── Mock data để test OutputPipeline ──
    schema = _schema()

    def _make_mock_parquet(path, source, n=20):
        records = [
            {
                "text"        : f"Đây là đoạn văn số {i} từ nguồn {source}.",
                "source"      : source,
                "token_length": 10 + i,
                "meta"        : json.dumps({"index": i}),
            }
            for i in range(n)
        ]
        pq.write_table(pa.Table.from_pylist(records, schema=schema), path)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Tạo 3 file mock
        p1 = os.path.join(tmpdir, "pretrain.parquet")
        p2 = os.path.join(tmpdir, "action.parquet")
        p3 = os.path.join(tmpdir, "books.parquet")
        _make_mock_parquet(p1, "chart_pretrain", n=30)
        _make_mock_parquet(p2, "action_score",   n=20)
        _make_mock_parquet(p3, "trading_books",  n=10)

        out_dir = os.path.join(tmpdir, "final")
        pipeline = OutputPipeline(seed=42)
        result   = pipeline.merge_and_split(
            input_paths = [p1, p2, p3],
            output_dir  = out_dir,
            val_ratio   = 0.1,
        )
        print(f"\nKết quả: {result}")

        # Preview
        pipeline.preview(os.path.join(out_dir, "train.parquet"), n=2)
        pipeline.preview(os.path.join(out_dir, "val.parquet"),   n=2)

        # Stats
        with open(os.path.join(out_dir, "stats.json")) as f:
            print(f"\nStats:\n{f.read()}")