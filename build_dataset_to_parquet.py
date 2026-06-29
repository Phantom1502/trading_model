"""
build_dataset_to_parquet.py
============================
Sinh dataset pretrain từ file Parquet INPUT lớn (vd 100 nến/dòng, hàng chục nghìn dòng)
và ghi OUTPUT cũng ra Parquet — xử lý theo CHUNK để không phải load toàn bộ file
vào RAM một lúc, tránh treo máy / hết RAM khi file quá to.

Khác với cách làm trước (đọc hết bằng pd.read_parquet() rồi ghép thành 1 chuỗi
text khổng lồ), script này:

  1. Đọc INPUT theo ROW GROUP bằng pyarrow.parquet.ParquetFile.iter_batches()
     → mỗi lần chỉ load 1 phần nhỏ vào RAM, không phải toàn bộ file.
  2. Với mỗi batch nhỏ, cắt random các đoạn con (20-30 nến) và sinh curriculum text.
  3. Ghi kết quả ra OUTPUT bằng pyarrow.parquet.ParquetWriter, APPEND theo từng
     chunk → không giữ toàn bộ output trong RAM.

SCHEMA OUTPUT (cố định, khớp đúng schema model đang dùng để load dữ liệu):

    text          : string  — nội dung pretrain (curriculum text)
    source        : string  — tên nguồn dữ liệu gốc (vd "XAUUSD_1Min"), truyền
                              vào qua tham số `source_name`, KHÔNG suy ra từ data
    token_length  : int64   — luôn = 0. Pipeline này KHÔNG tokenize, không load
                              tokenizer — cột này chỉ giữ chỗ đúng schema, việc đo
                              độ dài token (nếu cần) làm ở bước load dữ liệu để train,
                              không làm ở bước sinh dataset.
    meta          : string  — JSON string chứa toàn bộ metadata sinh sample:
                              {"source_chart_index", "slice_start", "slice_end",
                               "num_candles", "num_layers"}

Yêu cầu: pip install pyarrow

Cách dùng:
    python build_dataset_to_parquet.py \\
        --input data/chart_XAUUSD_dataset_1Min.parquet \\
        --output data/curriculum_pretrain_dataset.parquet \\
        --source-name XAUUSD_1Min \\
        --slices-per-chart 4 \\
        --batch-size 2000
"""

import argparse
import json
import random
from typing import Iterator, List, Optional

from app.utils.chart.candle_parser import CandleParser
from app.utils.chart.curriculum_generator import CurriculumGenerator
from app.memlm.tokenizer import VietnameseTokenizer
import os

# pyarrow chỉ cần cho phần đọc/ghi Parquet (iter_input_batches, build_dataset_to_parquet).
# Import trễ (lazy) để generate_samples_from_charts() vẫn test/dùng được độc lập
# ngay cả trên máy chưa cài pyarrow.
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    _HAS_PYARROW = True
except ImportError:
    _HAS_PYARROW = False


# Schema cố định để mọi row-group ghi ra đều cùng kiểu dữ liệu,
# tránh lỗi schema-mismatch khi pyarrow tự suy luận kiểu theo từng batch.
def _build_schema():
    return pa.schema([
        ("text",         pa.string()),
        ("source",       pa.string()),
        ("token_length", pa.int64()),
        ("meta",         pa.string()),
    ])


# ══════════════════════════════════════════════════════════════════
# SINH SAMPLE TỪ 1 BATCH CHART (list[str])
# ══════════════════════════════════════════════════════════════════

def generate_samples_from_charts(
    raw_charts: List[str],
    source_name: str = "unknown",
    slices_per_chart: int = 4,
    min_slice_len: int = 20,
    max_slice_len: int = 30,
    curriculum_mode: str = "random",
    min_layers: int = 2,
    max_layers: Optional[int] = None,
    swing_window: int = 2,
    rng: Optional[random.Random] = None,
    chart_index_offset: int = 0,
) -> List[dict]:
    """
    Sinh list các record (dict) từ 1 batch chart thô, đúng schema:
        text, source, token_length, meta

    `token_length` luôn = 0 (không tokenize ở bước sinh dataset).
    `meta` là JSON string gói lại slice_start/slice_end/num_candles/num_layers/
    source_chart_index — giữ thông tin debug/truy vấn mà không cần thêm cột.

    Parameters
    ----------
    chart_index_offset : để đánh số `source_chart_index` đúng và liên tục
                          across nhiều batch (vd batch thứ N thì offset = N * batch_size).
    """
    rng = rng or random
    records: List[dict] = []
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    tok_path = os.path.join(base_dir, "app", "memlm", "custom_tokenizer")
    print(f"Testing tokenizer: {tok_path}\n")

    tok = VietnameseTokenizer(pretrained_name=tok_path)
    print(f"Vocab size (BPE base + price): {tok.vocab_size:,}")
    print(f"  BPE base  : {len(tok.tokenizer):,}")
    print(f"  Price vocab: {len(tok.price_vocab):,}")
    print()

    for local_idx, raw in enumerate(raw_charts):
        chart_idx = chart_index_offset + local_idx
        base_parser = CandleParser(raw, swing_window=swing_window)
        n = len(base_parser)

        if n < min_slice_len:
            sub_ranges = [(0, n)]
        else:
            sub_ranges = []
            for _ in range(slices_per_chart):
                slice_len = rng.randint(min_slice_len, min(max_slice_len, n))
                max_start = n - slice_len
                start = rng.randint(0, max_start)
                sub_ranges.append((start, start + slice_len))

        for start, end in sub_ranges:
            sub_parser = base_parser.slice(start, end)
            gen = CurriculumGenerator(sub_parser)

            if curriculum_mode == "full":
                texts_and_layers = [(gen.generate_full_curriculum(), gen.num_layers)]
            elif curriculum_mode == "layers":
                # Mỗi tầng là 1 record riêng, mỗi record num_layers = 1
                texts_and_layers = [(t, 1) for t in gen.generate_layers_separately()]
            else:  # "random"
                text = gen.generate_random_subset(
                    min_layers=min_layers, max_layers=max_layers, rng=rng
                )
                texts_and_layers = [(text, text.count("===") // 2)]

            for text, n_layers in texts_and_layers:
                meta = {
                    "source_chart_index": chart_idx,
                    "slice_start":        start,
                    "slice_end":          end,
                    "num_candles":        end - start,
                    "num_layers":         n_layers,
                }
                
                # calculate token length
                tokens = tok.encode(text)
                
                
                records.append({
                    "text":         text,
                    "source":       source_name,
                    "token_length": len(tokens),
                    "meta":         json.dumps(meta, ensure_ascii=False),
                })

    return records


# ══════════════════════════════════════════════════════════════════
# XỬ LÝ FILE PARQUET LỚN THEO CHUNK (đọc + sinh + ghi)
# ══════════════════════════════════════════════════════════════════

def iter_input_batches(
    input_path: str,
    text_column: str = "text",
    batch_size: int = 2000,
) -> Iterator[List[str]]:
    """
    Đọc file Parquet INPUT theo từng batch nhỏ (không load hết file vào RAM).
    Mỗi lần yield ra 1 list các chuỗi chart thô (cột `text_column`).
    """
    if not _HAS_PYARROW:
        raise ImportError(
            "Cần cài pyarrow để đọc/ghi Parquet: pip install pyarrow"
        )
    pf = pq.ParquetFile(input_path)
    for record_batch in pf.iter_batches(batch_size=batch_size, columns=[text_column]):
        # to_pylist() chỉ convert batch nhỏ này, không phải toàn bộ file
        yield record_batch.column(text_column).to_pylist()


def build_dataset_to_parquet(
    input_path: str,
    output_path: str,
    source_name: str = "unknown",
    text_column: str = "text",
    batch_size: int = 2000,          # số dòng input đọc mỗi lần (không phải số sample output)
    slices_per_chart: int = 4,
    min_slice_len: int = 20,
    max_slice_len: int = 30,
    curriculum_mode: str = "random",
    min_layers: int = 2,
    max_layers: Optional[int] = None,
    swing_window: int = 2,
    seed: Optional[int] = None,
) -> int:
    """
    Đọc input_path theo chunk, sinh dataset, ghi APPEND ra output_path (parquet),
    đúng schema: text / source / token_length / meta.

    Vì xử lý theo chunk, RAM sử dụng chỉ tỉ lệ với `batch_size`, KHÔNG tỉ lệ với
    tổng số dòng trong file input — an toàn cho file vài GB hoặc hàng triệu dòng.

    Parameters
    ----------
    source_name : giá trị ghi vào cột `source` cho TẤT CẢ record sinh ra từ
                  input_path này (vd "XAUUSD_1Min"). Đặt cố định theo tham số,
                  không suy luận từ tên file hay nội dung.

    Returns
    -------
    int — tổng số sample (record) đã sinh và ghi ra output_path.
    """
    if not _HAS_PYARROW:
        raise ImportError(
            "Cần cài pyarrow để đọc/ghi Parquet: pip install pyarrow"
        )

    rng    = random.Random(seed) if seed is not None else random
    schema = _build_schema()

    writer: Optional[pq.ParquetWriter] = None
    total_samples = 0

    try:
        for batch_idx, raw_charts in enumerate(
            iter_input_batches(input_path, text_column=text_column, batch_size=batch_size)
        ):
            records = generate_samples_from_charts(
                raw_charts,
                source_name=source_name,
                slices_per_chart=slices_per_chart,
                min_slice_len=min_slice_len,
                max_slice_len=max_slice_len,
                curriculum_mode=curriculum_mode,
                min_layers=min_layers,
                max_layers=max_layers,
                swing_window=swing_window,
                rng=rng,
                chart_index_offset=batch_idx * batch_size,
            )

            if not records:
                continue

            table = pa.Table.from_pylist(records, schema=schema)

            if writer is None:
                writer = pq.ParquetWriter(output_path, schema, compression="snappy")
            writer.write_table(table)

            total_samples += len(records)
            print(f"[batch {batch_idx}] +{len(records)} sample "
                  f"(từ {len(raw_charts)} dòng input) | tổng cộng: {total_samples}")

    finally:
        if writer is not None:
            writer.close()

    print(f"\n✅ Hoàn tất. Tổng {total_samples} sample → {output_path}")
    return total_samples


# ══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sinh dataset pretrain từ Parquet lớn, ghi ra Parquet, xử lý theo chunk.")
    parser.add_argument("--input",  required=True, help="Đường dẫn file parquet INPUT (cột 'text' chứa chart thô)")
    parser.add_argument("--output", required=True, help="Đường dẫn file parquet OUTPUT")
    parser.add_argument("--source-name", default="unknown", help="Giá trị cột 'source' cho mọi record sinh ra")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--batch-size", type=int, default=2000, help="Số dòng input đọc mỗi lần (RAM tỉ lệ với số này)")
    parser.add_argument("--slices-per-chart", type=int, default=4)
    parser.add_argument("--min-slice-len", type=int, default=20)
    parser.add_argument("--max-slice-len", type=int, default=30)
    parser.add_argument("--curriculum-mode", default="random", choices=["full", "layers", "random"])
    parser.add_argument("--min-layers", type=int, default=2)
    parser.add_argument("--max-layers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    build_dataset_to_parquet(
        input_path=args.input,
        output_path=args.output,
        source_name=args.source_name,
        text_column=args.text_column,
        batch_size=args.batch_size,
        slices_per_chart=args.slices_per_chart,
        min_slice_len=args.min_slice_len,
        max_slice_len=args.max_slice_len,
        curriculum_mode=args.curriculum_mode,
        min_layers=args.min_layers,
        max_layers=args.max_layers,
        seed=args.seed,
    )