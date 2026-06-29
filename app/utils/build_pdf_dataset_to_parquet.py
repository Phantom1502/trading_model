"""
build_pdf_dataset_to_parquet.py
================================
Trích xuất nội dung NGUYÊN VĂN từ một KHO PDF (sách trading, text-based —
không phải scan ảnh), chunk theo ĐOẠN VĂN (paragraph), và ghi ra Parquet
CÙNG SCHEMA với dataset nến (text / source / token_length / meta) để có thể
gộp chung 2 dataset lại train cùng lúc.

Khác với dataset nến (sinh text bằng curriculum), dataset này là pretrain
THUẦN TRÍCH XUẤT — không diễn giải lại, không AI tóm tắt, giữ nguyên văn
từng đoạn sách.

XỬ LÝ THEO CHUNK (giống file nến): mỗi PDF được đọc và xử lý XONG HẲN rồi
ghi ra ngay, KHÔNG giữ nội dung của nhiều sách trong RAM cùng lúc — an toàn
cho kho sách có hàng trăm file hoặc vài file rất dày.

CÁCH TÁCH ĐOẠN VĂN:
    pdftotext -layout giữ lại khoảng trống (2+ dòng trống liên tiếp) giữa
    các đoạn văn thật trong PDF gốc. Script này:
      1. Trích text từng trang bằng pdfplumber (text-based PDF).
      2. Ghép text toàn bộ sách lại (giữ ranh giới trang để biết page_number).
      3. Tách thành đoạn văn bằng regex trên 2+ dòng trống liên tiếp.
      4. Trong mỗi đoạn, NỐI các dòng bị word-wrap lại thành 1 dòng liên tục
         (xuống dòng giữa câu là artifact của PDF, không phải ngắt ý thật).
      5. Lọc bỏ đoạn quá ngắn (header/footer/số trang rác).

SCHEMA OUTPUT (giống hệt file nến để gộp chung dataset):
    text          : string — nguyên văn 1 đoạn văn từ sách
    source        : string — tên file PDF nguồn (vd "price_action_basics.pdf")
    token_length  : int64  — luôn = 0 (không tokenize ở bước này)
    meta          : string — JSON: {"page_number", "paragraph_index", "char_length"}

Yêu cầu: pip install pdfplumber pyarrow

Cách dùng:
    python build_pdf_dataset_to_parquet.py \\
        --input-dir data/books \\
        --output data/pdf_pretrain_dataset.parquet \\
        --min-paragraph-len 80
"""

import argparse
import glob
import json
import os
import re
import statistics
from typing import Iterator, List, Optional

import pdfplumber
from app.memlm.tokenizer import VietnameseTokenizer

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    _HAS_PYARROW = True
except ImportError:
    _HAS_PYARROW = False


def _build_schema():
    return pa.schema([
        ("text",         pa.string()),
        ("source",       pa.string()),
        ("token_length", pa.int64()),
        ("meta",         pa.string()),
    ])


# ══════════════════════════════════════════════════════════════════
# TRÍCH XUẤT ĐOẠN VĂN THEO TỪNG TRANG (dựa vào KHOẢNG CÁCH DỌC giữa dòng)
# ══════════════════════════════════════════════════════════════════
#
# LƯU Ý KỸ THUẬT: pdfplumber.extract_text() KHÔNG giữ lại dòng trống giữa
# các đoạn văn (toàn bộ text bị nối thành các dòng liên tục, mất thông tin
# "đoạn nào kết thúc ở đâu"). Vì vậy KHÔNG thể tách đoạn bằng cách tìm
# blank-line trên text đã extract.
#
# Giải pháp đáng tin cậy hơn: dùng extract_text_lines(), mỗi dòng có tọa độ
# (top, bottom). Khoảng cách dọc (gap) giữa 2 dòng LIÊN TIẾP trong cùng 1
# đoạn văn là ĐỀU NHAU (line-height bình thường, vd ~3pt), còn khoảng cách
# giữa đoạn này và đoạn sau LUÔN LỚN HƠN RÕ RỆT (vd ~20pt). Do đó:
#   1. Tính tất cả các gap giữa dòng liên tiếp trong trang.
#   2. Lấy gap TRUNG VỊ (median) làm gap "bình thường trong dòng".
#   3. Nếu gap > median * threshold_ratio → đó là ranh giới đoạn văn mới.

def split_page_into_paragraphs(
    page: "pdfplumber.page.Page",
    gap_threshold_ratio: float = 1.8,
) -> List[str]:
    """
    Tách 1 trang PDF thành các đoạn văn dựa trên khoảng cách dọc giữa dòng.

    Parameters
    ----------
    gap_threshold_ratio : nếu gap giữa 2 dòng > median_gap * ratio này,
                          coi đó là ranh giới đoạn văn mới.
    """
    lines = page.extract_text_lines()
    if not lines:
        return []

    if len(lines) == 1:
        text = lines[0]["text"].strip()
        return [text] if text else []

    gaps = [lines[i]["top"] - lines[i - 1]["bottom"] for i in range(1, len(lines))]
    positive_gaps = [g for g in gaps if g > 0]
    median_gap = statistics.median(positive_gaps) if positive_gaps else 1.0
    threshold = max(median_gap * gap_threshold_ratio, median_gap + 2.0)

    paragraphs: List[str] = []
    current_lines: List[str] = [lines[0]["text"]]

    for i in range(1, len(lines)):
        gap = lines[i]["top"] - lines[i - 1]["bottom"]
        if gap > threshold:
            joined = re.sub(r"\s+", " ", " ".join(current_lines)).strip()
            if joined:
                paragraphs.append(joined)
            current_lines = [lines[i]["text"]]
        else:
            current_lines.append(lines[i]["text"])

    joined = re.sub(r"\s+", " ", " ".join(current_lines)).strip()
    if joined:
        paragraphs.append(joined)

    return paragraphs


def extract_paragraphs_per_page(pdf_path: str) -> List[List[str]]:
    """
    Trích đoạn văn của TỪNG TRANG trong 1 PDF text-based.
    Trả về list theo trang, mỗi phần tử là list các đoạn văn của trang đó
    (giữ ranh giới trang để biết page_number ở bước sinh record).
    """
    pages_paragraphs: List[List[str]] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            pages_paragraphs.append(split_page_into_paragraphs(page))
    return pages_paragraphs


def is_likely_noise(paragraph: str, min_len: int = 80) -> bool:
    """
    Lọc đoạn khả năng là rác: header/footer/số trang/đoạn quá ngắn.
    - Quá ngắn (dưới min_len ký tự).
    - Tỷ lệ chữ cái quá thấp so với tổng độ dài (bảng số liệu lẫn vào text).
    """
    if len(paragraph) < min_len:
        return True

    alpha_count = sum(1 for ch in paragraph if ch.isalpha())
    if alpha_count / max(len(paragraph), 1) < 0.5:
        return True

    return False


# ══════════════════════════════════════════════════════════════════
# SINH RECORD TỪ 1 FILE PDF (đúng schema text/source/token_length/meta)
# ══════════════════════════════════════════════════════════════════

def generate_records_from_pdf(
    pdf_path: str,
    tokenizer: Optional[VietnameseTokenizer] = None,
    min_paragraph_len: int = 80,
    source_name: Optional[str] = None,
) -> List[dict]:
    """
    Đọc 1 file PDF, trả về list record đúng schema.

    Parameters
    ----------
    source_name : nếu None, dùng tên file (không kèm đường dẫn) làm `source`.
                  Truyền riêng nếu muốn gán tên nguồn khác (vd tên sách đẹp hơn
                  tên file thực tế).
    """
    source_name = source_name or os.path.basename(pdf_path)
    records: List[dict] = []

    pages_paragraphs = extract_paragraphs_per_page(pdf_path)

    paragraph_index = 0
    for page_number, paragraphs in enumerate(pages_paragraphs, start=1):
        for paragraph in paragraphs:
            if is_likely_noise(paragraph, min_len=min_paragraph_len):
                continue

            token_length = 0
            if tokenizer:
                token_length = len(tokenizer.tokenize(paragraph))

            meta = {
                "page_number":     page_number,
                "paragraph_index": paragraph_index,
                "char_length":     len(paragraph),
            }
            records.append({
                "text":         paragraph,
                "source":       source_name,
                "token_length": token_length,
                "meta":         json.dumps(meta, ensure_ascii=False),
            })
            paragraph_index += 1

    return records


# XỬ LÝ KHO PDF (nhiều file), GHI RA PARQUET THEO TỪNG FILE
# ══════════════════════════════════════════════════════════════════

def iter_pdf_paths(input_dir: str, pattern: str = "*.pdf") -> Iterator[str]:
    """Liệt kê đường dẫn PDF trong thư mục (không đọc nội dung)."""
    search_pattern = os.path.join(input_dir, "**", pattern)
    for path in sorted(glob.glob(search_pattern, recursive=True)):
        yield path


def build_pdf_dataset_to_parquet(
    input_dir: str,
    output_path: str,
    tokenizer: Optional[VietnameseTokenizer] = None,
    min_paragraph_len: int = 80,
    pattern: str = "*.pdf",
) -> int:
    """
    Xử lý TOÀN BỘ kho PDF trong `input_dir`, ghi ra `output_path` (parquet),
    XỬ LÝ TỪNG FILE MỘT — đọc xong 1 sách, sinh record, ghi append, rồi mới
    sang sách tiếp theo. KHÔNG giữ nội dung nhiều sách trong RAM cùng lúc,
    an toàn cho kho sách lớn.

    Returns
    -------
    int — tổng số đoạn văn (record) đã ghi ra output_path.
    """
    if not _HAS_PYARROW:
        raise ImportError("Cần cài pyarrow để ghi Parquet: pip install pyarrow")

    schema = _build_schema()
    writer: Optional[pq.ParquetWriter] = None
    total_records = 0

    try:
        for pdf_path in iter_pdf_paths(input_dir, pattern=pattern):
            try:
                records = generate_records_from_pdf(
                    pdf_path, tokenizer=tokenizer, min_paragraph_len=min_paragraph_len
                )
            except Exception as e:
                print(f"⚠️  Bỏ qua {pdf_path} do lỗi đọc: {e}")
                continue

            if not records:
                print(f"⚠️  {pdf_path}: không trích được đoạn văn nào (có thể là PDF scan ảnh).")
                continue

            table = pa.Table.from_pylist(records, schema=schema)
            if writer is None:
                writer = pq.ParquetWriter(output_path, schema, compression="snappy")
            writer.write_table(table)

            total_records += len(records)
            print(f"[{os.path.basename(pdf_path)}] +{len(records)} đoạn văn "
                  f"| tổng cộng: {total_records}")

    finally:
        if writer is not None:
            writer.close()

    print(f"\n✅ Hoàn tất. Tổng {total_records} đoạn văn từ kho PDF → {output_path}")
    return total_records


# ══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trích đoạn văn từ kho PDF trading (text-based), ghi ra Parquet cùng schema dataset nến."
    )
    parser.add_argument("--input-dir", required=True, help="Thư mục chứa các file PDF (quét cả thư mục con)")
    parser.add_argument("--output",    required=True, help="Đường dẫn file parquet OUTPUT")
    parser.add_argument("--min-paragraph-len", type=int, default=80,
                         help="Bỏ qua đoạn văn ngắn hơn N ký tự (header/footer/rác)")
    parser.add_argument("--pattern", default="*.pdf", help="Pattern tên file cần quét (mặc định *.pdf)")
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    tok_path = os.path.join(base_dir, "app", "memlm", "custom_tokenizer")
    print(f"Testing tokenizer: {tok_path}\n")

    tok = VietnameseTokenizer(pretrained_name=tok_path)
    print(f"Vocab size (BPE base + price): {tok.vocab_size:,}")
    print(f"  BPE base  : {len(tok.tokenizer):,}")
    print(f"  Price vocab: {len(tok.price_vocab):,}")
    print()

    build_pdf_dataset_to_parquet(
        input_dir=args.input_dir,
        output_path=args.output,
        tokenizer=tok,
        min_paragraph_len=args.min_paragraph_len,
        pattern=args.pattern,
    )