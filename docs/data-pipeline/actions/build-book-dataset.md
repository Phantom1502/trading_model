# Action: Build Book/PDF Dataset (trích văn bản sách trading)

Files: `app/utils/chart/book_and_output_pipeline.py` (`BookPipeline`,
khuyến nghị), `build_pdf_dataset_to_parquet.py` (script gốc, cùng logic).

## Mục đích

Trích **nguyên văn** đoạn văn từ kho PDF sách trading (text-based, **không
phải scan ảnh**) — pretrain kiểu trích xuất thuần, không diễn giải/tóm tắt
lại bằng AI.

## Cách tách đoạn văn — dựa trên khoảng cách dòng, không phải blank-line

`pdfplumber.extract_text()` **không giữ lại** dòng trống giữa các đoạn (mọi
dòng bị nối liên tục) → không thể tách đoạn bằng cách tìm blank-line trên
text đã extract.

Giải pháp: dùng `page.extract_text_lines()` (có toạ độ `top`/`bottom` từng
dòng):
1. Tính gap dọc giữa mọi cặp dòng liên tiếp trong trang.
2. Lấy **median** làm "gap bình thường trong 1 đoạn" (line-height).
3. Gap giữa 2 dòng liên tiếp > `median * gap_threshold_ratio` (mặc định
   1.8, tối thiểu `median + 2.0`) → ranh giới đoạn văn mới.

```python
book = BookPipeline(tokenizer, min_paragraph_len=500)
book.build(input_dir="data/books", output_path="data/books.parquet")
```

## Gom đoạn ngắn thay vì bỏ (`_merge_short_paragraphs`)

`BookPipeline` (bản mới) **gộp** các đoạn ngắn hơn `min_paragraph_len` vào
đoạn liền kề thay vì loại bỏ hoàn toàn — giữ được nội dung thay vì mất dữ
liệu chỉ vì 1 đoạn bị pdfplumber tách hơi vụn. Bản script gốc
(`build_pdf_dataset_to_parquet.py`) dùng `is_likely_noise()` để **loại bỏ
hẳn** đoạn quá ngắn hoặc tỉ lệ ký tự chữ cái thấp — cân nhắc dùng bản nào
tuỳ mục tiêu (giữ tối đa nội dung vs. lọc noise mạnh tay).

## Lọc noise (`is_likely_noise`, dùng ở bản script gốc)

Loại bỏ nếu:
- Đoạn ngắn hơn `min_len` (mặc định 80 ký tự).
- Tỉ lệ ký tự alpha / tổng độ dài < 0.5 (bảng số liệu, header/footer lẫn
  vào text).

## Xử lý theo từng file — không giữ nhiều sách trong RAM

Mỗi PDF được đọc, extract, ghi ra parquet ngay rồi giải phóng — không đợi
xử lý xong toàn bộ kho sách mới ghi. `try/except` bọc quanh **từng file**
để 1 file lỗi (PDF hỏng, scan ảnh không có text layer...) không làm sập
toàn bộ quá trình — chỉ in cảnh báo và bỏ qua file đó.

## Schema output

Giống schema chung — xem `docs/conventions/parquet-schema.md`. `meta` chứa
`page_number`, `paragraph_index`, `char_length`.

## Giới hạn hiện tại

- **Chỉ xử lý PDF text-based.** PDF scan ảnh (không có text layer) sẽ
  extract ra rỗng hoặc rác — cần OCR riêng nếu có PDF dạng này (không nằm
  trong scope hiện tại của `BookPipeline`).
- `token_length` chỉ được tính nếu truyền `tokenizer` vào constructor —
  nếu không truyền, cột này = 0 và cần tính lại ở bước load dữ liệu train.
