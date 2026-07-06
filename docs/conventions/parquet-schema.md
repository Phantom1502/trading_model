# Convention: Schema Parquet chung cho mọi nhánh dữ liệu

Áp dụng cho **tất cả** output của 4 nhánh sinh dữ liệu (chart pretrain,
action/trade-simulation, sách/PDF, và bất kỳ nguồn mới nào) — điều kiện bắt
buộc để có thể gộp chung (`OutputPipeline.merge_and_split`,
`mix_data_parquet.py`).

## Schema

```python
pa.schema([
    ("text",         pa.string()),
    ("source",       pa.string()),
    ("token_length", pa.int64()),
    ("meta",         pa.string()),   # JSON string
])
```

| Cột | Ý nghĩa | Quy ước |
|---|---|---|
| `text` | Nội dung pretrain (curriculum text, action sample, đoạn văn sách...) | Text thuần, sẵn sàng đưa vào `tokenizer.encode_batch()` |
| `source` | Tên nguồn dữ liệu gốc, **truyền vào qua tham số**, không tự suy ra từ nội dung | vd `"XAUUSD_1Min"`, `"action_data"`, tên file PDF |
| `token_length` | Số token sau khi tokenize | `0` nếu bước sinh dữ liệu **không** tokenize (không load tokenizer) — việc đo độ dài token khi đó dời sang bước load dữ liệu để train, không làm ở bước sinh dataset. Nếu tokenizer có sẵn lúc sinh (đa số action/curriculum/book pipeline hiện tại có), tính luôn để tiện lọc/thống kê sau này. |
| `meta` | JSON string chứa metadata debug/truy vấn, khác nhau theo nhánh | Không thêm cột riêng cho từng loại metadata — gói hết vào 1 cột JSON để giữ schema đồng nhất giữa các nhánh |

## Ví dụ `meta` theo từng nhánh

```jsonc
// Chart pretrain curriculum
{"source_chart_index": 12, "slice_start": 30, "slice_end": 55, "num_candles": 25, "num_layers": 3}

// Action/trade-simulation
{"exit_type": "tp_hit"}

// Book/PDF
{"page_number": 42, "paragraph_index": 7, "char_length": 512}
```

## Vì sao dùng `meta` JSON thay vì cột riêng

- Mỗi nhánh có metadata khác nhau hoàn toàn (chart index vs page number) —
  thêm cột riêng cho mỗi nhánh sẽ làm schema phình to và đầy `null` khi
  gộp chung các nguồn.
- `token_length=0` là giá trị hợp lệ có chủ đích (không phải bug) — nghĩa
  là "chưa tính", không phải "tính ra 0 token".

## Khi viết pipeline sinh dữ liệu mới

1. Luôn dùng đúng 4 cột trên, đúng kiểu dữ liệu (`pa.schema` tường minh,
   không để pyarrow tự suy luận kiểu theo batch — suy luận tự động dễ gây
   schema-mismatch giữa các row-group/batch khác nhau).
2. Ghi theo batch bằng `pq.ParquetWriter` + `iter_batches()` phía input
   (nếu input cũng là parquet lớn) — không load hết vào RAM. Xem các
   action doc trong `docs/data-pipeline/actions/` để lấy pattern cụ thể.
3. `source_name` luôn là tham số truyền vào hàm build, không hard-code
   trong logic — để cùng 1 pipeline code dùng lại được cho nhiều asset/nguồn
   khác nhau.
