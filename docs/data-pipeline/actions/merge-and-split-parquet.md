# Action: Merge & Split Parquet (tổng hợp dataset cuối cùng)

Files: `app/utils/chart/book_and_output_pipeline.py` → `OutputPipeline`
(khuyến nghị, có shuffle/split/stats), `merge_parquet.py` (script gom file
theo dung lượng, không shuffle/split), `mix_data_parquet.py` (mix nhiều
category theo tỉ lệ trước khi train, dùng cho local file lớn ngoài
`ChunkedMixLoader`).

## OutputPipeline — merge + shuffle + split train/val

```python
from app.utils.chart.book_and_output_pipeline import OutputPipeline

out = OutputPipeline(seed=42)
result = out.merge_and_split(
    input_paths = ["data/pretrain.parquet", "data/action.parquet", "data/books.parquet"],
    output_dir  = "data/final",
    val_ratio   = 0.05,
    min_token_length = 0,   # > 0 để lọc sample quá ngắn
)
# result = {"train": N, "val": M, "total": N+M}
```

- Đọc toàn bộ record vào RAM (`_load_all`) → **chỉ phù hợp khi tổng dữ
  liệu vừa với RAM**. Với dữ liệu rất lớn, dùng `ChunkedMixLoader`
  (streaming, xem `docs/training/data-loading.md`) thay vì pipeline này.
- Shuffle bằng `random.Random(seed)` — **cố định seed** để tái lập được
  train/val split giữa các lần chạy.
- Ghi kèm `stats.json`: phân phối theo `source`, thống kê `token_length`
  (mean/min/max/total) cho cả train và val — dùng để kiểm tra nhanh dataset
  có bị lệch nguồn hay không trước khi train.
- `preview(parquet_path, n)`: xem nhanh N record đầu để sanity-check thủ
  công trước khi train.

## `merge_dir` — chỉ gom file, không shuffle

Dùng khi đã có nhiều file `.parquet` nhỏ (part files) trong 1 thư mục và
chỉ cần gom lại (không cần shuffle/split):

```python
out.merge_dir("data/parts/", "data/merged.parquet", target_size_mb=500)
```

`target_size_mb=None` → gom thành 1 file; nếu chỉ định → chia thành nhiều
file `~target_size_mb` (dùng khi 1 file quá lớn gây khó xử lý ở downstream).

## `mix_data_parquet.py` — mix nhiều category theo tỉ lệ (offline, trước khi train)

Dùng khi bạn muốn **tạo sẵn** 1 bộ file mix theo tỉ lệ cố định (thay vì mix
on-the-fly lúc train qua `ChunkedMixLoader`):

```python
CATEGORY_CONFIG = {
    "vi":        {"files": [...], "ratio": 0.10},
    "wiki_en":   {"files": [...], "ratio": 0.20},
    "python":    {"files": [...], "ratio": 0.30},
    "math":      {"files": [...], "ratio": 0.15},
    "social_en": {"files": [...], "ratio": 0.10},
    "trading":   {"files": [...], "ratio": 0.15},
}
```

- Thuật toán: mỗi vòng chọn category **đang thiếu tỉ lệ nhất** (deficit lớn
  nhất so với ratio mục tiêu), đọc tối đa `INTERLEAVE_TOKENS` từ nó, rồi
  chọn lại — nhờ vậy 1 part-file `PART_FLUSH_TOKENS` (~50M token) chứa
  **nhiều đoạn nhỏ xen kẽ** nhiều category, thay vì 1 cục thuần 1 loại.
- **Checkpoint có thể resume**: `checkpoint.json` lưu `file_idx`,
  `row_group_idx` theo từng category — chạy lại script sẽ tiếp tục đúng vị
  trí thay vì làm lại từ đầu (quan trọng vì xử lý hàng trăm GB có thể mất
  nhiều giờ/nhiều lần chạy).
- Ghi atomic (`os.replace`) để checkpoint không hỏng nếu crash giữa chừng.
- `CHUNK_TOKEN_LIMIT = 1_000_000_000` (1B token/file gộp) — mỗi khi đạt
  ngưỡng này, đóng "chunk" hiện tại (thư mục `chunk_XXXXX/`) và bắt đầu
  chunk mới.

### Khi thêm category mới vào `mix_data_parquet.py`

1. Thêm entry vào `CATEGORY_CONFIG`, đảm bảo **tổng ratio vẫn = 1.0**
   (assert sẽ raise nếu sai).
2. Nếu đã có `checkpoint.json` cũ (chưa hoàn tất mix trước đó), **xoá hoặc
   backup checkpoint cũ** trước khi thêm category — state cũ không biết về
   category mới, dễ gây lệch tỉ lệ ngầm.
