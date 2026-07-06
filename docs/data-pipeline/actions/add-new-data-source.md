# Action: Thêm nguồn dữ liệu mới vào pretraining

Checklist tổng hợp khi muốn thêm 1 loại dữ liệu mới (nguồn HF streaming,
file local, hoặc pipeline sinh dữ liệu tự custom) vào vòng lặp pretrain.

## Bước 1 — Sinh dữ liệu (nếu chưa có sẵn ở dạng parquet chuẩn)

Xác định dữ liệu thuộc loại nào và dùng đúng action tương ứng:

| Loại dữ liệu | Dùng action |
|---|---|
| Giá OHLC thô (CSV) → chart token | `docs/data-pipeline/actions/build-chart-dataset.md` |
| Chart token → text dạy đọc hiểu chart | `docs/data-pipeline/actions/build-pretrain-curriculum.md` |
| Chart + hành động giao dịch → outcome | `docs/data-pipeline/actions/build-action-dataset.md` |
| Sách/PDF trading | `docs/data-pipeline/actions/build-book-dataset.md` |
| Nguồn text tổng quát khác (đã có sẵn ở HF hoặc file local) | xem bước 2 |

**Mọi output đều phải theo đúng schema chung** — xem
`docs/conventions/parquet-schema.md`. Đây là điều kiện để gộp chung được
với các nguồn khác qua `OutputPipeline`/`mix_data_parquet.py`.

## Bước 2 — Nếu là nguồn streaming từ HuggingFace (giống Wikipedia/VTSNLP)

Subclass `_BaseChunkedLoader` trong `app/memlm/dataset.py`:

```python
class ChunkedMyNewSourceLoader(_BaseChunkedLoader):
    def _load_dataset(self):
        return load_dataset("my/dataset-name", split="train", streaming=True)

    def _extract_text(self, sample):
        return sample.get("text", "").strip() or None
```

Thêm case tương ứng vào `if/elif` trong `app/memlm/train.py::main()` và
`app/memlm/trainer/pretrain.py::run_pretrain()` (2 nơi phải đồng bộ — xem
`data.source == "mix"` làm ví dụ nhánh đã thêm gần nhất).

## Bước 3 — Nếu là file/glob parquet local

Không cần code mới — dùng trực tiếp:

```python
cfg.data.source           = "parquet"
cfg.data.parquet_path     = "data/my_new_source.parquet"
cfg.data.parquet_text_col = "text"   # đổi nếu cột tên khác
```

Hoặc nếu cần lọc theo metadata (`filter_fn` không serializable qua config):

```python
from dataset import ChunkedParquetLoader
loader = ChunkedParquetLoader(
    cfg, tokenizer, "data/my_new_source.parquet",
    filter_fn=lambda s: s.get("genre") == "Lịch sử",
)
run_pretrain(cfg, model, tokenizer, data_loader_gen=loader)
```

## Bước 4 — Nếu muốn mix theo tỉ lệ cố định với các nguồn khác

Hai lựa chọn (xem `docs/data-pipeline/actions/merge-and-split-parquet.md`):

- **Mix on-the-fly lúc train**: thêm vào `cfg.data.mix.sources`, dùng
  `cfg.data.source = "mix"` (`ChunkedMixLoader`).
- **Mix offline trước khi train** (tạo sẵn file đã mix theo tỉ lệ, dùng khi
  dữ liệu quá lớn/nhiều category): thêm vào `CATEGORY_CONFIG` trong
  `mix_data_parquet.py`, đảm bảo tổng `ratio` vẫn = 1.0.

## Bước 5 — Kiểm tra trước khi train full

1. Chạy thử với `cfg.data.chunk_size` nhỏ + `cfg.train.total_chunks=1` để
   xác nhận loader không lỗi và text hợp lệ.
2. Nếu nguồn có price token (`O_/H_/L_/C_`) trộn cùng text tổng quát,
   **luôn** dùng `strict_chart_mode=True` ở tokenizer — xem
   `docs/model/tokenizer.md`.
3. Chạy `OutputPipeline.preview()` hoặc đọc thử vài dòng bằng
   `pd.read_parquet(...).head()` để sanity-check nội dung + `token_length`
   trước khi đưa vào vòng lặp train dài.
4. Nếu nguồn mới làm tổng effective batch size lệch khỏi mốc ~1M
   token/step, xem lại `docs/conventions/known-pitfalls.md` — batch nhỏ
   gây val loss noisy.
