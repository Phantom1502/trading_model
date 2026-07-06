# Data Loading — Incremental / RAM-safe

File: `app/memlm/dataset.py`.

## Vì sao incremental

RAM hạn chế (Colab T4) → không load toàn bộ dataset. Thay vào đó: load từng
**chunk** N sample → tokenize → train xong chunk → giải phóng → load chunk
tiếp theo.

`TOKENIZE_BATCH = 128` — tokenize streaming theo batch nhỏ 128 câu một lúc
(`encode_batch`) rồi giải phóng ngay, giữ RAM peak thấp thay vì tokenize cả
chunk 10k-20k sample cùng lúc.

## Dataset classes

- **`TokenChunkDataset`**: cắt mỗi document thành các segment độc lập
  `seg_len + 1` token (input + label lệch 1 token), phần đuôi (`tail`) được
  giữ lại nếu đủ dài (`min_tail_len=64`) bằng cách lấy `seg_len+1` token
  **cuối cùng** của document (có thể overlap với segment trước). Document
  ngắn hơn 128 token bị bỏ qua (hard-coded).
- **`SequentialDocumentDataset`**: dùng khi `cfg.data.sequential_mode=True`.
  Document được **shuffle** nhưng các segment của **cùng 1 document xếp
  tuần tự** (dùng `DataLoader(shuffle=False)`) — quan trọng nếu sau này cần
  state xuyên-segment (context memory) trong cùng 1 document.
  `stride` (mặc định = `seg_len`, có thể override qua `cfg.data.window_stride`)
  quyết định overlap giữa các window liên tiếp.

## Collate

`collate_fn`: pad theo `max_len` trong batch, `pad_id` cho input, `-100` cho
label (bị `ignore_index` trong cross-entropy bỏ qua).

## `_BaseChunkedLoader` — logic chung

```python
class _BaseChunkedLoader:
    def _load_dataset(self)         -> raises NotImplementedError  # subclass override
    def _extract_text(self, sample) -> raises NotImplementedError  # subclass override
```

- `start_chunk > 0` → `.skip(n_skip)` trên HF streaming dataset để resume
  đúng vị trí (dựa vào `chunk_size * start_chunk`).
- `__next__()`: gọi `_load_one_chunk()` → tokenize → split train/val theo
  `val_ratio` → build `DataLoader` qua `make_dataloaders()`.
- `StopIteration` khi hết dữ liệu (`exhausted`) hoặc đã đạt `total_chunks`.

### ⚠️ Stateless resume by design

`ChunkedMixLoader` (và các loader khác) **không track thứ tự file** khi
resume — chỉ model weights được khôi phục qua checkpoint. Nghĩa là resume
training **không đảm bảo thấy lại đúng thứ tự dữ liệu cũ**, chỉ đảm bảo tiếp
tục học từ đúng trạng thái model. Đây là quyết định thiết kế có chủ đích để
đơn giản hoá logic loader — nếu cần reproducibility tuyệt đối về thứ tự dữ
liệu, phải tự lưu thêm state riêng.

## Các loader cụ thể

| Loader | Nguồn | Ghi chú |
|---|---|---|
| `ChunkedWikiLoader` | `wikimedia/wikipedia` | streaming |
| `ChunkedVTSNLPLoader` | `VTSNLP/vietnamese_curated_dataset` (12.2M rows) | filter theo `domains` |
| `ChunkedParquetLoader` | 1 file/glob parquet local | `filter_fn` tuỳ chọn (không serializable → khởi tạo thủ công, xem docstring `train.py`) |
| `ChunkedMixLoader` | nhiều parquet, interleave theo tỉ lệ | dùng `datasets.interleave_datasets`, `stopping_strategy` |

### `ChunkedMixLoader` — mix nhiều nguồn

```python
cfg.data.source = "mix"
cfg.data.mix.sources = {
    "wiki_vi"  : ("/data/wiki_vi/*.parquet",   0.05),
    "wiki_en"  : ("/data/wiki_en/*.parquet",   0.30),
    "math"     : ("/data/math/*.parquet",       0.10),
    "social_vi": ("/data/social_vi/*.parquet",  0.25),
    "python"   : ("/data/python/*.parquet",     0.30),
}
cfg.data.mix.stopping_strategy = "all_exhausted"  # hoặc "first_exhausted"
```

Tổng `probabilities` phải = 1.0 (assert). `"all_exhausted"` oversample
nguồn nhỏ cho tới khi mọi nguồn hết; `"first_exhausted"` dừng ngay khi
nguồn đầu tiên hết.

## Lưu ý khi thêm nguồn dữ liệu mới

1. Subclass `_BaseChunkedLoader`, chỉ cần override `_load_dataset()` (trả
   về HF streaming dataset) và `_extract_text(sample)` (trả `str | None`).
2. Đảm bảo `_extract_text` trả `None` (không raise) cho sample thiếu field
   — loop `_load_one_chunk` chỉ skip, không crash.
3. Nếu cần filter phức tạp không serializable qua config (lambda, closure),
   khởi tạo loader **thủ công** rồi truyền vào `run_pretrain(data_loader_gen=...)`
   thay vì đi qua `main()` — xem ví dụ trong docstring `train.py`.
