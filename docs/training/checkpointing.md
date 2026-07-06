# Checkpointing & HuggingFace Hub Upload

File: `app/memlm/utils/checkpoint.py`.

## `save_checkpoint`

```python
save_checkpoint(
    path, model, optimizer=None, scheduler=None,
    global_step=0, chunk_idx=0, val_loss=None,
    extra=None, model_cfg=None,
)
```

- Lưu `model.state_dict()`, tuỳ chọn `optimizer`/`scheduler` state,
  `global_step`, `chunk_idx`, `val_loss`.
- **`model_cfg`** (dataclass `ModelConfig`) được lưu qua `dataclasses.asdict()`
  — đây là cơ chế để `load_model_for_inference()` (`generate.py`) tự dựng
  đúng kiến trúc lúc train mà không cần biết trước config, tránh lỗi
  `size mismatch` / `Missing key(s)` khi load lại.

## `load_checkpoint`

```python
state = load_checkpoint(path, model, optimizer=None, scheduler=None, device="cpu")
# state: {global_step, chunk_idx, val_loss, extra, model_cfg, file_order}
```

- `model.load_state_dict(ckpt["model"])` — **mặc định `strict=True`**
  (implicit). Nếu kiến trúc đổi (thêm/bớt param) so với checkpoint, load sẽ
  raise lỗi — đây là tín hiệu đúng để phát hiện checkpoint không tương
  thích, không nên bắt exception rồi bỏ qua.
- `file_order`: luôn `None` với checkpoint hiện tại (`ChunkedMixLoader`
  không track thứ tự file — xem `docs/training/data-loading.md`). Field
  này tồn tại để tương thích ngược nếu sau này có version loader khác lưu
  lại thứ tự.

## Upload / Download HuggingFace Hub

```python
hf_upload_latest(local_path, repo_id, token=None, filename="last_chunk.pt")
hf_download_latest(repo_id, local_path, token=None, filename="last_chunk.pt")
```

- **Overwrite duy nhất 1 file** (`last_chunk.pt` mặc định) mỗi chunk —
  **không giữ lịch sử checkpoint** trên Hub, tiết kiệm storage. Nếu cần
  lịch sử nhiều checkpoint, đổi `filename` theo `chunk_idx` khi gọi.
- Token ưu tiên: tham số > `env HF_TOKEN` > cache của `huggingface-cli login`.
- Cả 2 hàm **không raise** khi lỗi mạng/auth — chỉ in cảnh báo và trả
  `False`, để training **không bị gián đoạn** vì upload thất bại.
- Repo phải tồn tại sẵn trên HF trước (`huggingface-cli repo create ... --type model`).

## Checklist khi checkpoint không tương thích

Đây là rủi ro lặp lại nhiều lần qua các lần refactor (xem
`docs/conventions/known-pitfalls.md`):

1. **Đổi kiến trúc** (thêm/bớt layer, đổi `d_model`, bật/tắt router, đổi
   memory mechanism...) → checkpoint cũ **không load được**, phải train từ
   đầu (`resume_from=None`).
2. **Dead parameters** (khai báo nhưng không dùng trong `forward`, ví dụ
   `norm1`/`Wq` từng tồn tại rồi bị bỏ) → lãng phí VRAM/dung lượng
   checkpoint, và gây `strict=True` load failures nếu param đó bị xoá khỏi
   model mới nhưng checkpoint cũ vẫn còn key đó (hoặc ngược lại). Dọn dẹp
   param không dùng ngay khi phát hiện, đừng để tích luỹ qua nhiều round.
3. **Đổi tokenizer/vocab** (custom BPE mới, đổi `n_price_bins`) → vocab
   size đổi → `token_emb`/`lm_head` shape đổi → checkpoint cũ không load
   được dù kiến trúc transformer giữ nguyên.
4. **`optimizer.load_state_dict()` phục hồi LR cũ** — nếu chỉ đổi
   hyperparameter train (không đổi kiến trúc), **bắt buộc**
   `reset_lr_for_new_round=True` khi gọi `main()`/`run_pretrain()` để LR mới
   thực sự có hiệu lực (xem `docs/training/pretrain-pipeline.md`).
