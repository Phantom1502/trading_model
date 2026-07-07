# Pretraining Pipeline

Files: `app/memlm/train.py`, `app/memlm/trainer/pretrain.py`,
`app/memlm/trainer/base.py`.

## Chạy train

Từ **thư mục gốc project** (xem `docs/conventions/running-from-root.md`
cho quy tắc import/chạy đầy đủ), có 3 cách tương đương:

```bash
# Chạy trực tiếp bằng đường dẫn file
python app/memlm/train.py

# Hoặc chạy như module (cách chuẩn nhất)
python -m app.memlm.train
```

hoặc trong notebook (Colab T4):

```python
from app.memlm.config import get_100m_config
from app.memlm.train import main

cfg = get_100m_config()
main(cfg)
```

`main(cfg, start_chunk=0, reset_lr_for_new_round=False)`:
1. Load tokenizer, set `cfg.model.vocab_size = tokenizer.vocab_size`.
2. Build model (`build_model(cfg)`).
3. Dựng data loader incremental theo `cfg.data.source`
   (`"wikipedia" | "vtsnlp" | "parquet" | "mix"`).
4. Gọi `run_pretrain(...)`.

## BaseTrainer (`trainer/base.py`)

- Optimizer: `AdamW(lr=cfg.train.lr, weight_decay, betas=(0.9, 0.95))`.
- Scheduler: warmup tuyến tính → cosine annealing **with warm restarts**
  (SGDR) theo chu kỳ `lr_decay_cycle_steps`, sàn `lr_min_ratio`.
- Mixed precision qua `torch.amp.GradScaler` (bật khi CUDA + `mixed_precision=True`).
- **Gradient accumulation**: loss chia cho `grad_accum` trước khi backward,
  optimizer step chỉ chạy mỗi `grad_accum` accum-step.
- `compute_loss`: cross-entropy mặc định, `ignore_index=-100` (padding label).
- Aux loss: `loss = lm_loss + mod_aux_loss_coef * aux_loss` — coef đọc từ
  `cfg.model.mod_aux_loss_coef` (mặc định 0.001 nếu không set).
- `evaluate()`: chạy tối đa `max_batches=50` batch val, không backward.
- `train_one_chunk()`: vòng lặp train 1 chunk dữ liệu — log định kỳ
  (`log_every`), eval + benchmark định kỳ (`eval_every`), lưu checkpoint
  định kỳ (`save_every`) và mỗi khi val_loss cải thiện (`best.pt`).

## PretrainTrainer (`trainer/pretrain.py`)

Kế thừa nguyên `BaseTrainer` — pretrain không cần loss khác cross-entropy.

`run_pretrain(cfg, model, tokenizer, data_loader_gen, start_chunk, reset_lr_for_new_round)`:
- Resume từ `cfg.train.resume_from` nếu có: load `global_step`,
  `best_val_loss`, và `chunk_idx` (nếu `start_chunk=0` thì lấy tiếp từ
  checkpoint).
- Với mỗi `(train_loader, val_loader)` sinh ra từ `data_loader_gen`:
  train 1 chunk → lưu checkpoint → (tuỳ chọn) upload lên HuggingFace Hub.

## Resume / train round mới

**Trường hợp 1 — chỉ đổi hyperparameter train (lr, lr_decay_cycle_steps...),
giữ nguyên kiến trúc:**

```python
cfg.train.resume_from = "checkpoints/chunk_33.pt"
cfg.train.lr = 1e-4
main(cfg, start_chunk=34, reset_lr_for_new_round=True)
```

`reset_lr_for_new_round=True` là **bắt buộc** nếu muốn `cfg.train.lr` mới có
hiệu lực thật — nếu không, `optimizer.load_state_dict()` sẽ phục hồi
nguyên `lr` cũ đã lưu trong checkpoint (xem
`docs/conventions/known-pitfalls.md`). Khi bật cờ này, optimizer được tạo
lại từ đầu và scheduler được "tua" tới đúng `global_step` hiện tại
(`for _ in range(trainer.global_step): trainer.scheduler.step()`).

**Trường hợp 2 — đổi kiến trúc (`num_slots`, `d_model`, `use_memory`,
`use_router`, số layer...):**

```python
cfg.train.resume_from = None   # KHÔNG thể resume
main(cfg, start_chunk=0)
```

## Data source qua `cfg.data.source`

| source | Cấu hình cần thiết |
|---|---|
| `"wikipedia"` | `dataset_name`, `dataset_subset` (mặc định `wikimedia/wikipedia`, `20231101.vi`) |
| `"vtsnlp"` | tuỳ chọn `cfg.data.vtsnlp_domains = ["Science", ...]` để lọc domain |
| `"parquet"` | `cfg.data.parquet_path`, `cfg.data.parquet_text_col` |
| `"mix"` | `cfg.data.mix.sources = {name: (glob_pattern, prob)}`, `stopping_strategy` |

Chi tiết loader: xem `docs/training/data-loading.md`.

## Benchmark tích hợp trong training loop

Mỗi lần `eval_every` step: chạy `run_all()` (`benchmark.py`) — 5 chiều
semantic/entity/fact/language/ood, trọng số weighted → `total`. Log qua
`log_bench()`. Xem `docs/evaluation/benchmark-suite.md`.