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
- `compute_loss`: cross-entropy, `ignore_index=-100` (padding label),
  **`reduction="sum"`** (không phải `"mean"`) — xem mục gradient accumulation
  ngay dưới đây để hiểu vì sao.

### Gradient accumulation — chuẩn hoá theo token, không theo micro-batch

**Đã sửa 1 bug quan trọng** (xem
`docs/conventions/known-pitfalls.md`): thiết kế trước đây tính loss
`reduction="mean"` **riêng từng micro-batch** rồi mới cộng dồn gradient
qua `grad_accum` micro-batch — điều này ngầm giả định mọi micro-batch có
**cùng số token hợp lệ**. Khi số token hợp lệ (`labels != -100`) khác nhau
giữa các micro-batch (do độ dài câu khác nhau, padding khác nhau), cách
tính này cho ra "trung bình của các trung bình" (mean-of-means) thay vì
trung bình đúng theo token — làm gradient bị lệch trọng số ngầm.

Cách tính mới:

1. `compute_loss()` trả **tổng loss** (`reduction="sum"`) của 1 micro-batch,
   không chia gì cả.
2. `_run_accum_window(batches)` tính trước `total_valid_tokens` = tổng số
   token hợp lệ trên **toàn bộ** `grad_accum` micro-batch trong cửa sổ.
3. `train_one_batch()` chia `loss_sum` của **micro-batch hiện tại** cho
   `total_valid_tokens` **của cả cửa sổ** trước khi `.backward()` — đây là
   mẫu số đúng để mỗi token trong toàn cửa sổ đóng góp trọng số như nhau
   vào gradient, bất kể nó nằm ở micro-batch nào.
4. Optimizer step chỉ chạy ở micro-batch **cuối cùng** của cửa sổ
   (`is_last_in_window=True`), không còn dựa vào đếm `accum_step % grad_accum`
   như thiết kế cũ.
5. **Giá trị dùng để log** (`TrainLogger`) vẫn là loss trung bình theo token
   của **chính micro-batch đó** (`loss_sum / num_tokens_this_batch`) — khác
   với mẫu số dùng cho gradient (`total_valid_tokens` của cả cửa sổ). Đừng
   nhầm lẫn 2 mẫu số này khi sửa code — nhầm sẽ khiến số loss hiển thị lệch
   khoảng `grad_accum` lần so với giá trị thật.

Cửa sổ cuối cùng của mỗi epoch có thể **ngắn hơn** `grad_accum` (nếu số
batch còn lại không chia hết) — `train_one_chunk()` vẫn chạy nốt cửa sổ
ngắn này qua `_run_accum_window()` thay vì bỏ qua, để không mất dữ liệu.

- `evaluate()`: cũng đổi sang tính **val loss trung bình theo token** trên
  toàn bộ token hợp lệ đã sample (`total_loss_sum / total_tokens`), thay vì
  trung bình đơn giản của các batch loss — nhất quán với cách tính loss
  train ở trên.
- `train_one_chunk()`: vòng lặp train 1 chunk dữ liệu, gom micro-batch vào
  `accum_buffer` tới khi đủ `grad_accum` thì gọi `_run_accum_window()`, sau
  đó `_maybe_eval_and_save()` kiểm tra `eval_every`/`save_every` để log
  eval, lưu `best.pt` (khi val_loss cải thiện) và checkpoint định kỳ.

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

Mỗi lần `eval_every` step (và ở cuối mỗi chunk): `BaseTrainer` gọi
`run_all_ict_benchmarks()` (`app/memlm/benchmark_ict.py`) — bộ benchmark
ICT (Swept/FVG/Shift...), chứ **không phải** `run_all()` tổng quát
(`benchmark.py`, semantic/entity/fact/language/ood) như thiết kế trước đây.
Log qua `log_bench()`.

`benchmark.py::run_all()` (xem `docs/evaluation/benchmark-suite.md`) vẫn
tồn tại và hữu ích để đánh giá thủ công/so sánh checkpoint
(`python app/memlm/benchmark.py <checkpoint>`), nhưng **không còn được gọi
tự động** trong vòng lặp train — nếu muốn theo dõi cả 2 bộ benchmark song
song trong lúc train, cần tự thêm lệnh gọi `run_all()` vào
`_maybe_eval_and_save()` (hoặc `train_one_chunk()`).