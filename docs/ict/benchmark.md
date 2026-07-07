# ICT Benchmark

> Xem lưu ý về nguồn thông tin ở đầu `docs/ict/detectors.md` — nội dung
> chi tiết bên dưới vẫn tổng hợp từ ghi chú thiết kế, chưa đối chiếu trực
> tiếp với source `benchmark_ict.py`. **Đã xác nhận qua code thật**: điểm
> tích hợp vào training loop và chữ ký hàm public (xem mục ngay dưới).

File: `app/memlm/benchmark_ict.py`.

## Tích hợp vào training loop (đã xác nhận qua `trainer/base.py`)

```python
from app.memlm.benchmark_ict import run_all_ict_benchmarks

bench = run_all_ict_benchmarks(model, tokenizer, cfg, verbose=False, step=global_step)
```

`BaseTrainer` gọi hàm này mỗi `eval_every` step và ở cuối mỗi chunk
(`train_one_chunk()`), thay cho `benchmark.py::run_all()` như thiết kế
trước đây — xem `docs/training/pretrain-pipeline.md`. Chữ ký hàm khớp
pattern chung của `log_bench()` (`utils/logger.py`): trả về `dict` các
điểm số dạng `{tên_metric: giá_trị_float}`.

## Framework

Dùng chung triết lý với `app/memlm/benchmark.py`
(`docs/evaluation/benchmark-suite.md`): so log-prob trung bình/token giữa
completion đúng và completion sai, dựa trên `BenchItem` /
`avg_logprob_per_token`.

```python
avg_logprob_per_token(model, tokenizer, prompt, completion, device, max_seq)
```

- Ground truth (completion "đúng") luôn lấy từ output của `build_facts()`
  (detector thật chạy trên dữ liệu thật) — **không viết tay**. Đây là điểm
  khác biệt quan trọng so với benchmark tổng quát (`benchmark.py`), nơi
  `BenchItem` positive/negative được viết tay vì đó là kiến thức tổng quát
  cố định, không phải output của 1 hệ thống có thể chạy lại được.

## Kết quả checkpoint đầu tiên (mốc tham chiếu)

```
avg_score: +0.071
Swept: 100%
FVG:    95%
Shift:  80%
```

## Phát hiện quan trọng: single-event vs multi-event template

Sample dạng **"full" template** (chỉ mô tả 1 sự kiện/pattern) cho điểm số
**thấp hơn nhiều** so với sample dạng **"short" template** (nhiều sự kiện
gộp lại). Nguyên nhân nhiều khả năng là **mất cân bằng phân phối lúc
train** — nếu tập train có nhiều sample multi-event hơn, model học tốt
pattern đó hơn và kém hơn ở single-event.

### Việc cần làm (đang cân nhắc, chưa quyết định dứt khoát)

Thêm breakdown theo `template_mode` vào `benchmark_ict.py` để tách riêng
điểm số single-event vs multi-event — giúp chẩn đoán chính xác vấn đề mất
cân bằng thay vì chỉ nhìn 1 con số `avg_score` gộp chung. Đánh đổi: thêm
breakdown ngay bây giờ giúp debug sớm, nhưng có thể trì hoãn vòng train
tiếp theo — cân nhắc theo mức độ ưu tiên hiện tại giữa "hiểu rõ vấn đề" và
"tiếp tục train để có nhiều checkpoint hơn mà so sánh".

## Khi thêm loại pattern mới vào benchmark

1. Đảm bảo detector cho pattern đó đã có test tích hợp qua `build_facts()`
   (xem `docs/ict/detectors.md`) trước khi dùng làm ground truth benchmark
   — benchmark dựa trên detector lỗi sẽ cho điểm số sai lệch một cách âm
   thầm, khó phát hiện hơn lỗi crash.
2. Theo dõi riêng theo `template_mode` (full/short) ngay từ đầu nếu quyết
   định thêm breakdown ở trên — tránh phải làm lại khi phát hiện mất cân
   bằng phân phối tương tự pattern cũ.