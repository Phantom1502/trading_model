# HellaSwag Benchmark

File: `app/memlm/benchmark_hellaswag.py`.

## Mục đích và giới hạn quan trọng

HellaSwag là benchmark **tiếng Anh**. Model của project này train chủ yếu
tiếng Việt → accuracy sẽ **thấp hơn nhiều** so với model tiếng Anh cùng
kích thước. Baseline ngẫu nhiên (4 lựa chọn) là **25%**.

**→ Dùng để theo dõi TREND qua các checkpoint của cùng 1 model, không dùng
để so sánh tuyệt đối với GPT-2/LLaMA hay các model tiếng Anh khác.**

## Cách chấm

Zero-shot: với mỗi item (`ctx` + 4 `endings`), tính
`avg_logprob_per_token(ctx, ending)` cho cả 4 ending, chọn ending có
log-prob cao nhất, so với `label` đúng.

```python
scores = [avg_logprob_per_token(model, tokenizer, item["ctx"], ending, device, max_seq)
          for ending in item["endings"]]
predicted = scores.index(max(scores))
correct = (predicted == item["label"])
```

## Kết quả tham chiếu đã ghi nhận

```
31% accuracy tại ~24k step, ~800M token đã train
```

So sánh: GPT-2 124M đạt ~29.4% sau **40B token** (tiếng Anh, cùng benchmark
tiếng Anh). Model 110M của project này đạt 31% chỉ với **~800M token**
(~1/50 lượng token) dù train chủ yếu tiếng Việt — cho thấy **data-efficient
hơn đáng kể** ở mốc so sánh này. Lưu ý: đây là so sánh không hoàn toàn công
bằng (khác ngôn ngữ chính, khác kiến trúc/tokenizer) — chỉ nên dùng làm tín
hiệu định hướng, không phải kết luận khoa học chặt chẽ.

## Cách chạy

```bash
python app/memlm/benchmark_hellaswag.py checkpoints/chunk_10.pt
python app/memlm/benchmark_hellaswag.py ckpt_10.pt ckpt_50.pt --n-samples 200
```

- `--n-samples`: mặc định 10000 khi chạy CLI trực tiếp — giảm xuống
  (vd 200) khi cần chạy nhanh trong vòng lặp so sánh nhiều checkpoint.
- `hellaswag_score_for_run_all()`: hàm rút gọn (`n_samples=200`,
  `verbose=False`) để tích hợp vào `benchmark.py::run_all()` nếu muốn thêm
  HellaSwag như 1 chiều nữa trong benchmark tổng — **hiện tại chưa được
  gọi trong `run_all()`**, đang là benchmark độc lập chạy riêng.

## Khi nào nên chạy full 10000 samples vs n nhỏ

- Theo dõi nhanh mỗi vài checkpoint trong lúc train dài: `n_samples=200`
  (đủ để thấy trend, tiết kiệm thời gian mỗi lần eval).
- Báo cáo kết quả cuối cùng / so sánh chính thức giữa các round train:
  chạy full hoặc `n_samples` lớn (vài nghìn) để giảm nhiễu do cỡ mẫu nhỏ.