# Benchmark Suite tổng quát (semantic / entity / fact / language / ood)

File: `app/memlm/benchmark.py`.

## Tổng quan

```python
from benchmark import run_all
from generate import load_model_for_inference

model, tokenizer, cfg = load_model_for_inference("checkpoints/best.pt")
results = run_all(model, tokenizer, cfg, verbose=True)
```

5 chiều độc lập, tổng hợp có trọng số:

```
TOTAL = semantic*0.20 + entity*0.20 + fact*0.30 + language*0.20 + ood*0.10
```

## Cơ chế chấm điểm log-prob (áp dụng cho semantic/entity/fact/ood)

```python
avg_logprob_per_token(model, tokenizer, prompt, completion, device, max_seq)
```

- Tính avg log-prob/token của `completion` khi nối sau `prompt`.
- Nếu `prompt + completion` vượt `max_seq`, cắt bớt **đầu prompt** (giữ
  toàn bộ completion) — completion luôn được giữ nguyên vẹn để tính đúng.
- `score_item()`: so `pos_mean` (trung bình log-prob các completion đúng)
  với `neg_mean` (completion sai) → `score = pos_mean - neg_mean`.
  **`score > 0`** nghĩa là model gán xác suất cao hơn cho câu đúng — đây là
  tiêu chí "pass" cho từng item, không phải giá trị tuyệt đối của log-prob.

## 4 bộ benchmark log-prob

| Bench | Đo gì | Ví dụ |
|---|---|---|
| `SEMANTIC_BENCH` | Phân loại phạm trù đúng (là gì) | "Con mèo là" → đúng: động vật có vú; sai: loài chim |
| `ENTITY_BENCH` | Kiến thức thực thể cụ thể (ai, ở đâu) | "Albert Einstein là" → nhà vật lý, không phải nhà văn |
| `FACT_BENCH` | Sự kiện/số liệu chính xác | "Thủ đô của Việt Nam là" → " Hà Nội" |
| `OOD_BENCH` | Suy luận ngoài phân phối — cả khái niệm hiện đại (blockchain, vaccine) và **suy luận logic với danh từ giả** ("Mọi flar đều là zent...") | kiểm tra model có suy luận theo cấu trúc câu chứ không chỉ pattern-match từ khoá quen thuộc |

`OOD_BENCH` đặc biệt hữu ích để phát hiện model có đang **overfit vào từ
khoá quen thuộc** thay vì học cấu trúc suy luận — các câu dùng danh từ bịa
(`flar`, `zent`, `nori`, `drako`...) buộc model phải dựa vào **quan hệ ngữ
pháp/logic** trong câu, không thể dựa vào việc đã thấy từ đó trong corpus.

## Language quality (không dùng log-prob)

`run_language_benchmark()`: sinh text tự do (`_generate_sample`, sampling
với temperature/top_k) từ `LANGUAGE_PROMPTS`, đo:

```
language_score = (distinct1 + distinct2) / 2 - repeat_ratio
```

- `distinct1`/`distinct2`: tỉ lệ token/bigram **duy nhất** trên tổng số —
  cao = đa dạng từ vựng, thấp = lặp từ nhiều.
- `repeat_ratio`: tỉ lệ token giống hệt token ngay trước nó (lặp liên tiếp)
  — dấu hiệu điển hình của model bị **mode collapse** hoặc chưa train đủ.

## Chạy trong training loop

`BaseTrainer` gọi `run_all(..., verbose=False, step=global_step)` mỗi
`eval_every` step, log qua `log_bench()` — dùng để theo dõi **trend** qua
các step, không chỉ nhìn val_loss đơn thuần (val_loss thấp không đảm bảo
model output chất lượng — ví dụ mode collapse có thể giữ loss thấp nhưng
`language_score` sẽ tệ).

## So sánh nhiều checkpoint

```bash
python benchmark.py checkpoints/chunk_10.pt checkpoints/chunk_50.pt
```

`compare_checkpoints()` load tuần tự từng checkpoint, chạy `run_all`, in
bảng so sánh, và **giải phóng GPU memory** (`del model` + `torch.cuda.empty_cache()`)
giữa các lần load — quan trọng khi so sánh nhiều checkpoint lớn trên GPU
hạn chế VRAM (T4).

## Khi thêm 1 dạng benchmark mới

1. Thêm `BenchItem(prompt, positive, negative, note)` vào 1 list mới hoặc
   list có sẵn — `positive`/`negative` nên có **nhiều biến thể diễn đạt**
   (không chỉ 1 câu) để trung bình hoá nhiễu.
2. Nếu là 1 *chiều* hoàn toàn mới (không phải thêm item vào chiều cũ),
   thêm trọng số vào `WEIGHTS` và cộng vào công thức `TOTAL` trong
   `run_all()` — nhớ giữ tổng trọng số = 1.0.
