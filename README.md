# MemoryLM

Repo gồm 2 phần độc lập:

1. **`app/memlm/`** — Mô hình ngôn ngữ tiếng Việt kiến trúc LLaMA-style, tích
   hợp một **Context Memory (M)** xuyên-segment (Read/Write cross-attention)
   để mở rộng "trí nhớ" ngoài cửa sổ ngữ cảnh thông thường.
2. **`app/utils/`** — Bộ công cụ encode dữ liệu nến (OHLC) thành chuỗi token
   text (FSQ codec) để dùng làm dữ liệu trading cho LLM, gồm cả script tạo
   dataset từ CSV giá thô.

---

## 1. `app/memlm/` — MemoryLM

### Kiến trúc (LLaMA-style)

| Thành phần      | Thay thế cho                | Ghi chú |
|------------------|------------------------------|---------|
| RMSNorm + Pre-Norm | LayerNorm + Post-Norm       | `model/layers.py` |
| SwiGLU FFN        | Linear-GELU-Linear           | `model/layers.py` |
| RoPE               | Absolute position embedding  | áp lên Q/K trong self-attention |
| No bias            | Linear có bias                | toàn bộ Linear trong block |
| Scaled init         | init cố định                  | theo `1/sqrt(2*n_layers)` cho các projection nằm trên residual |

### Context Memory (M)

Mỗi `MemoryBlock` (`model/block.py`) có một bộ nhớ `M` dạng `(B, num_slots, d_model)`,
cập nhật bằng EMA với `alpha` **cố định** (suy ra từ `half_life`):

```
PHA READ  : hiện tại hỏi quá khứ      → m_out = CrossAttn(Q=token, K=M, V=M)
PHA WRITE : quá khứ tra vấn hiện tại  → Q' = CrossAttn(Q=M, K=token, V=token)
M_new = alpha * M_old + (1 - alpha) * Q'
```

`alpha` cố định nên `detach_memory()` ngay sau mỗi batch là an toàn (đã verify
thực nghiệm) — không cần BPTT window phức tạp.

### Tokenizer

`tokenizer.py` bọc PhoBERT BPE (`vinai/phobert-base`, vocab ~64k, **không có
bản Fast**) và cộng thêm một dải vocab riêng cho **price token**
(`O_<bin>`, `H_<bin>`, `L_<bin>`, `C_<bin>`, `<chart>`, `</chart>`) dùng khi
train lẫn dữ liệu trading (xem mục 2). Bật `strict_chart_mode=True` trong
`TokenizerConfig` nếu trộn dữ liệu trading với Wikipedia/VTSNLP, để tránh
match nhầm các ký hiệu khoa học tự nhiên (`H_0`, `C_1`, `O_157`, ...) với
price token.

### Dữ liệu

`dataset.py` load incremental (RAM thấp) theo từng chunk, hỗ trợ 2 nguồn:

- `ChunkedWikiLoader` — `wikimedia/wikipedia` (`20231101.vi`)
- `ChunkedVTSNLPLoader` — `VTSNLP/vietnamese_curated_dataset` (12.2M rows,
  có thể lọc theo `domain`)

Chọn nguồn qua `cfg.data.source = "wikipedia" | "vtsnlp"`.

### Training

```bash
pip install -r requirements.txt
cd app/memlm
python train.py
```

```python
from config import get_100m_config
from train import main

cfg = get_100m_config()
main(cfg)
```

**Resume / train round mới** — đọc kỹ docstring trong `train.py` và
`trainer/pretrain.py`, đặc biệt cờ `reset_lr_for_new_round`:

- Chỉ đổi hyperparameter train (lr, lr_decay_cycle_steps, ...), giữ nguyên
  kiến trúc → resume được, **bắt buộc** `reset_lr_for_new_round=True` nếu
  muốn `cfg.train.lr` mới thực sự có hiệu lực (nếu không, optimizer/scheduler
  sẽ phục hồi nguyên lr cũ từ checkpoint).
- Đổi kiến trúc (`num_slots`, `d_model`, `use_memory`, ...) → **không thể**
  resume, phải train lại từ đầu với `resume_from=None`.

### Inference

```python
from generate import load_model_for_inference, generate

model, tokenizer, cfg = load_model_for_inference("checkpoints/best.pt")
print(generate(model, tokenizer, cfg, "Trí tuệ nhân tạo là"))
```

`load_model_for_inference` tự đọc `model_cfg` đã lưu trong checkpoint để
build đúng kiến trúc lúc train, tránh lỗi `size mismatch` / `Missing key(s)`.

### Mở rộng vocab tuỳ chọn

`scripts/add_custom_tokens.py` — thêm token chuyên ngành trading
(`scripts/example_trading_tokens.txt`) vào **một bản tokenizer riêng**
(không sửa tokenizer gốc, tránh lệch vocab với checkpoint cũ):

```bash
python scripts/add_custom_tokens.py \
    --new-tokens-file scripts/example_trading_tokens.txt \
    --output-dir custom_tokenizer
```

Cách này là cách **cũ** (qua `add_tokens()`); cách **mới** khuyến nghị là
dùng price-token-vocab riêng có sẵn trong `tokenizer.py` (không cần script
này, không cần train lại tokenizer).

---

## 2. `app/utils/` — Chart → Token Codec (dữ liệu trading)

Encode chuỗi nến OHLC thành token text để LLM học, và ngược lại.

- **`chartcodec.py`**
  - `calculate_atr(df, period)` — ATR (EMA-smoothed).
  - `ChartCodec` — FSQ (Finite Scalar Quantization) 1024 bin/kênh, chuẩn hoá
    giá theo `(price - anchor_open) / (scale * anchor_atr)`, rời rạc hoá độc
    lập từng kênh O/H/L/C. `encode_window()` / `decode_window()` cho 1 cửa
    sổ nến.
  - Hằng số `*_SCALE` (M1/M5/M15/H1/D1) — hệ số scale ứng với từng khung
    thời gian, suy ra bằng phân tích thống kê (`chart_data_scale_factor.py`,
    percentile 99.9% của tỉ lệ dạt xa ATR).

- **`chartdatasetbuilder.py`** — `ChartDatasetBuilder` quét toàn bộ lịch sử
  giá theo cửa sổ trượt (`window_size`, `stride`), dùng `ChartCodec` để sinh
  `DataFrame`/`.parquet` (`start_idx`, `end_idx`, `anchor_open`, `anchor_atr`,
  `text`) — sẵn sàng đưa vào `tokenizer.encode_batch(df["text"].tolist())`.

- **`chart_generate_ds.py`** — script chạy nhanh, ví dụ:

  ```python
  codec = ChartCodec(scale=M1_SCALE)
  builder = ChartDatasetBuilder(codec, window_size=100, stride=10, atr_period=100)
  dataset = builder.build_from_file("data/XAUUSD_1Min.csv", "data/chart_dataset.parquet")
  ```

- **`chart_data_scale_factor.py`** — script độc lập tính lại hằng số
  `SCALE` cho một khung thời gian mới từ dữ liệu lịch sử thật (không phải
  thư viện, chạy trực tiếp bằng `python chart_data_scale_factor.py`).

> Quy ước quan trọng: cửa sổ lấy **forward** từ thời điểm neo `t` (không
> lookahead bias), anchor (`anchor_open`, `anchor_atr`) lấy tại đúng `t` và
> **phải lưu lại cùng mẫu** vì token chỉ mang thông tin tương đối, không tự
> suy ra được giá trị thật khi decode nếu thiếu 2 giá trị này.

---

## Cài đặt

```bash
pip install -r requirements.txt
```

`requirements.txt` gồm:
- `torch`, `transformers`, `datasets`, `accelerate` — cho `app/memlm`
- `pandas`, `numpy`, `pyarrow` — cho `app/utils` (pyarrow cần để đọc/ghi
  `.parquet`)

GPU (CUDA) là tuỳ chọn nhưng khuyến nghị cho training — code tự detect qua
`torch.cuda.is_available()`.
