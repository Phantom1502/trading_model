# app/llama/ — Nhánh pretrain dùng HF `LlamaForCausalLM` chuẩn

Nhánh song song với `app/memlm/` (kiến trúc tự viết). Mục tiêu: pretrain
xong có thể cắm thẳng vào TRL (`SFTTrainer`/`DPOTrainer`/`RewardTrainer`)
cho các giai đoạn sau mà không phải tự viết loss masking/log-ratio tay.

## Khác biệt cốt lõi so với `app/memlm/`

| | `app/memlm/` | `app/llama/` |
|---|---|---|
| Kiến trúc | Tự viết (`RMSNorm`/`RoPE`/`SwiGLU`/`DepthRouter`) | `transformers.LlamaForCausalLM` gốc |
| Attention | Full MHA, không KV-cache | GQA (`num_key_value_heads=3`) + KV-cache |
| Tokenizer | Wrapper tự chế, price token cộng ID **ảo** ngoài tokenizer thật | `PreTrainedTokenizerFast` thật, price token `add_tokens()` **thật** |
| Định dạng price token | `O_512` (trần) | `<px_O_512>` (an toàn hơn, xem lý do trong `tokenizer.py`) |
| Checkpoint | `torch.save(state_dict)` | `model.save_pretrained()` — load thẳng bằng `from_pretrained()` |
| generate() | Vòng lặp forward lại toàn bộ sequence mỗi step | `model.generate()` (KV-cache) |
| SFT/DPO | Phải tự viết | TRL dùng thẳng (xem `train_sft_example.py`) |

## File

- `config.py` — `LlamaConfig` (~110-125M, GQA) + `DataConfig`/`TrainConfig`/`TokenizerConfig`.
- `tokenizer.py` — build tokenizer thật (base BPE + `add_tokens()` price vocab).
  Có `convert_legacy_price_tokens()` để chuyển dữ liệu trading cũ (`O_512 ...`)
  sang định dạng mới (`<px_O_512>`) khi load, **không cần sửa** `app/utils/chart/*`.
- `model.py` — `build_model(cfg, tokenizer)`, tự `resize_token_embeddings`.
- `dataset.py` — loader streaming RAM-safe (giữ nguyên logic `app/memlm/dataset.py`),
  tokenize qua tokenizer HF thật, có `attention_mask`.
- `trainer.py` — loop pretrain, checkpoint format HF, cosine warm-restart LR giống cũ.
- `train.py` — entry point, convention giống `app/memlm/train.py`.
- `benchmark.py` — bench semantic/entity/fact/language/ood, tái dùng dữ liệu
  bench từ `app/memlm/benchmark.py`, viết lại tầng gọi model theo HF convention.
- `train_sft_example.py` — khung sườn SFT bằng TRL (multi-turn, ChatML).

## Cách dùng nhanh

```bash
cd app/llama
python tokenizer.py ../memlm/custom_tokenizer custom_tokenizer_llama   # 1 lần
python train.py
```

```python
from config import Config
from train import main
main(Config())
```

## Việc CẦN LÀM TIẾP (chưa nằm trong nhánh này)

1. **`benchmark_ict.py`** (đánh giá Swept/FVG/Shift) — không có trong tài
   liệu cung cấp nên chưa rewrite được. Khi làm, cần: (a) bỏ `_split_segments`/
   `causal_mask` thủ công, dùng HF convention như `benchmark.py` ở đây; (b)
   `convert_legacy_price_tokens()` trước khi encode vì ground-truth completion
   dạng `O_512` giờ phải là `<px_O_512>`; (c) `max_seq` lấy từ
   `cfg.llama.max_position_embeddings`.
2. **Migrate dữ liệu đã sinh sẵn** (`data/*.parquet` từ `app/utils/chart/`) —
   không cần sửa file sinh dữ liệu, chỉ cần bật `text_transform=convert_legacy_price_tokens`
   khi load (đã tự động khi `cfg.data.source in ("parquet", "mix")`).
3. **Train tokenizer base mới nếu muốn tối ưu lại vocab** — có thể dùng thẳng
   `custom_tokenizer` hiện có của `app/memlm/`, không bắt buộc train lại.
4. **SequenceRouter/DepthRouter** — không migrate được thẳng sang
   `LlamaDecoderLayer` gốc; nếu vẫn muốn hướng skip/run theo sequence, cần
   subclass `LlamaModel` riêng (ngoài phạm vi nhánh này).
5. **Train lại từ đầu** — checkpoint `app/memlm/` (đã train ~2B+ token) không
   tương thích ngược, đây là đánh đổi đã biết trước khi chuyển nhánh.