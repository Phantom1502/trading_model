# MemoryLM — Kiến trúc model

`app/memlm/model/` — LLaMA-style transformer ~110M params, không có bias trên
Linear, dùng RMSNorm + Pre-Norm, SwiGLU FFN, RoPE.

## Thành phần (`model/layers.py`)

| Thành phần | File | Ghi chú |
|---|---|---|
| `RMSNorm` | `layers.py` | Thay LayerNorm — không trừ mean, chỉ chuẩn hoá theo RMS. Tính ở float32 để ổn định khi mixed precision. |
| `precompute_freqs_cis` / `apply_rope` | `layers.py` | RoPE — xoay Q/K bằng số phức, mã hoá vị trí tương đối. |
| `SwiGLU` | `layers.py` | `w2(SiLU(w1(x)) * w3(x))`. `hidden_dim ≈ 8/3 * d_model`, làm tròn bội 64. Init std=0.02, `w2` bị `_scaled_init()` ghi đè sau vì nằm trên residual. |

## Attention (`model/attention.py`)

`SelfAttentionRoPE`:
- Pre-Norm nằm bên trong (tự gọi `RMSNorm` trước khi tính Q/K/V).
- 4 Linear `Wq/Wk/Wv/Wo`, tất cả `bias=False`.
- RoPE áp lên Q và K sau khi reshape theo head.
- `Wo` bị ghi đè init bởi `TransformerBlock._scaled_init()` vì nằm trên residual.

## Block (`model/block.py`)

`TransformerBlock`:
```
x = x + self_attn(x)      # residual quanh attention (norm nằm trong self_attn)
x = x + ffn(norm2(x))      # residual quanh FFN
```
- `_scaled_init(n_layers)`: scale `1/sqrt(2*n_layers)` cho `Wo.weight` và `ffn.w2.weight`
  — chuẩn practice để residual không nổ ở model sâu.
- `use_router=False` (layer đầu/cuối, và cấu hình mặc định hiện tại): luôn
  chạy full block — dense, không routing.
- `use_router=True` (routing Skip/Run cho layer giữa) tồn tại trong code
  (`DepthRouter` trong `block.py`) nhưng phần nghiên cứu/phát triển tiếp
  cho cơ chế này đã **chuyển sang project riêng** — không còn là hướng phát
  triển chính ở đây, không cần document sâu trong repo này nữa.

## Model wrapper (`model/lm.py`)

`MemoryLM`:
- `token_emb` (Embedding) → N `TransformerBlock` → `norm_out` (RMSNorm) → `lm_head`.
- **Weight tying**: `lm_head.weight = token_emb.weight`.
- `freqs_cis` precompute 1 lần cho `max_seq * 2`, lưu làm buffer không persistent.
- `causal_mask(T, device)`: additive mask (`-inf` phía trên đường chéo).
- `build_model(cfg)`: entry point dựng model từ `ModelConfig`.

## Cấu hình tham chiếu (`config.py`)

- `get_100m_config()`: d_model=512, n_heads=8, n_layers=8, max_seq=512,
  batch=32, grad_accum=64 (⇒ ~1M token/step — xem `docs/conventions/known-pitfalls.md`).
- `get_110m_config()`: n_layers=30 (sâu hơn nhiều), batch=8, grad_accum=64,
  dùng khi bật `use_router` để giữ FLOPs hợp lý dù layer count cao.

## Lưu ý quan trọng

- Toàn bộ Linear trong block **không có bias** (LLaMA-style).
- Init cố định (`std=0.02`) áp cho mọi Linear trước, các projection nằm trên
  đường residual (`Wo`, `ffn.w2`) bị **ghi đè sau** bởi scaled init — thứ tự
  gọi trong `__init__` quan trọng, đừng đảo ngược khi refactor.
- Context Memory (M) qua Read/Write cross-attention đã được tách ra thành
  project riêng — codebase `app/memlm/` hiện tại **không còn** memory module,
  đã dọn sạch dead code liên quan (xem README lịch sử cũ nếu cần tham chiếu).