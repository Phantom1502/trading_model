"""
generate.py — Sinh văn bản từ model đã train
================================================
Hỗ trợ prompt dài hơn max_seq qua 2 giai đoạn:

    Giai đoạn 1 — PREFILL:
        Feed toàn bộ phần prompt vượt quá max_seq vào M theo từng chunk.
        M tích lũy context từ đầu prompt trước khi bắt đầu generate.

        prompt (2000 token, max_seq=512):
            chunk 0 [0:512]   → forward → M update → detach
            chunk 1 [512:1024] → forward → M update → detach
            chunk 2 [1024:1488] → forward → M update → detach
            active_window = prompt[-512:]   ← cửa sổ bắt đầu generate

    Giai đoạn 2 — GENERATE với sliding window:
        Mỗi bước sinh ra 1 token mới và gắn vào cửa sổ active.
        Khi cửa sổ tràn max_seq, flush token cũ nhất vào M rồi trượt:

            [t0 t1 ... t511 | NEW]  → T=513 > max_seq
                flush [t0] vào M → detach
            [t1 t2 ... t511 NEW]    → T=512, tiếp tục generate

        M luôn giữ thông tin về những gì đã trôi qua cửa sổ attention.

Usage:
    from generate import generate, load_model_for_inference

    model, tokenizer, cfg = load_model_for_inference("checkpoints/best.pt")
    text = generate(model, tokenizer, cfg, "Trí tuệ nhân tạo là")

    # Prompt dài — M sẽ tự prefill
    long_prompt = "..." * 5000  # dài hơn max_seq
    text = generate(model, tokenizer, cfg, long_prompt, max_new=200)
"""

import torch
import torch.nn.functional as F

from model import causal_mask


# ══════════════════════════════════════════════════════════════════════════
# Sampling helper
# ══════════════════════════════════════════════════════════════════════════

def _sample_next(
    logits     : torch.Tensor,   # (1, vocab)
    temperature: float,
    top_k      : int,
    top_p      : float,
) -> int:
    """Top-k + top-p sampling, trả về token id tiếp theo."""
    logits = logits / temperature

    if top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits = logits.masked_fill(logits < v[:, -1:], float("-inf"))

    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs   = F.softmax(sorted_logits, dim=-1)
        cumprob = torch.cumsum(probs, dim=-1)
        remove  = cumprob - probs > top_p
        sorted_logits[remove] = float("-inf")
        logits = torch.zeros_like(logits).scatter_(1, sorted_idx, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()


# ══════════════════════════════════════════════════════════════════════════
# Prefill — warm up M với phần prompt dài hơn max_seq
# ══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def prefill_memory(
    model      : torch.nn.Module,
    prompt_ids : list[int],
    max_seq    : int,
    device     : torch.device,
) -> list[int]:
    """
    Feed toàn bộ phần đầu của prompt (vượt quá max_seq) vào M theo chunk.
    Trả về active_window = max_seq token cuối của prompt, dùng làm context
    bắt đầu cho vòng generate.

    Nếu prompt ngắn hơn hoặc bằng max_seq: không làm gì, trả về nguyên prompt.

    Lưu ý: hàm này KHÔNG reset M — caller phải reset trước khi gọi.
    """
    if len(prompt_ids) <= max_seq:
        return prompt_ids   # không cần prefill

    # Phần cần feed vào M (tất cả trừ cửa sổ cuối max_seq)
    prefix = prompt_ids[:-max_seq]

    n_chunks = (len(prefix) + max_seq - 1) // max_seq
    print(f"[prefill] prompt={len(prompt_ids)} token > max_seq={max_seq} "
          f"→ prefill M với {len(prefix)} token ({n_chunks} chunk)...")

    pos = 0
    while pos < len(prefix):
        end   = min(pos + max_seq, len(prefix))
        chunk = prefix[pos:end]
        ids_t = torch.tensor([chunk], dtype=torch.long, device=device)
        model(ids_t, attn_mask=causal_mask(len(chunk), device))
        model.detach_memory()
        pos = end

    return prompt_ids[-max_seq:]   # active window để bắt đầu generate


# ══════════════════════════════════════════════════════════════════════════
# Generate
# ══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def generate(
    model,
    tokenizer,
    cfg,
    prompt      : str,
    max_new     : int   = 100,
    temperature : float = 0.8,
    top_k       : int   = 50,
    top_p       : float = 0.95,
    new_token_only : bool  = False,   # nếu True, chỉ trả về phần sinh thêm, không kèm prompt
) -> str:
    """
    Sinh văn bản từ prompt. Tự động prefill M nếu prompt dài hơn max_seq.

    Args:
        prompt     : văn bản đầu vào (bất kỳ độ dài nào)
        max_new    : số token tối đa sinh thêm
        temperature: nhiệt độ sampling (thấp → đoán chắc hơn, cao → đa dạng hơn)
        top_k      : chỉ lấy top-k token có xác suất cao nhất
        top_p      : nucleus sampling — lấy tập token có tổng xác suất >= top_p
        new_token_only: nếu True, chỉ trả về phần sinh thêm, không kèm prompt gốc

    Returns:
        Chuỗi văn bản đầy đủ (prompt gốc + phần sinh thêm).
    """
    device  = next(model.parameters()).device
    max_seq = cfg.model.max_seq
    model.eval()

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

    # ── Giai đoạn 1: Reset M và prefill nếu prompt dài ───────────────────
    model.reset_memory(batch_size=1, device=device)
    active = prefill_memory(model, prompt_ids, max_seq, device)

    # ── Giai đoạn 2: Generate với sliding window ──────────────────────────
    ids       = torch.tensor([active], dtype=torch.long, device=device)
    generated = []   # chỉ lưu token MỚI (không bao gồm prompt)

    for _ in range(max_new):
        T = ids.size(1)

        # Cửa sổ tràn max_seq → flush token cũ nhất vào M, trượt cửa sổ
        if T > max_seq:
            n_flush = T - max_seq
            flush   = ids[:, :n_flush]
            model(flush, attn_mask=causal_mask(n_flush, device))
            model.detach_memory()
            ids = ids[:, n_flush:]
            T   = max_seq

        # Forward — lấy logit của token cuối
        logits    = model(ids, attn_mask=causal_mask(T, device))
        next_tok  = _sample_next(logits[:, -1, :], temperature, top_k, top_p)

        generated.append(next_tok)
        ids = torch.cat([ids, torch.tensor([[next_tok]], device=device)], dim=1)

        if next_tok == tokenizer.eos_id:
            break

    if new_token_only:
        return tokenizer.decode(generated)  # chỉ phần sinh thêm
    # Decode: prompt gốc + phần sinh thêm
    return tokenizer.decode(prompt_ids + generated)


# ══════════════════════════════════════════════════════════════════════════
# Load model for inference
# ══════════════════════════════════════════════════════════════════════════

def load_model_for_inference(checkpoint_path: str, device: str = None, fallback_cfg=None):
    """
    Load model + tokenizer + config từ checkpoint để inference.

    Đọc model_cfg đã lưu TRONG checkpoint để build đúng kiến trúc lúc train,
    tránh lỗi "Missing key(s)" hoặc "size mismatch" khi kiến trúc khác nhau.

    Args:
        checkpoint_path: đường dẫn .pt
        device         : "cuda"/"cpu", tự detect nếu None
        fallback_cfg   : Config dùng làm nền khi checkpoint không có model_cfg
    """
    from config import get_100m_config, ModelConfig
    from tokenizer import load_tokenizer
    from model import build_model
    from utils import load_checkpoint
    import torch as _torch

    device = device or ("cuda" if _torch.cuda.is_available() else "cpu")

    cfg = fallback_cfg or get_100m_config()
    cfg.train.device = device

    # Đọc model_cfg từ checkpoint TRƯỚC khi build model
    ckpt_raw = _torch.load(checkpoint_path, map_location=device)
    if "model_cfg" in ckpt_raw and ckpt_raw["model_cfg"] is not None:
        saved = ckpt_raw["model_cfg"]
        cfg.model = ModelConfig(**saved)
        print(f"  Đã đọc model_cfg: use_memory={cfg.model.use_memory}, "
              f"d_model={cfg.model.d_model}, n_layers={cfg.model.n_layers}, "
              f"num_slots={cfg.model.num_slots}, max_seq={cfg.model.max_seq}")
    else:
        print("  CẢNH BÁO: checkpoint không có model_cfg — dùng config mặc định.")

    tokenizer = load_tokenizer(cfg)
    cfg.model.vocab_size = tokenizer.vocab_size

    model = build_model(cfg).to(device)
    load_checkpoint(checkpoint_path, model, device=device)
    model.eval()

    return model, tokenizer, cfg


# ══════════════════════════════════════════════════════════════════════════
# CLI / demo
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/best.pt"

    model, tokenizer, cfg = load_model_for_inference(ckpt_path)

    # Prompt ngắn — không cần prefill
    short_prompts = [
        "Trí tuệ nhân tạo là",
        "Lịch sử Việt Nam bắt đầu",
        "Mô hình ngôn ngữ lớn có khả năng",
    ]
    for p in short_prompts:
        out = generate(model, tokenizer, cfg, p, max_new=80)
        print(f"\n['{p}']\n{out}\n" + "-" * 60)

    # Demo prompt dài — M sẽ prefill
    long_prompt = (
        "Việt Nam là một quốc gia nằm ở phía đông bán đảo Đông Dương, "
        "thuộc khu vực Đông Nam Á. Đây là một đất nước có lịch sử lâu đời "
        "với nền văn hóa phong phú và đa dạng. Trong suốt chiều dài lịch sử, "
        "dân tộc Việt Nam đã trải qua nhiều cuộc kháng chiến chống ngoại xâm "
        "và xây dựng đất nước. "
    ) * 20   # ~2000 từ, vượt max_seq=512

    print(f"\n[Prompt dài ~{len(tokenizer.encode(long_prompt))} token]")
    out = generate(model, tokenizer, cfg, long_prompt, max_new=100)
    print(out[-300:])   # in 300 ký tự cuối