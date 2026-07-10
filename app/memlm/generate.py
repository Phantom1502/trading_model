"""
generate.py — Sinh văn bản từ model đã train
================================================
Sliding window đơn giản: khi context vượt max_seq thì cắt phần đầu,
không cần prefill hay flush vào memory.

Giả định chạy/import từ THƯ MỤC GỐC project (xem ghi chú đầu file
train.py để biết 3 cách chạy hợp lệ).

Usage:
    from app.memlm.generate import load_model_for_inference, generate

    model, tokenizer, cfg = load_model_for_inference("checkpoints/best.pt")
    print(generate(model, tokenizer, cfg, "Trí tuệ nhân tạo là"))
"""

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = _THIS_DIR
while not os.path.isdir(os.path.join(_REPO_ROOT, "app")):
    _parent = os.path.dirname(_REPO_ROOT)
    if _parent == _REPO_ROOT:
        raise RuntimeError(
            "Không tìm thấy thư mục gốc project (thư mục chứa 'app/')."
        )
    _REPO_ROOT = _parent
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════
# Sampling
# ══════════════════════════════════════════════════════════════════════════

def _sample_next(
    logits     : torch.Tensor,   # (1, vocab)
    temperature: float,
    top_k      : int,
    top_p      : float,
) -> int:
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

    return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1).item()


# ══════════════════════════════════════════════════════════════════════════
# Generate
# ══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def generate(
    model, tokenizer, cfg, prompt,
    max_new=100, temperature=0.8, top_k=50, top_p=0.95, new_token_only=False,
):
    device  = next(model.parameters()).device
    max_seq = cfg.model.max_seq
    model.eval()

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    active = prompt_ids[-max_seq:] if len(prompt_ids) > max_seq else prompt_ids
    ids = torch.tensor([active], dtype=torch.long, device=device)

    generated = []
    past_key_values = None
    cur_len = 0
    next_tok = None

    for step in range(max_new):
        if step == 0:
            logits, past_key_values = model(ids, use_cache=True)
            cur_len = ids.size(1)
        else:
            if cur_len >= max_seq:
                trim = cur_len - (max_seq - 1)
                past_key_values = [
                    (k[:, :, trim:, :], v[:, :, trim:, :]) for k, v in past_key_values
                ]
                cur_len -= trim

            next_input = torch.tensor([[next_tok]], dtype=torch.long, device=device)
            logits, past_key_values = model(next_input, past_key_values=past_key_values, use_cache=True)
            cur_len += 1

        next_tok = _sample_next(logits[:, -1, :], temperature, top_k, top_p)
        generated.append(next_tok)
        if next_tok == tokenizer.eos_id:
            break

    if new_token_only:
        return tokenizer.decode(generated)
    return tokenizer.decode(prompt_ids + generated)


# ══════════════════════════════════════════════════════════════════════════
# Load model for inference
# ══════════════════════════════════════════════════════════════════════════

def load_model_for_inference(checkpoint_path: str, device: str = None, fallback_cfg=None):
    """
    Load model + tokenizer + config từ checkpoint.
    Đọc model_cfg đã lưu trong checkpoint để build đúng kiến trúc lúc train.
    """
    from app.memlm.config import get_100m_config, ModelConfig
    from app.memlm.tokenizer import load_tokenizer
    from app.memlm.model import build_model
    from app.memlm.utils import load_checkpoint
    import torch as _torch

    device = device or ("cuda" if _torch.cuda.is_available() else "cpu")
    cfg    = fallback_cfg or get_100m_config()
    cfg.train.device = device

    ckpt_raw = _torch.load(checkpoint_path, map_location=device)
    if "model_cfg" in ckpt_raw and ckpt_raw["model_cfg"] is not None:
        cfg.model = ModelConfig(**ckpt_raw["model_cfg"])
        print(f"  model_cfg: d_model={cfg.model.d_model}, n_layers={cfg.model.n_layers}, "
              f"max_seq={cfg.model.max_seq}")
    else:
        print("  CẢNH BÁO: checkpoint không có model_cfg — dùng config mặc định.")

    tokenizer            = load_tokenizer(cfg)
    cfg.model.vocab_size = tokenizer.vocab_size

    model = build_model(cfg).to(device)
    load_checkpoint(checkpoint_path, model, device=device)
    model.eval()

    return model, tokenizer, cfg


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/best.pt"

    model, tokenizer, cfg = load_model_for_inference(ckpt_path)

    prompts = [
        "Trí tuệ nhân tạo là",
        "Lịch sử Việt Nam bắt đầu",
        "Mô hình ngôn ngữ lớn có khả năng",
    ]
    for p in prompts:
        out = generate(model, tokenizer, cfg, p, max_new=80)
        print(f"\n['{p}']\n{out}\n" + "-" * 60)