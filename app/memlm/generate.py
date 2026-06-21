"""
generate.py — Sinh văn bản từ model đã train
================================================
Usage:
    from generate import generate, load_model_for_inference

    model, tokenizer, cfg = load_model_for_inference("checkpoints/best.pt")
    text = generate(model, tokenizer, cfg, "Trí tuệ nhân tạo là")
"""

import torch
import torch.nn.functional as F

from model import causal_mask


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
) -> str:
    device = next(model.parameters()).device
    model.eval()

    ids = torch.tensor(
        [tokenizer.encode(prompt, add_special_tokens=False)],
        dtype=torch.long, device=device,
    )

    # Reset M cho document mới (prompt mới = context mới)
    model.reset_memory(batch_size=1, device=device)

    for _ in range(max_new):
        T = ids.size(1)
        if T > cfg.model.max_seq:
            ids = ids[:, -cfg.model.max_seq:]
            T   = cfg.model.max_seq

        mask   = causal_mask(T, device)
        logits = model(ids, attn_mask=mask)

        next_logits = logits[:, -1, :] / temperature

        # Top-k filter
        if top_k > 0:
            v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
            next_logits = next_logits.masked_fill(
                next_logits < v[:, -1:], float("-inf")
            )

        # Top-p (nucleus) filter
        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
            probs   = F.softmax(sorted_logits, dim=-1)
            cumprob = torch.cumsum(probs, dim=-1)
            remove  = cumprob - probs > top_p
            sorted_logits[remove] = float("-inf")
            next_logits = torch.zeros_like(next_logits).scatter_(
                1, sorted_idx, sorted_logits
            )

        probs   = F.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        ids     = torch.cat([ids, next_id], dim=1)

        if next_id.item() == tokenizer.eos_id:
            break

    return tokenizer.decode(ids[0].tolist())


def load_model_for_inference(checkpoint_path: str, device: str = None, fallback_cfg=None):
    """
    Load model + tokenizer + config từ checkpoint để inference.

    QUAN TRỌNG: đọc đúng ModelConfig đã lưu TRONG checkpoint (model_cfg),
    không dùng config mặc định cứng. Nếu build model bằng config khác với
    lúc train (ví dụ use_memory khác, num_slots khác, d_model khác), sẽ lỗi
    "Missing key(s)" hoặc "Unexpected key(s) in state_dict" vì kiến trúc
    không khớp.

    Args:
        checkpoint_path: đường dẫn .pt
        device         : "cuda"/"cpu", tự detect nếu None
        fallback_cfg   : Config dùng làm NỀN khi checkpoint cũ KHÔNG có
                          model_cfg (checkpoint train từ trước khi tính
                          năng này được thêm vào). Nếu None, dùng
                          get_100m_config() làm nền — nhưng nếu checkpoint
                          đó train với use_memory=False hoặc kiến trúc khác,
                          vẫn sẽ lỗi vì không có cách nào suy ra kiến trúc
                          gốc từ checkpoint cũ.
    """
    from config import get_100m_config, ModelConfig
    from tokenizer import load_tokenizer
    from model import build_model
    from utils import load_checkpoint
    import torch as _torch

    device = device or ("cuda" if _torch.cuda.is_available() else "cpu")

    cfg = fallback_cfg or get_100m_config()
    cfg.train.device = device

    # ── Đọc trước model_cfg từ checkpoint (nếu có), TRƯỚC khi build model ──
    ckpt_raw = _torch.load(checkpoint_path, map_location=device)
    if "model_cfg" in ckpt_raw and ckpt_raw["model_cfg"] is not None:
        # Ghi đè cfg.model bằng đúng config đã lưu lúc train
        saved = ckpt_raw["model_cfg"]
        cfg.model = ModelConfig(**saved)
        print(f"  Đã đọc model_cfg từ checkpoint: use_memory={cfg.model.use_memory}, "
              f"d_model={cfg.model.d_model}, n_layers={cfg.model.n_layers}, "
              f"num_slots={cfg.model.num_slots}")
    else:
        print("  CẢNH BÁO: checkpoint không có model_cfg (checkpoint cũ). "
              "Dùng config mặc định — có thể lỗi nếu kiến trúc lúc train khác.")

    tokenizer = load_tokenizer(cfg)
    cfg.model.vocab_size = tokenizer.vocab_size

    model = build_model(cfg).to(device)
    load_checkpoint(checkpoint_path, model, device=device)
    model.eval()

    return model, tokenizer, cfg


if __name__ == "__main__":
    import sys
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
