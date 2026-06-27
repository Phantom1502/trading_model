"""
benchmark_hellaswag.py — Đánh giá MemoryLM trên HellaSwag
===========================================================
Zero-shot evaluation: model chọn completion có avg log-prob/token cao nhất.

Thiết kế memory:
    Memory KHÔNG reset giữa các item — đúng theo nguyên lí thiết kế:
    M chạy xuyên suốt, model tự học cái gì cần giữ qua gate.
    Inference cần nạp context trước (prefill) để M có thông tin trước
    khi evaluate — xem phần prefill bên dưới nếu cần.

Lưu ý:
    - HellaSwag gốc là tiếng Anh — model tiếng Việt sẽ cho accuracy thấp
      (~25% = random baseline), dùng để theo dõi TREND qua các checkpoint,
      không phải để so sánh tuyệt đối với GPT-2/LLaMA.
    - Để dùng bản tiếng Việt: thay _load_hellaswag() bằng nguồn của bạn,
      đảm bảo mỗi item có cấu trúc {ctx, endings (list 4 str), label (int)}.

Usage:
    python benchmark_hellaswag.py checkpoints/chunk_10.pt
    python benchmark_hellaswag.py checkpoints/chunk_10.pt --n-samples 200
    python benchmark_hellaswag.py ckpt_10.pt ckpt_50.pt --n-samples 200
"""

import torch
import torch.nn.functional as F
from datasets import load_dataset
from model import causal_mask


# ══════════════════════════════════════════════════════════════════════════
# Core: avg log-prob per token
# ══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def avg_logprob_per_token(
    model,
    tokenizer,
    prompt     : str,
    completion : str,
    device     : torch.device,
    max_seq    : int = 512,
) -> float:
    """
    Tính log-prob trung bình trên mỗi token của `completion` cho trước `prompt`.

    Normalize theo số token — quan trọng với BPE vì completion dài hơn
    sẽ có tổng log-prob thấp hơn chỉ vì nhiều token hơn.
    Returns: giá trị trong (-inf, 0], càng gần 0 xác suất càng cao.
    """
    prompt_ids     = tokenizer.encode(prompt,     add_special_tokens=False)
    completion_ids = tokenizer.encode(completion, add_special_tokens=False)

    if not completion_ids:
        return float("-inf")

    full_ids = prompt_ids + completion_ids
    if len(full_ids) > max_seq:
        keep = max_seq - len(completion_ids)
        if keep <= 0:
            return float("-inf")
        full_ids = prompt_ids[-keep:] + completion_ids

    ids_t = torch.tensor([full_ids], dtype=torch.long, device=device)
    T     = ids_t.size(1)
    mask  = causal_mask(T, device)

    logits    = model(ids_t, attn_mask=mask)
    log_probs = F.log_softmax(logits[0], dim=-1)

    n_prompt = T - len(completion_ids)
    total_lp = 0.0
    for i, tok_id in enumerate(completion_ids):
        pos = n_prompt + i - 1
        if pos < 0:
            continue
        total_lp += log_probs[pos, tok_id].item()

    return total_lp / len(completion_ids)


# ══════════════════════════════════════════════════════════════════════════
# Load HellaSwag
# ══════════════════════════════════════════════════════════════════════════

def _load_hellaswag(split: str = "validation", n_samples: int = None) -> list[dict]:
    """
    Load HellaSwag từ HuggingFace.
    Trả về list[dict] với keys: ctx, endings, label.

    Để dùng bản tiếng Việt: thay hàm này bằng nguồn của bạn,
    đảm bảo mỗi item có cùng cấu trúc {ctx, endings (list 4 str), label (int)}.
    """
    ds = load_dataset("Rowan/hellaswag", split=split, trust_remote_code=True)
    if n_samples:
        ds = ds.select(range(min(n_samples, len(ds))))

    items = []
    for row in ds:
        # ctx đầy đủ = activity_label + ctx_a + ctx_b
        ctx = f"{row['activity_label']}: {row['ctx']}"
        items.append({
            "ctx"    : ctx,
            "endings": row["endings"],    # list 4 chuỗi
            "label"  : int(row["label"]), # index đúng (0-3)
        })
    return items


# ══════════════════════════════════════════════════════════════════════════
# Evaluate
# ══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_hellaswag(
    model,
    tokenizer,
    cfg,
    n_samples : int = 500,
    split     : str = "validation",
    verbose   : bool = True,
) -> dict:
    """
    Chạy HellaSwag evaluation.

    Memory chạy xuyên suốt toàn bộ evaluation — không reset giữa các item.
    Model tự học cái gì cần giữ qua gate mechanism.

    Returns dict:
        accuracy        : float (0.0 → 1.0)
        n_correct       : int
        n_total         : int
        random_baseline : 0.25
        delta_vs_random : accuracy - 0.25
    """
    device  = next(model.parameters()).device
    max_seq = cfg.model.max_seq
    model.eval()

    items   = _load_hellaswag(split=split, n_samples=n_samples)
    n_total = len(items)

    if verbose:
        print(f"\n{'═'*60}")
        print(f"  HELLASWAG  ({split}, n={n_total})")
        print(f"  Random baseline: 25.0%")
        print(f"  Memory: xuyên suốt (không reset giữa item)")
        print(f"{'═'*60}")

    n_correct = 0

    for i, item in enumerate(items):
        scores = [
            avg_logprob_per_token(
                model, tokenizer,
                item["ctx"], ending,
                device, max_seq,
            )
            for ending in item["endings"]
        ]

        pred  = scores.index(max(scores))
        label = item["label"]

        if pred == label:
            n_correct += 1

        if verbose and (i + 1) % 100 == 0:
            acc_so_far = n_correct / (i + 1) * 100
            print(f"  [{i+1:>4}/{n_total}] acc so far: {acc_so_far:.1f}%")

    accuracy = n_correct / n_total

    if verbose:
        print(f"\n{'─'*60}")
        print(f"  Accuracy    : {accuracy*100:.2f}%")
        print(f"  Correct     : {n_correct}/{n_total}")
        print(f"  vs random   : {(accuracy - 0.25)*100:+.2f}%")
        print(f"{'═'*60}\n")

    return {
        "accuracy"        : accuracy,
        "n_correct"       : n_correct,
        "n_total"         : n_total,
        "random_baseline" : 0.25,
        "delta_vs_random" : accuracy - 0.25,
    }


# ══════════════════════════════════════════════════════════════════════════
# Wrapper cho run_all() trong benchmark.py
# ══════════════════════════════════════════════════════════════════════════

def hellaswag_score_for_run_all(
    model,
    tokenizer,
    cfg,
    n_samples: int = 200,
) -> float:
    """Trả về accuracy (0.0 → 1.0), dùng để tích hợp vào run_all()."""
    result = run_hellaswag(
        model, tokenizer, cfg,
        n_samples=n_samples,
        verbose=False,
    )
    return result["accuracy"]


# ══════════════════════════════════════════════════════════════════════════
# So sánh nhiều checkpoint
# ══════════════════════════════════════════════════════════════════════════

def compare_hellaswag(
    checkpoint_paths : list[str],
    n_samples        : int = 200,
    verbose          : bool = False,
) -> None:
    """
    Load từng checkpoint, chạy HellaSwag, in bảng so sánh.

    Usage:
        compare_hellaswag([
            "checkpoints/chunk_10.pt",
            "checkpoints/chunk_30.pt",
            "checkpoints/chunk_50.pt",
        ])
    """
    from generate import load_model_for_inference

    rows = []
    for path in checkpoint_paths:
        print(f"\n── Loading {path} ──")
        model, tokenizer, cfg = load_model_for_inference(path)
        r = run_hellaswag(model, tokenizer, cfg, n_samples=n_samples, verbose=verbose)
        r["checkpoint"] = path
        rows.append(r)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n{'═'*60}")
    print(f"  {'CHECKPOINT':<30} {'ACC':>7}  {'vs RND':>7}")
    print(f"{'─'*60}")
    for r in rows:
        name = r["checkpoint"].split("/")[-1]
        print(
            f"  {name:<30} {r['accuracy']*100:>6.2f}%  "
            f"{r['delta_vs_random']*100:>+6.2f}%"
        )
    print(f"{'═'*60}")


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="HellaSwag benchmark cho MemoryLM")
    parser.add_argument("checkpoints", nargs="+", help="Path tới checkpoint .pt")
    parser.add_argument("--n-samples", type=int, default=500,
                        help="Số item evaluate (default 500, max ~10k)")
    parser.add_argument("--split",     type=str, default="validation",
                        choices=["validation", "train"],
                        help="Split của HellaSwag dataset")
    parser.add_argument("--verbose",   action="store_true",
                        help="In chi tiết từng item (chỉ nên dùng với n nhỏ)")
    args = parser.parse_args()

    if len(args.checkpoints) == 1:
        from generate import load_model_for_inference
        model, tokenizer, cfg = load_model_for_inference(args.checkpoints[0])
        run_hellaswag(
            model, tokenizer, cfg,
            n_samples=args.n_samples,
            split=args.split,
            verbose=True,
        )
    else:
        compare_hellaswag(
            args.checkpoints,
            n_samples=args.n_samples,
            verbose=args.verbose,
        )