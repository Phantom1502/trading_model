"""
benchmark.py — Đánh giá pretrained model qua log-probability scoring
=======================================================================
Đo semantic embedding đã hội tụ chưa, KHÔNG dùng QA/multiple-choice
vì model chưa instruction-tuned.

Nguyên lý:
    score = avg_logprob_per_token(positive) - avg_logprob_per_token(negative)

    score > 0  → model gán xác suất cao hơn cho chuỗi đúng nghĩa
    score tăng dần qua các checkpoint → embedding đang tiếp tục học

3 cấp đánh giá (thường hội tụ theo thứ tự này khi train):
    Cấp 1 — Semantic : phân biệt loại thực thể rộng (động vật vs hành tinh)
    Cấp 2 — Entity   : phân biệt trong cùng miền (nhà vật lý vs ca sĩ)
    Cấp 3 — Fact     : kiểm tra sự kiện cụ thể (thủ đô, năm sinh, ...)

────────────────────────────────────────────────────────────────────────────
Lưu ý quan trọng khi dùng với PhoBERT tokenizer:

PhoBERT tokenize BPE → một "từ" có thể thành nhiều token.
PHẢI normalize log-prob theo SỐ TOKEN (không phải số từ), không thì
chuỗi nhiều token sẽ luôn thua chuỗi ngắn hơn một cách bất công.

avg_logprob_per_token = sum(log P(t_i | context)) / n_tokens

────────────────────────────────────────────────────────────────────────────
Usage nhanh:

    from benchmark import run_all, SEMANTIC_BENCH, ENTITY_BENCH, FACT_BENCH
    from generate import load_model_for_inference

    model, tokenizer, cfg = load_model_for_inference("checkpoints/chunk_10.pt")
    results = run_all(model, tokenizer, cfg, verbose=True)
    # results = {"semantic": 6.25, "entity": 3.10, "fact": 1.40}

So sánh 2 checkpoint:

    r1 = run_all(model_ckpt10, tokenizer, cfg)
    r2 = run_all(model_ckpt50, tokenizer, cfg)
    print(r1, r2)  # xem 3 đường cong tăng không
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List
import math


# ══════════════════════════════════════════════════════════════════════════
# Dataset benchmark
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class BenchItem:
    """1 mục benchmark: prompt + danh sách positive/negative."""
    prompt   : str
    positive : List[str]
    negative : List[str]
    note     : str = ""   # ghi chú tuỳ chọn, không ảnh hưởng tính toán


# ── Cấp 1: Semantic — phân biệt loại thực thể rộng ──────────────────────

SEMANTIC_BENCH: List[BenchItem] = [
    BenchItem(
        prompt   = "Con mèo là",
        positive = ["động vật", "thú nuôi", "sinh vật"],
        negative = ["quốc gia", "hành tinh", "phần mềm", "công thức"],
    ),
    BenchItem(
        prompt   = "Hà Nội là",
        positive = ["thành phố", "thủ đô", "đô thị"],
        negative = ["động vật", "hành tinh", "phần mềm", "công thức toán học"],
    ),
    BenchItem(
        prompt   = "Python là",
        positive = ["ngôn ngữ lập trình", "phần mềm", "công cụ lập trình"],
        negative = ["quốc gia", "loài chim", "núi lửa", "hành tinh"],
    ),
    BenchItem(
        prompt   = "Albert Einstein là",
        positive = ["nhà khoa học", "nhà vật lý", "học giả", "nhà nghiên cứu"],
        negative = ["cây bụi", "quốc gia", "trò chơi điện tử", "huyện"],
    ),
    BenchItem(
        prompt   = "Sông Hồng là",
        positive = ["con sông", "dòng sông", "nguồn nước"],
        negative = ["loài chim", "phần mềm", "vũ khí", "hành tinh"],
    ),
    BenchItem(
        prompt   = "Bóng đá là",
        positive = ["môn thể thao", "trò chơi", "hoạt động thể chất"],
        negative = ["ngôn ngữ lập trình", "quốc gia", "loài cá", "hành tinh"],
    ),
]


# ── Cấp 2: Entity — phân biệt trong cùng miền ────────────────────────────

ENTITY_BENCH: List[BenchItem] = [
    BenchItem(
        prompt   = "Albert Einstein là",
        positive = ["nhà vật lý", "nhà khoa học"],
        negative = ["nhà hóa học", "nhà văn", "ca sĩ", "vận động viên"],
        note     = "người nổi tiếng — phân biệt ngành khoa học",
    ),
    BenchItem(
        prompt   = "Hà Nội là thủ đô của",
        positive = ["Việt Nam", "nước Việt Nam"],
        negative = ["Trung Quốc", "Nhật Bản", "Thái Lan", "Campuchia"],
        note     = "địa lý — phân biệt quốc gia trong khu vực",
    ),
    BenchItem(
        prompt   = "Hổ là loài động vật",
        positive = ["ăn thịt", "thuộc họ mèo lớn", "nguy hiểm"],
        negative = ["ăn cỏ", "dưới nước", "bay được", "sống ở Bắc Cực"],
        note     = "sinh vật — phân biệt đặc tính trong cùng miền động vật",
    ),
    BenchItem(
        prompt   = "Ngôn ngữ Python thường được dùng để",
        positive = ["lập trình", "phân tích dữ liệu", "xây dựng ứng dụng"],
        negative = ["nấu ăn", "leo núi", "chơi thể thao", "vẽ tranh sơn dầu"],
        note     = "công nghệ — phân biệt ứng dụng của công cụ",
    ),
    BenchItem(
        prompt   = "Mặt Trời là",
        positive = ["ngôi sao", "thiên thể", "nguồn năng lượng"],
        negative = ["hành tinh", "vệ tinh", "tiểu hành tinh", "sao chổi"],
        note     = "thiên văn — phân biệt loại thiên thể",
    ),
]


# ── Cấp 3: Fact — sự kiện cụ thể, chỉ có 1 đáp án đúng ─────────────────

FACT_BENCH: List[BenchItem] = [
    BenchItem(
        prompt   = "Thủ đô của Việt Nam là",
        positive = [" Hà Nội"],
        negative = [" Thành phố Hồ Chí Minh", " Đà Nẵng", " Huế", " Cần Thơ"],
        note     = "có khoảng trắng đầu để ghép sát prompt, tránh BPE khác",
    ),
    BenchItem(
        prompt   = "Nước láng giềng phía Bắc của Việt Nam là",
        positive = [" Trung Quốc"],
        negative = [" Lào", " Campuchia", " Thái Lan", " Myanmar"],
    ),
    BenchItem(
        prompt   = "Ngôn ngữ lập trình được dùng để huấn luyện nhiều mô hình AI nhất hiện nay là",
        positive = [" Python"],
        negative = [" Java", " C++", " JavaScript", " Ruby"],
    ),
    BenchItem(
        prompt   = "Đơn vị đo nhiệt độ trong hệ SI là",
        positive = [" Kelvin"],
        negative = [" Celsius", " Fahrenheit", " Rankine"],
    ),
    BenchItem(
        prompt   = "Sông dài nhất thế giới là sông",
        positive = [" Nile", " Nin"],
        negative = [" Amazon", " Dương Tử", " Mekong", " Hồng Hà"],
        note     = "positive có 2 cách viết phổ biến trong tiếng Việt",
    ),
]


# ══════════════════════════════════════════════════════════════════════════
# Core scoring
# ══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def avg_logprob_per_token(
    model,
    tokenizer,
    prompt    : str,
    completion: str,
    device    : torch.device,
    max_seq   : int = 512,
) -> float:
    """
    Tính log-prob trung bình trên MỖI TOKEN của `completion` cho trước `prompt`.

    Cụ thể: P(completion | prompt) = prod P(t_i | prompt + t_0..t_{i-1})
    → lấy log → chia số token của completion.

    Normalize theo số token (không phải số từ) vì PhoBERT dùng BPE —
    một từ dài có thể thành 3-4 subword, nếu không normalize thì chuỗi
    nhiều token sẽ luôn có log-prob thấp hơn (tổng log-prob âm hơn) dù
    đúng nghĩa hơn.

    Returns: giá trị trong (-inf, 0], càng gần 0 càng xác suất cao.
    """
    # Tokenize riêng phần prompt và phần completion
    prompt_ids     = tokenizer.encode(prompt,     add_special_tokens=False)
    completion_ids = tokenizer.encode(completion, add_special_tokens=False)

    if not completion_ids:
        return float("-inf")

    # Ghép và cắt nếu quá dài
    full_ids = prompt_ids + completion_ids
    if len(full_ids) > max_seq:
        # Ưu tiên giữ nguyên completion, cắt prompt từ bên trái
        keep_prompt = max_seq - len(completion_ids)
        if keep_prompt <= 0:
            return float("-inf")
        full_ids = prompt_ids[-keep_prompt:] + completion_ids

    ids_tensor = torch.tensor([full_ids], dtype=torch.long, device=device)
    B, T = ids_tensor.shape

    # Model forward
    from model import causal_mask
    mask   = causal_mask(T, device)
    logits = model(ids_tensor, attn_mask=mask)   # (1, T, vocab)

    log_probs = F.log_softmax(logits[0], dim=-1)  # (T, vocab)

    # Chỉ tính log-prob cho phần completion
    n_prompt = len(prompt_ids) if len(full_ids) == len(prompt_ids) + len(completion_ids) \
               else T - len(completion_ids)
    n_prompt = max(0, T - len(completion_ids))

    total_logp = 0.0
    for i, tok_id in enumerate(completion_ids):
        pos = n_prompt + i - 1          # vị trí token TRƯỚC trong sequence
        if pos < 0:
            continue
        total_logp += log_probs[pos, tok_id].item()

    return total_logp / len(completion_ids)   # normalize theo số token


@torch.no_grad()
def score_item(model, tokenizer, item: BenchItem, device: torch.device, max_seq: int = 512) -> dict:
    """
    Tính score cho 1 BenchItem.

    Returns dict:
        pos_scores  : list log-prob/token cho từng chuỗi positive
        neg_scores  : list log-prob/token cho từng chuỗi negative
        pos_mean    : trung bình positive
        neg_mean    : trung bình negative
        score       : pos_mean - neg_mean (dương = tốt)
    """
    pos_scores = [
        avg_logprob_per_token(model, tokenizer, item.prompt, p, device, max_seq)
        for p in item.positive
    ]
    neg_scores = [
        avg_logprob_per_token(model, tokenizer, item.prompt, n, device, max_seq)
        for n in item.negative
    ]

    pos_mean = sum(pos_scores) / len(pos_scores)
    neg_mean = sum(neg_scores) / len(neg_scores)

    return {
        "prompt"    : item.prompt,
        "pos_scores": pos_scores,
        "neg_scores": neg_scores,
        "pos_mean"  : pos_mean,
        "neg_mean"  : neg_mean,
        "score"     : pos_mean - neg_mean,
    }


# ══════════════════════════════════════════════════════════════════════════
# Benchmark runners
# ══════════════════════════════════════════════════════════════════════════

def run_benchmark(
    model,
    tokenizer,
    bench     : List[BenchItem],
    level_name: str,
    device    : torch.device,
    max_seq   : int = 512,
    verbose   : bool = True,
) -> float:
    """
    Chạy 1 cấp benchmark. Trả về score trung bình của cả cấp.

    Reset M trước mỗi item (mỗi prompt là 1 context độc lập).
    """
    model.eval()
    scores = []

    if verbose:
        print(f"\n{'─'*60}")
        print(f"  [{level_name.upper()}]")
        print(f"{'─'*60}")

    for item in bench:
        # Reset memory cho mỗi item — context mới, không dùng M của item trước
        if hasattr(model, "reset_memory"):
            model.reset_memory(batch_size=1, device=device)

        r = score_item(model, tokenizer, item, device, max_seq)
        scores.append(r["score"])

        if verbose:
            pos_fmt = "  ".join(f"{s:+.2f}" for s in r["pos_scores"])
            neg_fmt = "  ".join(f"{s:+.2f}" for s in r["neg_scores"])
            print(f"\n  prompt  : {item.prompt!r}")
            print(f"  pos_lp  : [{pos_fmt}]  mean={r['pos_mean']:+.3f}")
            print(f"  neg_lp  : [{neg_fmt}]  mean={r['neg_mean']:+.3f}")
            status = "✓" if r["score"] > 0 else "✗"
            print(f"  score   : {r['score']:+.3f}  {status}")

    avg = sum(scores) / len(scores)
    if verbose:
        n_pass = sum(1 for s in scores if s > 0)
        print(f"\n  ► {level_name} avg_score = {avg:+.3f}  |  pass = {n_pass}/{len(scores)}")
    return avg


def run_all(
    model,
    tokenizer,
    cfg,
    verbose: bool = True,
    step   : int  = None,
) -> dict:
    """
    Chạy toàn bộ 3 cấp benchmark. Trả về dict 3 score + tổng hợp.

    Args:
        step: global_step hiện tại (chỉ để in log, không bắt buộc)

    Usage:
        results = run_all(model, tokenizer, cfg, verbose=True, step=global_step)
        # {"semantic": 6.25, "entity": 3.10, "fact": 1.40, "total": 10.75}
    """
    device  = next(model.parameters()).device
    max_seq = cfg.model.max_seq

    step_str = f"step {step}" if step is not None else "checkpoint"
    if verbose:
        print(f"\n{'═'*60}")
        print(f"  BENCHMARK  —  {step_str}")
        print(f"{'═'*60}")

    sem = run_benchmark(model, tokenizer, SEMANTIC_BENCH, "semantic", device, max_seq, verbose)
    ent = run_benchmark(model, tokenizer, ENTITY_BENCH,   "entity",   device, max_seq, verbose)
    fct = run_benchmark(model, tokenizer, FACT_BENCH,     "fact",     device, max_seq, verbose)

    total = sem + ent + fct

    if verbose:
        print(f"\n{'═'*60}")
        print(f"  SUMMARY  ({step_str})")
        print(f"{'─'*60}")
        print(f"  semantic : {sem:+.3f}")
        print(f"  entity   : {ent:+.3f}")
        print(f"  fact     : {fct:+.3f}")
        print(f"  total    : {total:+.3f}")
        print(f"{'═'*60}\n")

    return {"semantic": sem, "entity": ent, "fact": fct, "total": total}


# ══════════════════════════════════════════════════════════════════════════
# So sánh nhiều checkpoint
# ══════════════════════════════════════════════════════════════════════════

def compare_checkpoints(checkpoint_paths: list[str], verbose: bool = False) -> None:
    """
    Load từng checkpoint, chạy benchmark, in bảng so sánh.

    Usage:
        compare_checkpoints([
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
        r = run_all(model, tokenizer, cfg, verbose=verbose)
        r["checkpoint"] = path
        rows.append(r)

        # Giải phóng RAM ngay — không giữ nhiều model cùng lúc
        del model
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # In bảng tổng hợp
    print(f"\n{'═'*72}")
    print(f"  {'CHECKPOINT':<35} {'semantic':>8} {'entity':>8} {'fact':>8} {'total':>8}")
    print(f"{'─'*72}")
    for r in rows:
        name = r["checkpoint"].split("/")[-1]
        print(
            f"  {name:<35} {r['semantic']:>+8.3f} {r['entity']:>+8.3f} "
            f"{r['fact']:>+8.3f} {r['total']:>+8.3f}"
        )
    print(f"{'═'*72}")


# ══════════════════════════════════════════════════════════════════════════
# CLI nhanh
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Chạy nhanh:

        python benchmark.py checkpoints/chunk_10.pt
        python benchmark.py checkpoints/chunk_10.pt checkpoints/chunk_50.pt
    """
    import sys

    paths = sys.argv[1:]
    if not paths:
        print("Usage: python benchmark.py <checkpoint> [checkpoint2 ...]")
        sys.exit(1)

    if len(paths) == 1:
        from generate import load_model_for_inference
        model, tokenizer, cfg = load_model_for_inference(paths[0])
        run_all(model, tokenizer, cfg, verbose=True)
    else:
        compare_checkpoints(paths, verbose=False)