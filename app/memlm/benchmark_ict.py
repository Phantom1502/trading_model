"""
benchmark_ict.py — Đánh giá MemoryLM trên hiểu biết ICT (Swept/FVG/Shift)
================================================================================
Dùng CÙNG phương pháp với benchmark.py (BenchItem + avg_logprob_per_token):
đo xem model có gán log-prob TRUNG BÌNH/TOKEN cao hơn cho completion ĐÚNG
(khớp fact thật từ app.ict detector) so với completion SAI (perturbed) hay
không — KHÔNG yêu cầu model tự generate đúng format từ đầu, phù hợp giai
đoạn pretrain (chưa SFT).

THIẾT KẾ QUAN TRỌNG — prompt dừng ngay tại "<eval>":
    prompt = chart + request + explanation + "<eval>"
    completion (positive/negative) = phần NỘI DUNG bên trong <eval>...</eval>

Cố tình dừng prompt ngay sau khi model đã thấy phần Lý giải (giống hệt lúc
train) rồi mới đo completion phần Eval — đây LÀ pattern thật model đã học
trong pretrain (Lý giải luôn đứng trước Eval trong 1 mẫu). Test này KHÔNG
nhằm cô lập "chart-only" (việc đó cần thiết kế khác, xem ghi chú cuối file)
mà nhằm đo: sau khi thấy đủ ngữ cảnh y hệt lúc train, model có phân biệt
được Eval ĐÚNG khớp chart/Lý giải hay không, so với 1 Eval SAI (lệch nến,
lệch loại pattern, lệch mức giá).

GROUND TRUTH luôn lấy từ chính app.ict detector (build_facts) qua
render_*_sample() thật — KHÔNG viết tay bất kỳ giá trị đúng/sai nào, giữ
đúng nguyên tắc golden test đã dùng xuyên suốt package app/ict/.
"""

import random
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from app.ict.candle import Candle
from app.ict.parser import CandleParser
from app.ict.facts import build_facts
from app.ict.render import (
    render_swept_sample, render_fvg_sample, render_shift_sample,
)

from benchmark import BenchItem, avg_logprob_per_token


# ══════════════════════════════════════════════════════════════════════
# Perturbation — sửa 1 field trong chuỗi eval (string-level, an toàn với
# cả case có tiền tố E1_/E2_ lẫn không tiền tố N==1)
# ══════════════════════════════════════════════════════════════════════

def _replace_first_field(eval_inner: str, key: str, new_value: Any) -> Optional[str]:
    """
    Thay giá trị field `key` ĐẦU TIÊN xuất hiện trong `eval_inner` (không
    phân biệt có tiền tố E1_/E2_ hay không, vì key luôn đứng ngay sau dấu
    '_' hoặc đầu chuỗi, kết thúc bằng '=').

    Trả về None nếu không tìm thấy field này (caller nên bỏ qua case đó
    thay vì tạo negative sai/rỗng).
    """
    pattern = re.compile(rf"(^|_){re.escape(key)}=([^\s]+)")
    match = pattern.search(eval_inner)
    if match is None:
        return None
    prefix = match.group(1)
    return eval_inner[:match.start()] + f"{prefix}{key}={new_value}" + eval_inner[match.end():]


def _extract_field(eval_inner: str, key: str) -> Optional[str]:
    """Lấy giá trị field `key` đầu tiên (chuỗi thô, chưa ép kiểu)."""
    pattern = re.compile(rf"(^|_){re.escape(key)}=([^\s]+)")
    match = pattern.search(eval_inner)
    return match.group(2) if match else None


# ══════════════════════════════════════════════════════════════════════
# Xây BenchItem từ 1 sample thật (render_*_sample output)
# ══════════════════════════════════════════════════════════════════════

def _make_prompt(sample: Dict[str, Any]) -> str:
    """Prompt dừng NGAY sau khi mở tag <eval> — completion là phần bên trong."""
    return f"{sample['chart']}\n{sample['request']}\n{sample['explanation']}\n<eval>"


def _inner_eval(sample: Dict[str, Any]) -> str:
    """Trích phần bên trong <eval>...</eval>, bỏ 2 tag."""
    e = sample["eval"]
    return e[len("<eval>"):-len("</eval>")]


def _build_swept_items(sample: Dict[str, Any], rng: random.Random) -> List[BenchItem]:
    inner = _inner_eval(sample)
    prompt = _make_prompt(sample)
    negatives = []

    # Perturb 1: sai loại (SWEEP_HIGH <-> SWEEP_LOW)
    t = _extract_field(inner, "T")
    if t in ("SWEEP_HIGH", "SWEEP_LOW"):
        wrong_t = "SWEEP_LOW" if t == "SWEEP_HIGH" else "SWEEP_HIGH"
        neg = _replace_first_field(inner, "T", wrong_t)
        if neg:
            negatives.append(neg)

    # Perturb 2: sai nến (lệch 2-5 nến so với thật)
    c = _extract_field(inner, "C")
    if c is not None:
        wrong_c = int(c) + rng.choice([-5, -3, 3, 5])
        neg = _replace_first_field(inner, "C", wrong_c)
        if neg:
            negatives.append(neg)

    # Perturb 3: sai mức swing level (lệch đáng kể, không phải do quantize)
    sl = _extract_field(inner, "SL")
    if sl is not None:
        wrong_sl = int(sl) + rng.choice([-30, -20, 20, 30])
        neg = _replace_first_field(inner, "SL", wrong_sl)
        if neg:
            negatives.append(neg)

    if not negatives:
        return []
    return [BenchItem(prompt=prompt, positive=[inner], negative=negatives, note="ict_swept")]


def _build_fvg_items(sample: Dict[str, Any], rng: random.Random) -> List[BenchItem]:
    inner = _inner_eval(sample)
    prompt = _make_prompt(sample)
    negatives = []

    t = _extract_field(inner, "T")
    if t in ("BULL", "BEAR"):
        wrong_t = "BEAR" if t == "BULL" else "BULL"
        neg = _replace_first_field(inner, "T", wrong_t)
        if neg:
            negatives.append(neg)

    c = _extract_field(inner, "C")
    if c is not None:
        wrong_c = int(c) + rng.choice([-5, -3, 3, 5])
        neg = _replace_first_field(inner, "C", wrong_c)
        if neg:
            negatives.append(neg)

    gs = _extract_field(inner, "GS")
    if gs is not None:
        wrong_gs = max(1, int(gs) + rng.choice([-15, -10, 10, 15]))
        neg = _replace_first_field(inner, "GS", wrong_gs)
        if neg:
            negatives.append(neg)

    if not negatives:
        return []
    return [BenchItem(prompt=prompt, positive=[inner], negative=negatives, note="ict_fvg")]


def _build_shift_items(sample: Dict[str, Any], rng: random.Random) -> List[BenchItem]:
    inner = _inner_eval(sample)
    prompt = _make_prompt(sample)
    negatives = []

    # Perturb: đảo cả loại lẫn hướng cùng lúc (BOS+BULL <-> CHoCH+BEAR là
    # cặp hợp lệ đối xứng — đảo cả 2 để negative vẫn "trông hợp lệ về mặt
    # cấu trúc", không phải lỗi rõ ràng dễ đoán mà không cần hiểu chart)
    t   = _extract_field(inner, "T")
    dr  = _extract_field(inner, "DIR")
    if t in ("BOS", "CHoCH") and dr in ("BULL", "BEAR"):
        wrong_t  = "CHoCH" if t == "BOS" else "BOS"
        wrong_dr = "BEAR" if dr == "BULL" else "BULL"
        neg = _replace_first_field(inner, "T", wrong_t)
        if neg:
            neg = _replace_first_field(neg, "DIR", wrong_dr)
        if neg:
            negatives.append(neg)

    c = _extract_field(inner, "C")
    if c is not None:
        wrong_c = int(c) + rng.choice([-5, -3, 3, 5])
        neg = _replace_first_field(inner, "C", wrong_c)
        if neg:
            negatives.append(neg)

    sl = _extract_field(inner, "SL")
    if sl is not None:
        wrong_sl = int(sl) + rng.choice([-30, -20, 20, 30])
        neg = _replace_first_field(inner, "SL", wrong_sl)
        if neg:
            negatives.append(neg)

    if not negatives:
        return []
    return [BenchItem(prompt=prompt, positive=[inner], negative=negatives, note="ict_shift")]


# ══════════════════════════════════════════════════════════════════════
# Sinh toàn bộ BenchItem từ 1 tập chart thật (list[list[Candle]])
# ══════════════════════════════════════════════════════════════════════

def build_ict_bench_items(
    charts: List[List[Candle]],
    seed: int = 42,
) -> Dict[str, List[BenchItem]]:
    """
    charts: list các cửa sổ nến thật (mỗi phần tử là list[Candle], khuyến
    nghị đúng 20 nến khớp cấu hình dataset đã dùng để pretrain).

    Trả về dict {"swept": [...], "fvg": [...], "shift": [...]} — mỗi
    value là list BenchItem, dùng trực tiếp với run_logprob_benchmark()
    có sẵn trong benchmark.py.

    CHỈ những chart THẬT SỰ có event mới sinh ra BenchItem (giống nguyên
    tắc SKIP của render.py khi 0 event) — số lượng BenchItem mỗi loại phụ
    thuộc bao nhiêu chart trong `charts` có đúng loại event đó.
    """
    rng = random.Random(seed)
    items: Dict[str, List[BenchItem]] = {"swept": [], "fvg": [], "shift": []}

    for candles in charts:
        parser = CandleParser.from_candles(candles, swing_window=2)
        raw = parser.raw_text
        initial_trend = rng.choice(["BULL", "BEAR"])
        facts = build_facts(parser, initial_trend=initial_trend, lookback=10)

        swept_sample = render_swept_sample(facts, raw, rng=rng)
        if swept_sample:
            items["swept"].extend(_build_swept_items(swept_sample, rng))

        fvg_sample = render_fvg_sample(facts, raw, rng=rng)
        if fvg_sample:
            items["fvg"].extend(_build_fvg_items(fvg_sample, rng))

        shift_sample = render_shift_sample(facts, raw, rng=rng)
        if shift_sample:
            items["shift"].extend(_build_shift_items(shift_sample, rng))

    return items


# ══════════════════════════════════════════════════════════════════════
# Runner — tích hợp cùng pattern run_logprob_benchmark() của benchmark.py
# ══════════════════════════════════════════════════════════════════════

def run_ict_benchmark(
    model,
    tokenizer,
    cfg,
    charts: List[List[Candle]],
    device=None,
    verbose: bool = True,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Chạy benchmark ICT đầy đủ (Swept/FVG/Shift), trả về dict điểm trung
    bình mỗi loại + "ict_total" (trung bình 3 loại) — cùng format trả về
    style run_all() để dễ ghép vào log_bench()/checkpoint tracking.
    """
    from benchmark import run_logprob_benchmark   # import trễ, tránh vòng lặp import nếu cần

    device = device or next(model.parameters()).device
    max_seq = cfg.model.max_seq

    items = build_ict_bench_items(charts, seed=seed)

    scores = {}
    for kind in ("swept", "fvg", "shift"):
        bench = items[kind]
        if not bench:
            if verbose:
                print(f"  ⚠ Không có BenchItem nào cho '{kind}' — kiểm tra lại `charts` đầu vào.")
            scores[f"ict_{kind}"] = 0.0
            continue
        scores[f"ict_{kind}"] = run_logprob_benchmark(
            model, tokenizer, bench, f"ict_{kind}", device, max_seq, verbose
        )

    present = [v for k, v in scores.items() if items[k.replace("ict_", "")]]
    scores["ict_total"] = sum(present) / len(present) if present else 0.0

    return scores