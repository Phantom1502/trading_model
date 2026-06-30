"""
basic.py — Lớp 1-2: yếu tố đơn nến & 2 nến liên tiếp
=========================================================
Mọi ngưỡng dưới đây là PLACEHOLDER ban đầu (theo nguyên tắc spec mục 4:
không chốt ngưỡng cảm tính). Trước khi golden test ở tests/test_bull_bear.py
và tests/test_pin_bar.py được coi là "chốt cuối cùng", phải chạy thống kê
phân phối trên data thật (Giai đoạn 2 trong lộ trình) rồi điền số thật vào
đây, thay vì giữ giá trị đoán ban đầu.
"""

from typing import Optional

from .candle import Candle


# ══════════════════════════════════════════════════════════════════════
# Ngưỡng — CẦN THỐNG KÊ LẠI (xem spec mục 4, Giai đoạn 2)
# ══════════════════════════════════════════════════════════════════════

DOJI_THRESHOLD_BINS = 1     # |Close - Open| phải > giá trị này mới tính Bull/Bear
PIN_BAR_WICK_RATIO  = 0.6   # wick dài >= wick_ratio * range
PIN_BAR_BODY_RATIO  = 0.3   # body <= body_ratio * range


# ══════════════════════════════════════════════════════════════════════
# Lớp 1 — Bull / Bear / Doji
# ══════════════════════════════════════════════════════════════════════

def classify_direction(c: Candle, threshold_bins: int = DOJI_THRESHOLD_BINS) -> str:
    """
    Trả về "BULL" | "BEAR" | "DOJI".

    Dùng strict '>' (không phải '>='): body phải VƯỢT threshold mới được
    tính Bull/Bear, đúng bằng threshold vẫn là DOJI. Xem
    tests/test_bull_bear.py — boundary_at_threshold / boundary_just_below_threshold
    để biết rõ behavior này.
    """
    diff = c.close - c.open
    if diff > threshold_bins:
        return "BULL"
    if diff < -threshold_bins:
        return "BEAR"
    return "DOJI"


# ══════════════════════════════════════════════════════════════════════
# Lớp 1 — Pin Bar (Hammer / Shooting Star)
# ══════════════════════════════════════════════════════════════════════

def is_pin_bar(
    c: Candle,
    wick_ratio: float = PIN_BAR_WICK_RATIO,
    body_ratio: float = PIN_BAR_BODY_RATIO,
) -> Optional[str]:
    """
    Trả về "HAMMER" | "SHOOTING_STAR" | None.

    HAMMER         : lower wick dài, upper wick ngắn, body nhỏ.
    SHOOTING_STAR  : upper wick dài, lower wick ngắn, body nhỏ.
    Cả 2 wick đều dài (gần bằng nhau) -> None, không phải Pin Bar
    (xem tests/test_pin_bar.py — near_miss_both_wicks_long).
    """
    rng, body = c.range(), c.body()
    if body > body_ratio * rng:
        return None

    upper, lower = c.upper_wick(), c.lower_wick()

    if lower >= wick_ratio * rng and upper < lower * 0.5:
        return "HAMMER"
    if upper >= wick_ratio * rng and lower < upper * 0.5:
        return "SHOOTING_STAR"
    return None


# ══════════════════════════════════════════════════════════════════════
# Lớp 2 — Engulfing (2 nến liên tiếp)
# ══════════════════════════════════════════════════════════════════════

def is_engulfing(prev: Candle, curr: Candle) -> Optional[str]:
    """
    Trả về "BULLISH_ENGULFING" | "BEARISH_ENGULFING" | None.

    Dùng '<=' / '>=' (non-strict) cho điều kiện "nuốt trọn" — khớp đúng
    biên (Open[curr] == Close[prev]) vẫn tính là engulfing. Xem
    tests/test_engulfing.py — boundary_exact_engulf.

    Caller chịu trách nhiệm xử lý edge_position (index=0, không có prev) —
    hàm này không tự bắt index âm, nhận thẳng 2 Candle.
    """
    prev_dir = classify_direction(prev)
    curr_dir = classify_direction(curr)

    if prev_dir == "BEAR" and curr_dir == "BULL":
        if curr.open <= prev.close and curr.close >= prev.open:
            return "BULLISH_ENGULFING"

    if prev_dir == "BULL" and curr_dir == "BEAR":
        if curr.open >= prev.close and curr.close <= prev.open:
            return "BEARISH_ENGULFING"

    return None