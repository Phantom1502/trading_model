"""
structure.py — Lớp 3: Swing High/Low, FVG (binary)
=========================================================
Khác basic.py (chỉ cần 1-2 Candle rời), các hàm ở đây cần CandleParser
đầy đủ vì phải so sánh nến không liền kề trong 1 cửa sổ.
"""

from typing import Optional

from .parser import CandleParser


# ══════════════════════════════════════════════════════════════════════
# Swing High / Swing Low
# ══════════════════════════════════════════════════════════════════════

def is_swing_high(parser: CandleParser, index: int, window: Optional[int] = None) -> bool:
    """
    True nếu High[index] là MAX trong cửa sổ [index-window, index+window].

    Lưu ý tie_breaking: nếu có >1 nến cùng đạt giá trị max trong window,
    TẤT CẢ đều trả True (so sánh bằng '==', không phải 'duy nhất max') —
    đây là hành vi đã xác nhận có chủ đích, xem tests/test_swing.py,
    case tie_breaking_equal_high.

    edge_position: không đủ context window 2 bên (đầu/cuối chuỗi) -> False.
    """
    w = window if window is not None else parser.swing_window
    n = len(parser)
    if index - w < 0 or index + w >= n:
        return False
    target = parser[index].high
    return target == max(parser[i].high for i in range(index - w, index + w + 1))


def is_swing_low(parser: CandleParser, index: int, window: Optional[int] = None) -> bool:
    """True nếu Low[index] là MIN trong cửa sổ — đối xứng với is_swing_high."""
    w = window if window is not None else parser.swing_window
    n = len(parser)
    if index - w < 0 or index + w >= n:
        return False
    target = parser[index].low
    return target == min(parser[i].low for i in range(index - w, index + w + 1))


# ══════════════════════════════════════════════════════════════════════
# Fair Value Gap (binary) — xét 3 nến liên tiếp [index-2, index-1, index]
# ══════════════════════════════════════════════════════════════════════

def is_fvg(parser: CandleParser, index: int) -> Optional[str]:
    """
    Trả về "BULL" | "BEAR" | None, xét bộ 3 nến (index-2, index-1, index).

    Chỉ so sánh nến ĐẦU (index-2) và nến THỨ BA (index) — nến giữa
    (index-1) KHÔNG ảnh hưởng tới kết quả binary dù wick của nó có chạm
    vào vùng gap hay không. Đây là phạm vi định nghĩa đã xác nhận có chủ
    đích, xem tests/test_fvg_binary.py, case near_miss_middle_candle_fills.

    Dùng strict '>' / '<' — chạm đúng bằng nhau (gap=0) KHÔNG tính là FVG,
    xem case boundary_gap_0_bin.

    edge_position: index < 2 (không đủ 3 nến) -> None.
    """
    if index - 2 < 0:
        return None
    c0 = parser[index - 2]
    c2 = parser[index]
    if c2.low > c0.high:
        return "BULL"
    if c2.high < c0.low:
        return "BEAR"
    return None