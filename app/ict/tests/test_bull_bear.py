"""
tests/test_bull_bear.py — Lớp 1: Bull/Bear/Doji
======================================================
LƯU Ý: DOJI_THRESHOLD_BINS hiện là placeholder (xem basic.py docstring) —
case boundary_at_threshold dùng đúng giá trị constant này, KHÔNG hardcode
số trong test, để khi Giai đoạn 2 (thống kê) cập nhật constant, test này
tự động phản ánh đúng theo giá trị mới mà không cần sửa tay.
"""

from app.ict.candle import Candle
from app.ict.basic import classify_direction, DOJI_THRESHOLD_BINS


def test_clear_bull():
    """O=500, C=520 (chênh lệch lớn) -> BULL."""
    c = Candle(open=500, high=525, low=495, close=520)
    assert classify_direction(c) == "BULL"


def test_clear_bear():
    """O=520, C=500 -> BEAR."""
    c = Candle(open=520, high=525, low=495, close=500)
    assert classify_direction(c) == "BEAR"


def test_boundary_at_threshold():
    """C - O đúng bằng DOJI_THRESHOLD_BINS -> vẫn DOJI (strict '>', không '>=')."""
    c = Candle(open=500, high=505, low=495, close=500 + DOJI_THRESHOLD_BINS)
    assert classify_direction(c) == "DOJI"


def test_boundary_just_above_threshold():
    """C - O = threshold + 1 -> BULL (vượt strict)."""
    c = Candle(open=500, high=505, low=495, close=500 + DOJI_THRESHOLD_BINS + 1)
    assert classify_direction(c) == "BULL"


def test_boundary_just_below_threshold():
    """C - O = threshold - 1 -> DOJI."""
    c = Candle(open=500, high=505, low=495, close=500 + DOJI_THRESHOLD_BINS - 1)
    assert classify_direction(c) == "DOJI"


def test_near_miss_doji_with_long_wick():
    """O=500, C=501 (gần Doji) nhưng H, L cách xa -> vẫn DOJI theo direction
    (wick không ảnh hưởng kết quả Bull/Bear/Doji, chỉ ảnh hưởng Pin Bar)."""
    c = Candle(open=500, high=600, low=400, close=501)
    assert classify_direction(c) == "DOJI"