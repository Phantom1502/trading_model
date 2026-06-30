"""
tests/test_engulfing.py — Lớp 2: Engulfing (2 nến liên tiếp)
====================================================================
"""

from app.ict.candle import Candle
from app.ict.basic import is_engulfing


def test_clear_bullish_engulfing():
    """Nến 1 Bear nhỏ, nến 2 Bull lớn nuốt trọn thân nến 1 -> BULLISH_ENGULFING."""
    prev = Candle(open=510, high=515, low=505, close=507)   # Bear nhỏ: 510 -> 507
    curr = Candle(open=506, high=525, low=504, close=515)   # Bull lớn: 506 -> 515, nuốt trọn [507,510]
    assert is_engulfing(prev, curr) == "BULLISH_ENGULFING"


def test_clear_bearish_engulfing():
    """Nến 1 Bull nhỏ, nến 2 Bear lớn nuốt trọn -> BEARISH_ENGULFING."""
    prev = Candle(open=505, high=510, low=503, close=508)   # Bull nhỏ: 505 -> 508
    curr = Candle(open=509, high=511, low=495, close=500)   # Bear lớn: 509 -> 500, nuốt trọn [505,508]
    assert is_engulfing(prev, curr) == "BEARISH_ENGULFING"


def test_boundary_exact_engulf():
    """Open[curr] == Close[prev] và Close[curr] == Open[prev] (khớp biên đúng
    bằng, không vượt) -> vẫn tính engulfing (non-strict '<='/'>=')."""
    prev = Candle(open=510, high=515, low=505, close=507)   # Bear: 510 -> 507
    curr = Candle(open=507, high=520, low=505, close=510)   # Bull: 507 -> 510, khớp đúng biên
    assert is_engulfing(prev, curr) == "BULLISH_ENGULFING"


def test_near_miss_almost_engulf():
    """Thân nến 2 thiếu 1 bin để nuốt trọn nến 1 -> None."""
    prev = Candle(open=510, high=515, low=505, close=507)   # Bear: 510 -> 507
    curr = Candle(open=507, high=520, low=505, close=509)   # Bull: 507 -> 509, KHÔNG đủ vượt 510 -> thiếu 1 bin
    assert is_engulfing(prev, curr) is None


def test_near_miss_same_direction():
    """2 nến cùng hướng (cả 2 Bull) -> None."""
    prev = Candle(open=500, high=510, low=498, close=506)   # Bull
    curr = Candle(open=505, high=520, low=503, close=515)   # Bull
    assert is_engulfing(prev, curr) is None


def test_edge_position_first_candle():
    """index = 0 (không có nến trước) -> caller không gọi is_engulfing với
    prev=None; hàm này nhận thẳng 2 Candle nên test ở mức caller (facts.py)
    phải tự bắt index=0 trước khi gọi. Test ở đây xác nhận hàm không tự ý
    raise khi 2 Candle hợp lệ nhưng không thỏa điều kiện engulfing."""
    same = Candle(open=500, high=510, low=495, close=505)
    assert is_engulfing(same, same) is None