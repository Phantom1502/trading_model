"""
tests/test_swing.py — Lớp 3: Swing High / Swing Low
===========================================================
"""

from app.ict.candle import Candle
from app.ict.parser import CandleParser
from app.ict.structure import is_swing_high, is_swing_low


def _make_parser(highs, lows, swing_window=2):
    """Tạo parser từ list high/low, open/close không quan trọng cho test này."""
    candles = [Candle(open=h - 5, high=h, low=l, close=h - 5) for h, l in zip(highs, lows)]
    return CandleParser.from_candles(candles, swing_window=swing_window)


def test_clear_swing_high():
    """1 nến cao vượt hẳn các nến lân cận trong swing_window -> True."""
    highs = [500, 505, 530, 505, 500]   # nến giữa (idx 2) cao vượt hẳn
    lows  = [490, 495, 520, 495, 490]
    parser = _make_parser(highs, lows, swing_window=2)
    assert is_swing_high(parser, 2) is True


def test_clear_swing_low():
    """Tương tự cho đáy."""
    highs = [510, 505, 480, 505, 510]
    lows  = [500, 495, 470, 495, 500]   # nến giữa (idx 2) thấp vượt hẳn
    parser = _make_parser(highs, lows, swing_window=2)
    assert is_swing_low(parser, 2) is True


def test_tie_breaking_equal_high():
    """2 nến trong window có High bằng nhau, cùng là max -> CẢ 2 đều True
    (theo định nghĩa target == max(...) hiện tại, không phải "duy nhất max").
    Đây là hành vi đã xác nhận có chủ đích trong spec mục 9."""
    highs = [500, 505, 530, 505, 530, 505, 500]   # idx 2 và idx 4 cùng = 530
    lows  = [490, 495, 520, 495, 520, 495, 490]
    parser = _make_parser(highs, lows, swing_window=2)
    assert is_swing_high(parser, 2) is True
    assert is_swing_high(parser, 4) is True


def test_edge_position_window_start():
    """index < swing_window (không đủ nến bên trái) -> False."""
    highs = [530, 505, 500, 495, 490]   # idx 0 là cao nhất nhưng không đủ context trái
    lows  = [520, 495, 490, 485, 480]
    parser = _make_parser(highs, lows, swing_window=2)
    assert is_swing_high(parser, 0) is False
    assert is_swing_high(parser, 1) is False


def test_edge_position_window_end():
    """index + swing_window >= n (không đủ nến bên phải) -> False."""
    highs = [490, 495, 500, 505, 530]   # idx cuối là cao nhất nhưng không đủ context phải
    lows  = [480, 485, 490, 495, 520]
    parser = _make_parser(highs, lows, swing_window=2)
    n = len(parser)
    assert is_swing_high(parser, n - 1) is False
    assert is_swing_high(parser, n - 2) is False


def test_near_miss_local_not_global_high():
    """Nến cao nhất trong swing_window hẹp nhưng KHÔNG phải cao nhất toàn
    chart -> vẫn True (đúng định nghĩa swing CỤC BỘ, không phải global)."""
    highs = [600, 505, 510, 505, 500, 495, 490]
    lows  = [590, 495, 500, 495, 490, 485, 480]
    # idx 0 = 600 là cao nhất toàn chart nhưng không đủ context (edge) -> bỏ qua
    # idx 2 = 510, cao nhất CỤC BỘ trong window [0..4], dù thấp hơn idx 0 toàn cục
    parser = _make_parser(highs, lows, swing_window=2)
    assert is_swing_high(parser, 2) is True