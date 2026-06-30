"""
tests/test_slice.py — Lớp 0: CandleParser.slice()
========================================================
"""

from app.ict.candle import Candle
from app.ict.parser import CandleParser


def _make_parser(n=20, swing_window=2):
    candles = [Candle(open=500 + i, high=510 + i, low=490 + i, close=505 + i) for i in range(n)]
    return CandleParser.from_candles(candles, swing_window=swing_window)


def test_clear_slice_middle():
    """Cắt [5:10] từ chuỗi 20 nến -> 5 nến đúng, đúng thứ tự."""
    parser = _make_parser(20)
    sub = parser.slice(5, 10)
    assert len(sub) == 5
    assert sub[0].open == parser[5].open
    assert sub[-1].open == parser[9].open


def test_boundary_slice_full_range():
    """Cắt [0:n] -> kết quả giống parser gốc."""
    parser = _make_parser(20)
    sub = parser.slice(0, len(parser))
    assert len(sub) == len(parser)
    assert [c.open for c in sub.candles] == [c.open for c in parser.candles]


def test_edge_position_slice_at_end():
    """Cắt [n-1:n] -> 1 nến cuối cùng, không lỗi."""
    parser = _make_parser(20)
    sub = parser.slice(len(parser) - 1, len(parser))
    assert len(sub) == 1
    assert sub[0].open == parser[-1].open


def test_tie_breaking_raw_text_rebuild():
    """Slice rồi build lại raw_text -> parse lại ra đúng candles đã cắt."""
    parser = _make_parser(20)
    sub = parser.slice(5, 10)
    reparsed = CandleParser(sub.raw_text)
    assert len(reparsed) == len(sub)
    assert [c.open for c in reparsed.candles] == [c.open for c in sub.candles]