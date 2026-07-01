"""
tests/test_facts.py — build_facts(): smoke test tích hợp toàn bộ detector
=================================================================================
Không phải golden test chi tiết từng detector (đã có riêng ở các file
khác) — chỉ xác nhận build_facts() gọi đúng interface, trả về đúng cấu
trúc dict, và initial_trend BẮT BUỘC phải truyền vào tường minh.
"""

import pytest

from app.ict.candle import Candle
from app.ict.parser import CandleParser
from app.ict.facts import build_facts


def _c(o, h, l, c):
    return Candle(open=o, high=h, low=l, close=c)


def test_clear_facts_structure():
    """build_facts trả về đúng 5 field, mỗi field đúng kiểu list/int."""
    candles = [
        _c(495, 505, 490, 500),
        _c(498, 508, 493, 503),
        _c(505, 530, 500, 525),   # swing high
        _c(520, 518, 510, 515),
        _c(515, 513, 505, 510),
        _c(510, 535, 505, 520),   # sweep high
    ]
    parser = CandleParser.from_candles(candles, swing_window=2)
    facts = build_facts(parser, initial_trend="BULL", lookback=10)

    assert facts["n_candles"] == 6
    assert isinstance(facts["swept"], list)
    assert isinstance(facts["fvg"], list)
    assert isinstance(facts["shift"], list)
    assert isinstance(facts["relations"], list)


def test_clear_facts_requires_initial_trend():
    """initial_trend là tham số BẮT BUỘC (positional/keyword), không có default —
    gọi thiếu phải raise TypeError, không được âm thầm dùng giá trị đoán."""
    candles = [_c(500, 510, 495, 505)]
    parser = CandleParser.from_candles(candles)
    with pytest.raises(TypeError):
        build_facts(parser)   # thiếu initial_trend


def test_clear_facts_shift_populated():
    """Chart có shift hợp lệ -> field "shift" trong facts phải khác rỗng
    (xác nhận build_facts thực sự gọi scan_all_shift, không còn bỏ trống)."""
    candles = [
        _c(495, 505, 490, 500),
        _c(498, 508, 493, 503),
        _c(505, 530, 500, 525),   # swing high H=530
        _c(520, 518, 510, 515),
        _c(515, 513, 505, 510),
        _c(510, 540, 505, 535),   # Close=535 > 530 -> BOS
    ]
    parser = CandleParser.from_candles(candles, swing_window=2)
    facts = build_facts(parser, initial_trend="BULL", lookback=10)

    assert len(facts["shift"]) == 1
    assert facts["shift"][0]["type"] == "BOS"