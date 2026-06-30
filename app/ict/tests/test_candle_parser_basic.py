"""
tests/test_candle_parser_basic.py — Lớp 0: parse + đếm nến
==================================================================
Theo bảng golden test spec mục 9, nhóm Lớp 0 / test_candle_parser_basic.py.
"""

from app.ict.candle import parse_candles
from app.ict.parser import CandleParser


def test_clear_parse_count():
    """5 nến hợp lệ -> len(parser) == 5."""
    raw = (
        "<chart> O_500 H_510 L_490 C_505 "
        "O_505 H_515 L_495 C_510 "
        "O_510 H_520 L_500 C_515 "
        "O_515 H_525 L_505 C_520 "
        "O_520 H_530 L_510 C_525 </chart>"
    )
    parser = CandleParser(raw)
    assert len(parser) == 5


def test_clear_parse_values():
    """1 nến O_500 H_510 L_490 C_505 -> Candle đúng 4 giá trị."""
    raw = "<chart> O_500 H_510 L_490 C_505 </chart>"
    candles = parse_candles(raw)
    assert len(candles) == 1
    c = candles[0]
    assert (c.open, c.high, c.low, c.close) == (500, 510, 490, 505)


def test_boundary_single_candle():
    """Chuỗi chỉ 1 nến -> parse thành công, không lỗi index."""
    raw = "<chart> O_500 H_510 L_490 C_505 </chart>"
    parser = CandleParser(raw)
    assert len(parser) == 1
    assert parser[0].open == 500


def test_near_miss_malformed_token():
    """Thiếu 1 trong 4 token (chỉ có O H L, thiếu C) -> bỏ qua nến lỗi.

    Quyết định behavior: regex parse_candles() yêu cầu đủ 4 token liên
    tiếp đúng thứ tự O H L C, nến lỗi (thiếu C) sẽ KHÔNG match -> bị bỏ
    qua hoàn toàn (không phải raise lỗi). Test này khẳng định rõ lựa
    chọn "im lặng bỏ qua" thay vì "raise" để code gen data không crash
    giữa chừng khi gặp dữ liệu nhiễu.
    """
    raw = "<chart> O_500 H_510 L_490 O_520 H_530 L_510 C_525 </chart>"
    candles = parse_candles(raw)
    # Chỉ nến thứ 2 (đủ 4 field) được parse, nến đầu (thiếu C) bị bỏ qua
    assert len(candles) == 1
    assert candles[0].open == 520


def test_edge_position_empty_chart():
    """<chart></chart> rỗng -> len == 0, không crash."""
    raw = "<chart></chart>"
    parser = CandleParser(raw)
    assert len(parser) == 0