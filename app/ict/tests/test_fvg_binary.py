"""
tests/test_fvg_binary.py — Lớp 3: Fair Value Gap (binary)
=================================================================
Đây là nhóm test QUAN TRỌNG NHẤT cần xác nhận bằng thống kê thật (spec
mục 4, Giai đoạn 2) — đặc biệt case boundary_gap_1_bin. Giá trị gap dùng
trong test này là ví dụ minh họa cho ĐÚNG ĐỊNH NGHĨA hiện tại (strict
'>'/'<'), KHÔNG phải kết luận cuối cùng về ngưỡng "FVG có ý nghĩa giao
dịch" — việc đó cần dữ liệu thật để quyết định, xem ghi chú trong
basic.py và spec mục 4.
"""

from app.ict.candle import Candle
from app.ict.parser import CandleParser
from app.ict.structure import is_fvg


def _make_parser_for_fvg(c0: Candle, c1: Candle, c2: Candle, swing_window=2):
    """3 nến liên tiếp đặt ở cuối 1 chuỗi đủ dài để is_fvg(index=2) hợp lệ."""
    return CandleParser.from_candles([c0, c1, c2], swing_window=swing_window)


def test_clear_bullish_fvg():
    """Low[nến 3] > High[nến 1], chênh lệch rõ (>10 bin) -> BULL."""
    c0 = Candle(open=500, high=510, low=495, close=505)
    c1 = Candle(open=515, high=525, low=512, close=520)
    c2 = Candle(open=530, high=540, low=525, close=535)   # Low=525 > High[c0]=510, chênh 15 bin
    parser = _make_parser_for_fvg(c0, c1, c2)
    assert is_fvg(parser, 2) == "BULL"


def test_clear_bearish_fvg():
    """High[nến 3] < Low[nến 1], chênh lệch rõ -> BEAR."""
    c0 = Candle(open=540, high=545, low=530, close=535)
    c1 = Candle(open=525, high=528, low=515, close=520)
    c2 = Candle(open=510, high=512, low=500, close=505)   # High=512 < Low[c0]=530, chênh 18 bin
    parser = _make_parser_for_fvg(c0, c1, c2)
    assert is_fvg(parser, 2) == "BEAR"


def test_boundary_gap_1_bin():
    """Chênh lệch đúng 1 bin -> vẫn BULL/BEAR hợp lệ (theo strict '>'),
    đây là case cần điền số thật sau Giai đoạn 2 — hiện dùng giá trị
    minh họa đúng định nghĩa code, không phải kết luận về ý nghĩa thực tế."""
    c0 = Candle(open=500, high=510, low=495, close=505)
    c1 = Candle(open=512, high=515, low=511, close=513)
    c2 = Candle(open=515, high=520, low=511, close=518)   # Low=511, High[c0]=510, chênh đúng 1 bin
    parser = _make_parser_for_fvg(c0, c1, c2)
    assert is_fvg(parser, 2) == "BULL"


def test_boundary_gap_0_bin():
    """Low[nến 3] == High[nến 1] (chạm nhau, không vượt) -> None (strict '>')."""
    c0 = Candle(open=500, high=510, low=495, close=505)
    c1 = Candle(open=512, high=515, low=511, close=513)
    c2 = Candle(open=515, high=520, low=510, close=518)   # Low=510 == High[c0]=510, KHÔNG vượt
    parser = _make_parser_for_fvg(c0, c1, c2)
    assert is_fvg(parser, 2) is None


def test_near_miss_middle_candle_fills():
    """2 đầu có gap thật nhưng nến giữa có wick chạm vào vùng gap -> VẪN
    BULL/BEAR theo định nghĩa hiện tại (chỉ so nến 1 và nến 3, nến giữa
    KHÔNG ảnh hưởng kết quả binary) — xác nhận rõ phạm vi định nghĩa."""
    c0 = Candle(open=500, high=510, low=495, close=505)
    c1 = Candle(open=515, high=525, low=505, close=520)   # wick dưới (Low=505) chạm vào vùng gap [510,525]
    c2 = Candle(open=530, high=540, low=525, close=535)
    parser = _make_parser_for_fvg(c0, c1, c2)
    assert is_fvg(parser, 2) == "BULL"   # vẫn tính FVG, không bị ảnh hưởng bởi wick nến giữa


def test_edge_position_first_two_candles():
    """index < 2 (không đủ 3 nến) -> None, không lỗi index âm."""
    c0 = Candle(open=500, high=510, low=495, close=505)
    c1 = Candle(open=515, high=525, low=512, close=520)
    parser = CandleParser.from_candles([c0, c1])
    assert is_fvg(parser, 0) is None
    assert is_fvg(parser, 1) is None