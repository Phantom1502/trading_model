"""
tests/test_fvg_graded.py — Lớp 4: grade_fvg (fill_pct)
==========================================================
Behavior ĐÃ CHỐT: fill_pct phản ánh VỊ TRÍ HIỆN TẠI — chỉ overlap của
NẾN CUỐI CÙNG (tại upto_index) với vùng gap, KHÔNG tích lũy lịch sử.
Nếu giá từng lấp sâu rồi rời khỏi gap, fill_pct PHẢI giảm theo.
"""

from app.ict.candle import Candle
from app.ict.parser import CandleParser
from app.ict.ict import grade_fvg


def _make_fvg_parser(c0, c1, c2, extra_candles=None, sw=2):
    candles = [c0, c1, c2] + (extra_candles or [])
    return CandleParser.from_candles(candles, swing_window=sw)


def _c(o, h, l, c):
    return Candle(open=o, high=h, low=l, close=c)


def test_clear_unfilled():
    """FVG vừa hình thành (index=2), không có nến nào sau đó -> fill_pct == 0."""
    c0 = _c(500, 510, 495, 505)
    c1 = _c(515, 525, 512, 520)
    c2 = _c(530, 540, 525, 535)   # gap: [510, 525], chưa có nến nào lấp
    parser = _make_fvg_parser(c0, c1, c2, extra_candles=[], sw=2)
    result = grade_fvg(parser, 2, upto_index=2)
    assert result is not None
    assert result["type"] == "BULL"
    assert result["gap_low"] == 510
    assert result["gap_high"] == 525
    assert result["gap_size_bins"] == 15
    assert result["fill_pct"] == 0.0


def test_clear_fully_filled():
    """Giá quay lại lấp hoàn toàn vùng gap [510, 525] -> fill_pct == 100."""
    c0 = _c(500, 510, 495, 505)
    c1 = _c(515, 525, 512, 520)
    c2 = _c(530, 540, 525, 535)   # gap [510, 525]
    # nến sau lấp đầy: low=505 < 510, high=530 > 525, overlap = [510,525] = 15 bins = 100%
    c3 = _c(530, 530, 505, 515)
    parser = _make_fvg_parser(c0, c1, c2, extra_candles=[c3])
    result = grade_fvg(parser, 2, upto_index=3)
    assert result is not None
    assert result["fill_pct"] == 100.0


def test_boundary_partial_fill_50():
    """Giá lấp 1 phần vùng gap [510, 520] (size=10 bin).
    Nến lấp c3: low=508 (dưới gap_low), high=515 (giữa gap) -> overlap=[510,515]=5 bin = 50%.
    """
    # gap [510, 520]: c0.high=510, c2.low=520, c2.low > c0.high -> BULL FVG
    c0 = _c(500, 510, 495, 505)
    c1 = _c(515, 519, 511, 517)
    c2 = _c(525, 535, 520, 530)   # gap [510, 520], size=10
    c3 = _c(520, 515, 508, 512)   # high=515, low=508 -> overlap=[510,515]=5 bin = 50%
    parser = _make_fvg_parser(c0, c1, c2, extra_candles=[c3])
    result = grade_fvg(parser, 2, upto_index=3)
    assert result is not None
    assert result["gap_low"] == 510
    assert result["gap_high"] == 520
    assert result["gap_size_bins"] == 10
    assert result["fill_pct"] == 50.0


def test_near_miss_fill_then_extend():
    """Giá lấp 1 phần rồi đảo chiều RỜI KHỎI gap -> fill_pct phản ánh
    VỊ TRÍ HIỆN TẠI (chỉ nến CUỐI CÙNG tại upto_index), KHÔNG giữ lại mức
    lấp sâu nhất từng đạt được trong lịch sử.

    Gap [510, 525], size=15:
    - nến 3 (idx 3): lấp 7 bin (overlap=[510,517]) — MỨC LẤP SÂU NHẤT lịch sử (46.7%)
    - nến 4 (idx 4): giá rời khỏi vùng lấp sâu, overlap=[518,520] = 2 bin — VỊ TRÍ HIỆN TẠI

    upto_index=4: fill_pct PHẢI = 2/15*100 = 13.3%, KHÔNG PHẢI 46.7% (nến 3).
    """
    c0 = _c(500, 510, 495, 505)
    c1 = _c(515, 525, 512, 520)
    c2 = _c(530, 540, 525, 535)           # gap [510, 525]
    c3 = _c(520, 520, 508, 516)           # lấp sâu: overlap=[510,517] = 7 bin (46.7%)
    c4 = _c(516, 520, 518, 519)           # rời khỏi vùng lấp sâu: overlap=[518,520] = 2 bin (13.3%)
    parser = _make_fvg_parser(c0, c1, c2, extra_candles=[c3, c4])
    result = grade_fvg(parser, 2, upto_index=4)
    assert result is not None
    assert result["fill_pct"] == 13.3
    assert result["fill_pct"] != round(100.0 * 7 / 15, 1)   # explicit: KHÔNG phải mức lịch sử


def test_clear_current_position_ignores_earlier_history():
    """Nến TRƯỚC ĐÓ hoàn toàn không lấp (0%), nến CUỐI lấp 100% -> fill_pct
    phải phản ánh đúng 100% (nến cuối), không bị pha loãng bởi lịch sử 0%
    trước đó — xác nhận tính "chỉ nến cuối" theo cả 2 chiều tăng/giảm."""
    c0 = _c(500, 510, 495, 505)
    c1 = _c(515, 525, 512, 520)
    c2 = _c(530, 540, 525, 535)           # gap [510, 525]
    c3 = _c(530, 535, 528, 532)           # KHÔNG chạm gap: low=528 > gap_high=525 -> overlap=0
    c4 = _c(528, 530, 505, 510)           # lấp toàn bộ: low=505<510, high=530>525 -> 100%
    parser = _make_fvg_parser(c0, c1, c2, extra_candles=[c3, c4])
    result = grade_fvg(parser, 2, upto_index=4)
    assert result is not None
    assert result["fill_pct"] == 100.0


def test_edge_position_upto_equals_index():
    """upto_index == index (chưa có nến nào sau khi FVG hình thành) -> fill_pct = 0,
    không lỗi index ngoài phạm vi."""
    c0 = _c(500, 510, 495, 505)
    c1 = _c(515, 525, 512, 520)
    c2 = _c(530, 540, 525, 535)
    parser = _make_fvg_parser(c0, c1, c2)
    result = grade_fvg(parser, 2, upto_index=2)
    assert result is not None
    assert result["fill_pct"] == 0.0