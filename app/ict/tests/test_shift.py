"""
tests/test_shift.py — Lớp 4: Shift / MSS (BOS vs CHoCH)
================================================================
Viết TRƯỚC khi is_shift() có logic thật (test-first, theo nguyên tắc đã
chốt trong spec Giai đoạn 3) — mục đích là ĐỊNH NGHĨA RÕ behavior mong
muốn trước, tránh đoán ngầm khi cài logic.

Quyết định thiết kế đã chốt (khác is_swept):
    - is_swept dùng WICK (High/Low) để xác định sweep — chạm là đủ.
    - is_shift dùng CLOSE để xác định phá cấu trúc — phải "đóng cửa xác
      nhận", wick chạm qua KHÔNG đủ (near_miss_wick_only_no_close_break).
    - BOS/CHoCH suy ra từ `trend` (caller truyền vào) + loại swing bị phá:
        trend=BULL, phá Swing High (Close > swing_level)  -> BOS  (tiếp diễn)
        trend=BULL, phá Swing Low  (Close < swing_level)  -> CHoCH (đảo chiều)
        trend=BEAR, phá Swing Low  (Close < swing_level)  -> BOS
        trend=BEAR, phá Swing High (Close > swing_level)  -> CHoCH
    - Tie-breaking: nhất quán với is_swept — chỉ báo cáo swing GẦN NHẤT
      trong lookback, dù nến có phá nhiều swing xa hơn cùng lúc.
"""

from app.ict.candle import Candle
from app.ict.parser import CandleParser
from app.ict.ict import is_shift, scan_all_shift


def _p(candles, sw=2):
    return CandleParser.from_candles(candles, swing_window=sw)


def _c(o, h, l, c):
    return Candle(open=o, high=h, low=l, close=c)


def test_clear_bos_same_direction():
    """Trend BULL, phá Swing High bằng CLOSE -> BOS (tiếp diễn)."""
    candles = [
        _c(495, 505, 490, 500),   # 0: padding
        _c(498, 508, 493, 503),   # 1: padding
        _c(505, 530, 500, 525),   # 2: swing high H=530
        _c(520, 518, 510, 515),   # 3
        _c(515, 513, 505, 510),   # 4
        _c(510, 540, 505, 535),   # 5: Close=535 > swing_level=530 -> BOS
    ]
    parser = _p(candles)
    result = is_shift(parser, 5, trend="BULL", lookback=10)
    assert result is not None
    assert result["type"] == "BOS"
    assert result["direction"] == "BULL"
    assert result["swing_idx"] == 2
    assert result["swing_level"] == 530
    assert result["broken_type"] == "HIGH"


def test_clear_choch_reversal():
    """Trend BULL, phá Swing Low bằng CLOSE -> CHoCH (đảo chiều)."""
    candles = [
        _c(525, 530, 520, 525),   # 0: padding
        _c(520, 525, 515, 520),   # 1: padding
        _c(505, 510, 480, 495),   # 2: swing low L=480
        _c(495, 505, 490, 500),   # 3
        _c(500, 510, 495, 505),   # 4
        _c(510, 515, 470, 475),   # 5: Close=475 < swing_level=480 -> CHoCH
    ]
    parser = _p(candles)
    result = is_shift(parser, 5, trend="BULL", lookback=10)
    assert result is not None
    assert result["type"] == "CHoCH"
    assert result["direction"] == "BEAR"
    assert result["swing_idx"] == 2
    assert result["swing_level"] == 480
    assert result["broken_type"] == "LOW"


def test_clear_bos_bear_trend():
    """Trend BEAR, phá Swing Low bằng CLOSE -> BOS (tiếp diễn xuống)."""
    candles = [
        _c(525, 530, 520, 525),   # 0: padding
        _c(520, 525, 515, 520),   # 1: padding
        _c(505, 510, 480, 495),   # 2: swing low L=480
        _c(495, 505, 490, 500),   # 3
        _c(500, 510, 495, 505),   # 4
        _c(510, 515, 470, 475),   # 5: Close=475 < swing_level=480 -> BOS (vì trend BEAR)
    ]
    parser = _p(candles)
    result = is_shift(parser, 5, trend="BEAR", lookback=10)
    assert result is not None
    assert result["type"] == "BOS"
    assert result["direction"] == "BEAR"


def test_clear_choch_bear_trend():
    """Trend BEAR, phá Swing High bằng CLOSE -> CHoCH (đảo chiều lên)."""
    candles = [
        _c(495, 505, 490, 500),   # 0: padding
        _c(498, 508, 493, 503),   # 1: padding
        _c(505, 530, 500, 525),   # 2: swing high H=530
        _c(520, 518, 510, 515),   # 3
        _c(515, 513, 505, 510),   # 4
        _c(510, 540, 505, 535),   # 5: Close=535 > swing_level=530 -> CHoCH (vì trend BEAR)
    ]
    parser = _p(candles)
    result = is_shift(parser, 5, trend="BEAR", lookback=10)
    assert result is not None
    assert result["type"] == "CHoCH"
    assert result["direction"] == "BULL"


def test_boundary_close_exactly_at_swing():
    """Close đúng bằng swing level (không vượt) -> None (strict '>')."""
    candles = [
        _c(495, 505, 490, 500),   # 0: padding
        _c(498, 508, 493, 503),   # 1: padding
        _c(505, 530, 500, 525),   # 2: swing high H=530
        _c(520, 518, 510, 515),   # 3
        _c(515, 513, 505, 510),   # 4
        _c(510, 535, 505, 530),   # 5: Close=530 == swing_level=530 -> KHÔNG phá
    ]
    parser = _p(candles)
    result = is_shift(parser, 5, trend="BULL", lookback=10)
    assert result is None


def test_near_miss_wick_only_no_close_break():
    """Wick (High) vượt qua swing nhưng Close KHÔNG vượt -> None.

    Đây là case quan trọng nhất phân biệt is_shift với is_swept: is_swept
    sẽ trả về sweep hợp lệ cho case này (vì dùng High), nhưng is_shift
    PHẢI trả về None vì cấu trúc chưa được "xác nhận" bằng giá đóng cửa.
    """
    candles = [
        _c(495, 505, 490, 500),   # 0: padding
        _c(498, 508, 493, 503),   # 1: padding
        _c(505, 530, 500, 525),   # 2: swing high H=530
        _c(520, 518, 510, 515),   # 3
        _c(515, 513, 505, 510),   # 4
        _c(510, 545, 505, 520),   # 5: High=545 > 530 (wick xuyên qua) nhưng Close=520 < 530
    ]
    parser = _p(candles)
    result = is_shift(parser, 5, trend="BULL", lookback=10)
    assert result is None

    # Đối chiếu: is_swept() PHẢI trả về sweep hợp lệ cho đúng case này
    from app.ict.ict import is_swept
    swept_result = is_swept(parser, 5, lookback=10)
    assert swept_result is not None
    assert swept_result["type"] == "SWEEP_HIGH"


def test_tie_breaking_multiple_swings_broken_same_candle():
    """1 nến phá nhiều swing cùng lúc (gần và xa) -> chỉ báo cáo swing GẦN
    NHẤT, nhất quán với hành vi is_swept (near_miss_multiple_swings_pick_nearest)."""
    candles = [
        _c(495, 505, 490, 500),   # 0: padding
        _c(498, 508, 493, 503),   # 1: padding
        _c(505, 525, 500, 515),   # 2: swing high H=525 (xa hơn)
        _c(515, 512, 505, 510),   # 3
        _c(510, 508, 500, 505),   # 4
        _c(500, 520, 495, 510),   # 5: swing high H=520 (gần hơn, thấp hơn)
        _c(510, 508, 500, 505),   # 6
        _c(505, 503, 495, 500),   # 7
        _c(495, 535, 490, 530),   # 8: Close=530 > CẢ 2 swing (520 và 525) -> chỉ báo swing gần nhất (idx 5)
    ]
    parser = _p(candles)
    result = is_shift(parser, 8, trend="BULL", lookback=10)
    assert result is not None
    assert result["swing_idx"] == 5   # gần nhất, không phải idx 2
    assert result["swing_level"] == 520


def test_edge_position_no_swing_in_lookback():
    """Không có swing nào trong lookback -> None, không lỗi."""
    candles = [
        _c(500, 505, 495, 500),
        _c(501, 506, 496, 501),
        _c(502, 507, 497, 502),
        _c(503, 508, 498, 503),
        _c(504, 520, 499, 515),
    ]
    parser = _p(candles)
    result = is_shift(parser, 4, trend="BULL", lookback=10)
    assert result is None


def test_clear_invalid_trend_raises():
    """trend không phải BULL/BEAR -> raise ValueError rõ ràng, không âm thầm sai."""
    candles = [
        _c(495, 505, 490, 500),
        _c(498, 508, 493, 503),
        _c(505, 530, 500, 525),
        _c(520, 518, 510, 515),
        _c(515, 513, 505, 510),
        _c(510, 540, 505, 535),
    ]
    parser = _p(candles)
    import pytest
    with pytest.raises(ValueError):
        is_shift(parser, 5, trend="SIDEWAYS", lookback=10)


# ══════════════════════════════════════════════════════════════════════
# scan_all_shift — trend evolution + broken tracking
# ══════════════════════════════════════════════════════════════════════

def test_clear_scan_trend_evolves_after_choch():
    """Sau 1 CHoCH (BULL -> BEAR), lần shift TIẾP THEO phải dùng trend MỚI
    (BEAR) để đánh giá BOS/CHoCH — không giữ nguyên trend ban đầu."""
    candles = [
        _c(525, 530, 520, 525),   # 0: padding
        _c(520, 525, 515, 520),   # 1: padding
        _c(505, 510, 480, 495),   # 2: swing low L=480 (mốc cho CHoCH đầu)
        _c(495, 505, 490, 500),   # 3
        _c(500, 510, 495, 505),   # 4
        _c(510, 515, 470, 475),   # 5: Close=475 < 480 -> CHoCH (BULL->BEAR), trend giờ = BEAR
        _c(470, 475, 460, 465),   # 6: padding (High=475, tránh chồng window swing high mới)
        _c(465, 480, 460, 475),   # 7: padding (High=480)
        _c(475, 495, 470, 490),   # 8: swing high H=495, window=[6..10] không chồng idx5 (High=515)
        _c(480, 485, 470, 475),   # 9: padding (High=485)
        _c(475, 470, 460, 465),   # 10: padding (High=470)
        _c(470, 505, 465, 500),   # 11: Close=500 > 495 -> trend=BEAR (đã evolve) -> CHoCH lần 2 (BEAR->BULL)
    ]
    parser = _p(candles)
    results = scan_all_shift(parser, initial_trend="BULL", lookback=10)

    assert len(results) == 2
    assert results[0]["type"] == "CHoCH"
    assert results[0]["direction"] == "BEAR"
    # Lần 2 PHẢI được đánh giá với trend=BEAR (đã evolve từ lần 1), nên
    # phá Swing High lúc này là CHoCH (đảo chiều BEAR->BULL), KHÔNG PHẢI
    # BOS (nếu code sai, không update trend, sẽ tính nhầm thành BOS vì
    # trend gốc vẫn tưởng là BULL trùng hướng phá High)
    assert results[1]["type"] == "CHoCH"
    assert results[1]["direction"] == "BULL"
    assert results[1]["swing_idx"] == 8


def test_tie_breaking_scan_broken_swing_not_reused():
    """Swing đã bị shift rồi -> KHÔNG được dùng làm mốc tham chiếu lại,
    giống nguyên tắc broken tracking của scan_all_swept()."""
    candles = [
        _c(495, 505, 490, 500),   # 0: padding
        _c(498, 508, 493, 503),   # 1: padding
        _c(505, 530, 500, 525),   # 2: swing high H=530
        _c(520, 518, 510, 515),   # 3
        _c(515, 513, 505, 510),   # 4
        _c(510, 540, 505, 535),   # 5: Close=535 > 530 -> BOS (trend BULL), swing idx 2 -> broken
        _c(520, 525, 510, 515),   # 6
        _c(510, 505, 495, 500),   # 7
        _c(495, 545, 490, 540),   # 8: Close=540 > 530 nữa, nhưng swing idx 2 đã broken -> KHÔNG tính lại
    ]
    parser = _p(candles)
    results = scan_all_shift(parser, initial_trend="BULL", lookback=10)

    assert len(results) == 1
    assert results[0]["shift_candle_idx"] == 5