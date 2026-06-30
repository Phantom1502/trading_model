"""
Golden test set cho is_swept (liquidity sweep detector).

Mục đích: KHÔNG phải test code style thông thường, mà là verify giải thuật
có khớp đúng định nghĩa ICT thật hay không, trước khi tin dùng để generate
hàng loạt data (đúng tinh thần "Anchor cho giải thuật").

Mỗi test case là 1 chuỗi nến cụ thể, tự thiết kế bằng tay để biết trước
đáp án đúng — không lấy từ data thị trường thật, vì mục tiêu là kiểm
logic, không phải kiểm tính tổng quát.

Chạy: pytest test_swept.py -v
"""
import pytest
from app.utils.chart.candle_parser import Candle, CandleParser


def C(o, h, l, c):
    return Candle(o, h, l, c)


# ══════════════════════════════════════════════════════════════════
# CASE 1 — Bearish sweep rõ ràng (cơ bản nhất, phải pass)
# ══════════════════════════════════════════════════════════════════
def test_bearish_sweep_basic():
    """
    Idx:    0    1    2    3    4    5
    High:  10   12   15   11    9   16   <- nến 5 wick lên 16, xuyên qua swing high (15 @ idx2)
    Close:  9   11   13   10    8   13   <- nhưng đóng cửa 13, dưới 15 -> reject -> SWEEP

    Swing high tại idx2 (window=2): max(high[0..4]) = max(10,12,15,11,9) = 15 -> đúng là swing high.
    Idx2 chưa từng bị phá trước idx5 (close[3]=10, close[4]=8, đều < 15).
    => Kỳ vọng: BEARISH_SWEEP tại idx5, swing_idx=2, depth = 16-15 = 1.
    """
    candles = [
        C(9, 10, 8, 9),    # 0
        C(9, 12, 9, 11),   # 1
        C(11, 15, 10, 13), # 2  <- swing high
        C(13, 11, 9, 10),  # 3
        C(10, 9, 7, 8),    # 4
        C(8, 16, 7, 13),   # 5  <- sweep candle
    ]
    parser = CandleParser.from_candles(candles, swing_window=2)

    assert parser.is_swing_high(2) is True

    result = parser.is_swept(5)
    assert result is not None
    assert result["type"] == "BEARISH_SWEEP"
    assert result["swept_candle_idx"] == 5
    assert result["swing_idx"] == 2
    assert result["swing_level"] == 15
    assert result["depth"] == pytest.approx(1.0)


# ══════════════════════════════════════════════════════════════════
# CASE 2 — Bullish sweep rõ ràng (đối xứng case 1)
# ══════════════════════════════════════════════════════════════════
def test_bullish_sweep_basic():
    """
    Idx:    0    1    2    3    4    5
    Low:   20   18   15   19   21   14   <- nến 5 wick xuống 14, xuyên qua swing low (15 @ idx2)
    Close: 21   19   17   20   22   18   <- đóng cửa 18, trên 15 -> reject -> SWEEP

    Swing low tại idx2: min(low[0..4]) = min(20,18,15,19,21) = 15 -> đúng.
    => Kỳ vọng: BULLISH_SWEEP tại idx5, swing_idx=2, depth = 15-14 = 1.
    """
    candles = [
        C(21, 22, 20, 21), # 0
        C(21, 20, 18, 19), # 1
        C(19, 18, 15, 17), # 2  <- swing low
        C(17, 20, 19, 20), # 3
        C(20, 23, 21, 22), # 4
        C(22, 19, 14, 18), # 5  <- sweep candle
    ]
    parser = CandleParser.from_candles(candles, swing_window=2)

    assert parser.is_swing_low(2) is True

    result = parser.is_swept(5)
    assert result is not None
    assert result["type"] == "BULLISH_SWEEP"
    assert result["swing_idx"] == 2
    assert result["swing_level"] == 15
    assert result["depth"] == pytest.approx(1.0)


# ══════════════════════════════════════════════════════════════════
# CASE 3 — Không phải sweep: giá vượt qua NHƯNG đóng cửa cũng vượt luôn
# (đây là breakout/continuation, không phải reject -> KHÔNG được tính sweep)
# ══════════════════════════════════════════════════════════════════
def test_not_sweep_when_close_also_breaks():
    candles = [
        C(9, 10, 8, 9),
        C(9, 12, 9, 11),
        C(11, 15, 10, 13),  # swing high = 15
        C(13, 11, 9, 10),
        C(10, 9, 7, 8),
        C(8, 17, 9, 16),    # high=17 > 15, NHƯNG close=16 cũng > 15 -> breakout thật, không reject
    ]
    parser = CandleParser.from_candles(candles, swing_window=2)
    result = parser.is_swept(5)
    assert result is None, "Đóng cửa vượt qua swing -> breakout, không phải sweep"


# ══════════════════════════════════════════════════════════════════
# CASE 4 — Không có sweep: giá chưa chạm tới swing level
# ══════════════════════════════════════════════════════════════════
def test_not_sweep_when_no_wick_reaches_level():
    candles = [
        C(9, 10, 8, 9),
        C(9, 12, 9, 11),
        C(11, 15, 10, 13),  # swing high = 15
        C(13, 11, 9, 10),
        C(10, 9, 7, 8),
        C(8, 12, 7, 10),    # high=12, chưa chạm 15
    ]
    parser = CandleParser.from_candles(candles, swing_window=2)
    result = parser.is_swept(5)
    assert result is None


# ══════════════════════════════════════════════════════════════════
# CASE 5 — Swing đã bị phá từ trước (liquidity "cũ", không còn ý nghĩa)
# Đây là case quan trọng nhất: nếu code KHÔNG check already_broken,
# nó sẽ báo nhầm sweep ở 1 swing đã chết.
# ══════════════════════════════════════════════════════════════════
def test_no_sweep_on_already_broken_swing():
    """
    Idx:    0    1    2    3    4    5    6
    High:  10   12   15   11    9   18   19
    Close:  9   11   13   10    8   17   15

    Nến 5 đóng cửa 17 > 15 -> breakout thật, swing idx2 "chết" (liquidity đã lấy).
    Nến 6 wick lên 19 (qua mức 15 cũ) nhưng đây KHÔNG còn là sweep của idx2 nữa,
    vì idx2 đã hết hiệu lực từ nến 5.
    """
    candles = [
        C(9, 10, 8, 9),     # 0
        C(9, 12, 9, 11),    # 1
        C(11, 15, 10, 13),  # 2  <- swing high = 15
        C(13, 11, 9, 10),   # 3
        C(10, 9, 7, 8),     # 4
        C(8, 18, 8, 17),    # 5  <- breakout thật: close 17 > 15 -> swing idx2 đã "chết"
        C(17, 19, 14, 15),  # 6  <- wick lên 19 nhưng swing cũ đã hết hiệu lực
    ]
    parser = CandleParser.from_candles(candles, swing_window=2)

    # Xác nhận tiền đề: swing idx2 đã bị phá trước idx6
    swing = parser._find_active_swing_high(6, lookback=20, w=2)
    assert swing is None, (
        "Swing idx2 đã bị đóng cửa vượt qua ở nến 5 -> không còn là swing "
        "'active' để tính sweep mới ở nến 6"
    )

    result = parser.is_swept(6)
    assert result is None, "Không nên báo sweep trên 1 swing liquidity đã bị lấy từ trước"


# ══════════════════════════════════════════════════════════════════
# CASE 6 — Có 2 swing trong lookback, phải chọn đúng swing GẦN NHẤT còn nguyên
# ══════════════════════════════════════════════════════════════════
def test_picks_nearest_active_swing_not_older_one():
    """
    Idx:    0    1    2    3    4    5    6    7    8    9
    High:  10   12   20   11    9   13   17   11    9   18

    Swing high idx2=20 (cao hơn, xa hơn) và idx6=17 (thấp hơn, gần hơn).
    Cả 2 đều "active" (chưa bị phá). Nến 9 sweep nên match đúng swing GẦN NHẤT
    (idx6=17), không phải swing xa hơn nhưng cao hơn (idx2=20).
    """
    candles = [
        C(9, 10, 8, 9),     # 0
        C(9, 12, 9, 11),    # 1
        C(11, 20, 10, 18),  # 2  <- swing high cao = 20 (xa)
        C(18, 11, 9, 10),   # 3
        C(10, 9, 7, 8),     # 4
        C(8, 13, 7, 11),    # 5
        C(11, 17, 10, 16),  # 6  <- swing high gần = 17
        C(16, 11, 9, 10),   # 7
        C(10, 9, 7, 8),     # 8
        C(8, 18, 7, 14),    # 9  <- wick 18 xuyên cả 2 mức, close 14 < cả 2 -> reject
    ]
    parser = CandleParser.from_candles(candles, swing_window=2)

    assert parser.is_swing_high(2) is True
    assert parser.is_swing_high(6) is True

    result = parser.is_swept(9, lookback=20)
    assert result is not None
    assert result["swing_idx"] == 6, "Phải chọn swing GẦN NHẤT còn active (idx6=17), không phải idx2=20"
    assert result["swing_level"] == 17
    assert result["depth"] == pytest.approx(1.0)  # 18 - 17


# ══════════════════════════════════════════════════════════════════
# CASE 7 — Không đủ nến phía trước (i - w < 0) -> phải trả None, không lỗi
# ══════════════════════════════════════════════════════════════════
def test_not_enough_candles_before_returns_none():
    candles = [C(10, 12, 9, 11), C(11, 13, 10, 12)]
    parser = CandleParser.from_candles(candles, swing_window=2)
    result = parser.is_swept(1)  # i=1, w=2 -> i-w = -1 < 0
    assert result is None


# ══════════════════════════════════════════════════════════════════
# CASE 8 — Không có swing nào trong lookback (chart toàn đi ngang, không
# có cấu trúc đỉnh/đáy rõ) -> phải trả None
# ══════════════════════════════════════════════════════════════════
def test_no_swing_structure_returns_none():
    candles = [C(10, 11, 9, 10) for _ in range(8)]  # nến giống hệt nhau, không có swing
    parser = CandleParser.from_candles(candles, swing_window=2)
    result = parser.is_swept(7)
    assert result is None


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))