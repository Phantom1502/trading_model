"""
ict.py — Lớp 4: yếu tố ICT nâng cao (Swept, FVG graded, Shift/MSS)
=========================================================================
Đây là phần "đắt" của package — mọi hàm trả về dict số hóa đầy đủ
(không chỉ bool), đúng theo quy ước is_swept() đã có trong code náp gốc
(spec mục 13): {"type", "swept_candle_idx", "swing_idx", "swing_level", "depth"}.

Trạng thái triển khai (theo lộ trình spec mục 10, Giai đoạn 3):
    is_swept   — đã có 8/8 golden test pass ở bản náp gốc, viết lại tại đây
                 theo cùng logic, BỔ SUNG 2 case còn thiếu (multiple swings,
                 re-sweep cùng swing) — xem tests/test_swept.py.
    grade_fvg  — CHƯA verify bằng golden test thật, khung tham số để rõ ý đồ.
    is_shift   — CHƯA triển khai logic — chỉ để stub + raise NotImplementedError,
                 PHẢI viết test trước (tests/test_shift.py) để định nghĩa rõ
                 behavior BOS vs CHoCH trước khi viết logic thật (test-first
                 theo đúng nguyên tắc đã chốt trong spec, Giai đoạn 3).
"""

from typing import Optional, List, Dict, Any

from .parser import CandleParser
from .structure import is_swing_high, is_swing_low
from .basic import classify_direction


# lookback=10 ĐÃ XÁC NHẬN bằng thống kê (Giai đoạn 2, N=19.4M nến, swing_window=2).
# Tần suất sweep bão hòa rõ rệt sau lookback=10-15: lb=5→8.59%, lb=10→11.72%
# (+3.13pp), lb=15→12.17% (+0.45pp), lb=20→12.23% (+0.06pp, gần như nhiễu).
# lookback=10 capture ~96% giá trị khả dụng tối đa (so với lb=20) với chi phí
# tìm kiếm thấp hơn. Depth distribution gần như bất biến theo lookback (giống
# pattern prominence bất biến theo swing_window) — xác nhận lookback không ảnh
# hưởng "chất lượng" sweep phát hiện được, chỉ ảnh hưởng số lượng. Xem README.md
# mục Swept.
SWEPT_LOOKBACK_DEFAULT = 10


# ══════════════════════════════════════════════════════════════════════
# Swept
# ══════════════════════════════════════════════════════════════════════

def _find_active_swing_high(
    parser: CandleParser, upto_index: int, lookback: int, window: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """
    Tìm swing high GẦN NHẤT (theo thời gian, không phải cao nhất) trong
    [upto_index - lookback, upto_index), CHƯA bị đánh dấu broken.

    Quét NGƯỢC từ upto_index-1 về trước -> đảm bảo lấy đúng swing gần nhất
    khi có nhiều swing hợp lệ trong lookback (tests/test_swept.py,
    case near_miss_multiple_swings_pick_nearest).
    """
    start = max(0, upto_index - lookback)
    for i in range(upto_index - 1, start - 1, -1):
        if is_swing_high(parser, i, window):
            return {"swing_idx": i, "swing_level": parser[i].high, "broken": False}
    return None


def _find_active_swing_low(
    parser: CandleParser, upto_index: int, lookback: int, window: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    start = max(0, upto_index - lookback)
    for i in range(upto_index - 1, start - 1, -1):
        if is_swing_low(parser, i, window):
            return {"swing_idx": i, "swing_level": parser[i].low, "broken": False}
    return None


def is_swept(
    parser: CandleParser,
    index: int,
    lookback: int = SWEPT_LOOKBACK_DEFAULT,
    window: Optional[int] = None,
    _broken_swing_indices: Optional[set] = None,
) -> Optional[Dict[str, Any]]:
    """
    Kiểm tra nến tại `index` có sweep (quét thanh khoản) qua swing high/low
    gần nhất trong `lookback` nến trước đó hay không.

    Trả về dict đầy đủ nếu có sweep:
        {"type": "SWEEP_HIGH"|"SWEEP_LOW", "swept_candle_idx", "swing_idx",
         "swing_level", "depth"}
    None nếu không sweep.

    `_broken_swing_indices`: set các swing_idx ĐÃ bị phá trước đó trong
    cùng 1 lần quét toàn chart — truyền vào để đảm bảo swing đã phá rồi
    không được tính sweep lại (tests/test_swept.py, case
    tie_breaking_re_sweep_same_swing). Khi gọi đơn lẻ (không quét toàn
    chart) có thể bỏ qua tham số này.
    """
    broken = _broken_swing_indices or set()

    swing_high = _find_active_swing_high(parser, index, lookback, window)
    if swing_high and swing_high["swing_idx"] not in broken:
        if parser[index].high > swing_high["swing_level"]:
            return {
                "type"            : "SWEEP_HIGH",
                "swept_candle_idx": index,
                "swing_idx"       : swing_high["swing_idx"],
                "swing_level"     : swing_high["swing_level"],
                "depth"           : parser[index].high - swing_high["swing_level"],
            }

    swing_low = _find_active_swing_low(parser, index, lookback, window)
    if swing_low and swing_low["swing_idx"] not in broken:
        if parser[index].low < swing_low["swing_level"]:
            return {
                "type"            : "SWEEP_LOW",
                "swept_candle_idx": index,
                "swing_idx"       : swing_low["swing_idx"],
                "swing_level"     : swing_low["swing_level"],
                "depth"           : swing_low["swing_level"] - parser[index].low,
            }

    return None


def scan_all_swept(parser: CandleParser, lookback: int = SWEPT_LOOKBACK_DEFAULT, window: Optional[int] = None) -> List[dict]:
    """
    Quét toàn chart, đảm bảo 1 swing đã bị sweep/phá thì KHÔNG được tính
    sweep lại lần sau (cộng dồn `broken` qua các vòng lặp) — đây là lý do
    is_swept() đơn lẻ không tự đủ để quét cả chart đúng, phải dùng hàm này.
    """
    broken: set = set()
    results = []
    for i in range(len(parser)):
        r = is_swept(parser, i, lookback=lookback, window=window, _broken_swing_indices=broken)
        if r:
            broken.add(r["swing_idx"])
            broken.add(r["swept_candle_idx"])   # nến thực hiện sweep cũng không được dùng làm swing mới
            results.append(r)
    return results


# ══════════════════════════════════════════════════════════════════════
# FVG graded — CHƯA verify bằng golden test, khung tham số
# ══════════════════════════════════════════════════════════════════════

def grade_fvg(parser: CandleParser, index: int, upto_index: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """
    Mở rộng is_fvg() (structure.py) thành graded: size gap (bin) + % đã lấp
    tính tại thời điểm `upto_index` (mặc định = cuối parser, tức "hiện tại").

    fill_pct phản ánh VỊ TRÍ HIỆN TẠI — chỉ overlap của NẾN CUỐI CÙNG
    (tại upto_index) với vùng gap, KHÔNG tích lũy lịch sử. Quyết định đã
    chốt: nếu giá từng lấp sâu rồi đảo chiều rời khỏi gap, fill_pct PHẢI
    giảm theo, không giữ lại mức lấp sâu nhất từng đạt được (đã xác nhận
    qua tests/test_fvg_graded.py, case near_miss_fill_then_extend).

    upto_index == index (chưa có nến nào sau khi FVG hình thành) -> fill_pct = 0.
    """
    from .structure import is_fvg

    fvg_type = is_fvg(parser, index)
    if fvg_type is None:
        return None

    c0 = parser[index - 2]
    c2 = parser[index]
    gap_low, gap_high = (c0.high, c2.low) if fvg_type == "BULL" else (c2.high, c0.low)
    gap_size = gap_high - gap_low

    upto = upto_index if upto_index is not None else len(parser) - 1

    if upto <= index:
        filled = 0
    else:
        # CHỈ nến cuối cùng (upto), không quét toàn bộ range index+1..upto
        c = parser[upto]
        overlap_low  = max(c.low,  gap_low)
        overlap_high = min(c.high, gap_high)
        filled = (overlap_high - overlap_low) if overlap_high > overlap_low else 0

    fill_pct = round(100.0 * filled / gap_size, 1) if gap_size > 0 else 100.0

    return {
        "type"        : fvg_type,
        "fvg_candle_idx": index,
        "gap_low"     : gap_low,
        "gap_high"    : gap_high,
        "gap_size_bins": gap_size,
        "fill_pct"    : min(fill_pct, 100.0),
    }


# ══════════════════════════════════════════════════════════════════════
# Shift / MSS — CHƯA triển khai, viết test trước theo nguyên tắc spec
# ══════════════════════════════════════════════════════════════════════

def is_shift(
    parser: CandleParser,
    index: int,
    trend: str,
    lookback: int = SWEPT_LOOKBACK_DEFAULT,
    window: Optional[int] = None,
    _broken_swing_indices: Optional[set] = None,
) -> Optional[Dict[str, Any]]:
    """
    Kiểm tra nến tại `index` có phá cấu trúc (Shift / MSS) hay không, dựa
    trên swing gần nhất trong `lookback`. Trả về BOS nếu phá cùng hướng
    `trend` hiện tại (tiếp diễn), CHoCH nếu phá ngược hướng (đảo chiều).

    Khác is_swept(): dùng CLOSE (không phải High/Low) để xác định phá —
    cấu trúc cần được "xác nhận" bằng giá đóng cửa, wick chạm qua không đủ
    (xem tests/test_shift.py, case near_miss_wick_only_no_close_break).

    `trend`: "BULL" | "BEAR" — hướng trend hiện tại TRƯỚC khi xét shift,
    do caller truyền vào (hàm này không tự suy ra trend dài hạn).

    `_broken_swing_indices`: giống is_swept(), set các swing_idx ĐÃ bị
    phá trước đó trong cùng 1 lần quét toàn chart — truyền vào để đảm bảo
    swing đã shift rồi không được dùng làm mốc tham chiếu lại. Khi gọi
    đơn lẻ (không quét toàn chart) có thể bỏ qua tham số này.

    Quy tắc suy BOS/CHoCH:
        trend=BULL, phá Swing High (Close > level) -> BOS,   direction=BULL
        trend=BULL, phá Swing Low  (Close < level) -> CHoCH, direction=BEAR
        trend=BEAR, phá Swing Low  (Close < level) -> BOS,   direction=BEAR
        trend=BEAR, phá Swing High (Close > level) -> CHoCH, direction=BULL

    Tie-breaking: chỉ báo cáo swing GẦN NHẤT trong lookback (nhất quán với
    is_swept), dù nến có phá nhiều swing xa hơn cùng lúc.

    Trả về dict: {"type", "direction", "shift_candle_idx", "swing_idx",
    "swing_level", "broken_type"} hoặc None nếu không có shift.
    """
    if trend not in ("BULL", "BEAR"):
        raise ValueError(f"trend phải là 'BULL' hoặc 'BEAR', nhận: {trend!r}")

    broken = _broken_swing_indices or set()
    candle = parser[index]

    if trend == "BULL":
        continuation_swing = _find_active_swing_high(parser, index, lookback, window)
        if continuation_swing and continuation_swing["swing_idx"] not in broken:
            if candle.close > continuation_swing["swing_level"]:
                return {
                    "type"            : "BOS",
                    "direction"       : "BULL",
                    "shift_candle_idx": index,
                    "swing_idx"       : continuation_swing["swing_idx"],
                    "swing_level"     : continuation_swing["swing_level"],
                    "broken_type"     : "HIGH",
                }
        reversal_swing = _find_active_swing_low(parser, index, lookback, window)
        if reversal_swing and reversal_swing["swing_idx"] not in broken:
            if candle.close < reversal_swing["swing_level"]:
                return {
                    "type"            : "CHoCH",
                    "direction"       : "BEAR",
                    "shift_candle_idx": index,
                    "swing_idx"       : reversal_swing["swing_idx"],
                    "swing_level"     : reversal_swing["swing_level"],
                    "broken_type"     : "LOW",
                }

    else:   # trend == "BEAR"
        continuation_swing = _find_active_swing_low(parser, index, lookback, window)
        if continuation_swing and continuation_swing["swing_idx"] not in broken:
            if candle.close < continuation_swing["swing_level"]:
                return {
                    "type"            : "BOS",
                    "direction"       : "BEAR",
                    "shift_candle_idx": index,
                    "swing_idx"       : continuation_swing["swing_idx"],
                    "swing_level"     : continuation_swing["swing_level"],
                    "broken_type"     : "LOW",
                }
        reversal_swing = _find_active_swing_high(parser, index, lookback, window)
        if reversal_swing and reversal_swing["swing_idx"] not in broken:
            if candle.close > reversal_swing["swing_level"]:
                return {
                    "type"            : "CHoCH",
                    "direction"       : "BULL",
                    "shift_candle_idx": index,
                    "swing_idx"       : reversal_swing["swing_idx"],
                    "swing_level"     : reversal_swing["swing_level"],
                    "broken_type"     : "HIGH",
                }

    return None


def scan_all_shift(
    parser: CandleParser,
    initial_trend: str,
    lookback: int = SWEPT_LOOKBACK_DEFAULT,
    window: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Quét toàn chart, xử lý 2 việc mà gọi is_shift() đơn lẻ KHÔNG tự làm được:

        1. Trend EVOLVE theo thời gian: khi gặp CHoCH, trend đảo chiều
           cho các lần quét TIẾP THEO (BOS giữ nguyên trend hiện tại).
        2. Swing đã bị shift rồi KHÔNG được dùng làm mốc tham chiếu lại
           (cộng dồn `broken` set, cùng nguyên tắc với scan_all_swept()).

    `initial_trend`: trend giả định TẠI NẾN ĐẦU chart — đây là input bên
    ngoài (vd suy từ HTF bias, hoặc quy ước đơn giản của caller), hàm này
    KHÔNG tự suy ra được, phải truyền vào tường minh.
    """
    trend: str = initial_trend
    broken: set = set()
    results = []

    for i in range(len(parser)):
        r = is_shift(parser, i, trend=trend, lookback=lookback, window=window, _broken_swing_indices=broken)
        if r:
            broken.add(r["swing_idx"])
            broken.add(r["shift_candle_idx"])   # nến thực hiện shift cũng không dùng làm swing mới
            if r["type"] == "CHoCH":
                trend = r["direction"]          # trend đảo chiều cho các lần quét sau
            results.append(r)

    return results