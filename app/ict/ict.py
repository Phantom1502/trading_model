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
    lookback: int = 10,
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


def scan_all_swept(parser: CandleParser, lookback: int = 10, window: Optional[int] = None) -> List[dict]:
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
            results.append(r)
    return results


# ══════════════════════════════════════════════════════════════════════
# FVG graded — CHƯA verify bằng golden test, khung tham số
# ══════════════════════════════════════════════════════════════════════

def grade_fvg(parser: CandleParser, index: int, upto_index: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """
    Mở rộng is_fvg() (structure.py) thành graded: size gap (bin) + % đã lấp
    tính tại thời điểm `upto_index` (mặc định = cuối parser, tức "hiện tại").

    fill_pct phản ánh trạng thái SAU CÙNG tại upto_index, KHÔNG PHẢI mức
    lấp sâu nhất từng đạt được trong quá khứ — quyết định đã chốt theo
    spec mục 9, case near_miss_fill_then_extend. Nghĩa là nếu giá đã lấp
    50% rồi đảo chiều mở rộng gap trở lại, fill_pct phải phản ánh đúng
    mức tại thời điểm hỏi, không giữ lại "lịch sử lấp sâu nhất".

    CẢNH BÁO: hàm này CHƯA có golden test xác nhận (tests/test_fvg_graded.py
    chưa viết) — coi là bản nháp tham số, không dùng để gen data chính
    thức cho tới khi qua Giai đoạn 3 trong lộ trình.
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
    filled = 0
    for i in range(index + 1, upto + 1):
        c = parser[i]
        # phần overlap giữa range nến i và vùng gap
        overlap_low  = max(c.low,  gap_low)
        overlap_high = min(c.high, gap_high)
        if overlap_high > overlap_low:
            filled = max(filled, overlap_high - overlap_low)

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

def is_shift(parser: CandleParser, index: int, trend: str, lookback: int = 10) -> Optional[Dict[str, Any]]:
    """
    STUB — chưa triển khai logic thật.

    Theo lộ trình (spec mục 10, Giai đoạn 3): viết tests/test_shift.py
    TRƯỚC để định nghĩa rõ behavior mong muốn (đặc biệt phân biệt BOS
    tiếp diễn trend vs CHoCH đảo chiều trend, và case near_miss_wick_only
    — wick chạm/vượt swing nhưng giá ĐÓNG CỬA không vượt thì KHÔNG tính
    là shift), rồi mới cài logic vào đây.

    `trend`: hướng trend hiện tại trước khi xét shift ("BULL"|"BEAR"),
    cần truyền vào từ caller vì hàm này không tự suy ra trend dài hạn.
    """
    raise NotImplementedError(
        "is_shift() chưa triển khai — viết tests/test_shift.py trước theo "
        "nguyên tắc test-first đã chốt trong spec (Giai đoạn 3)."
    )