"""
relations.py — Lớp 5: build_relations (Tầng 2 trong spec)
================================================================
Nhận list các "event" đã detect (output của is_swept/grade_fvg/is_shift...,
mỗi event là 1 dict có ít nhất field "type" và 1 chỉ số nến liên quan),
trả về quan hệ giữa từng cặp event: thứ tự thời gian, cùng/ngược hướng,
có chồng lấp vùng giá không.

Đây vẫn là FACT tính được 100% từ dữ liệu đã detect (không phải phán
đoán) — verify bằng golden test giống Tầng 1, không phải qua outcome
thật như Tầng 3 (xem spec mục 2).

CHƯA có golden test (tests/test_relations.py chưa viết) — đặc biệt case
tie_breaking_same_candle_index (2 event cùng index nến, ai trước ai sau)
cần CHỐT LOGIC trước khi viết test, hàm dưới đây để placeholder rõ ràng
thay vì đoán ngầm.
"""

from typing import List, Dict, Any, Optional


def _event_candle_idx(event: Dict[str, Any]) -> int:
    """Lấy chỉ số nến đại diện cho 1 event, tùy loại event có field khác nhau."""
    for key in ("swept_candle_idx", "fvg_candle_idx", "shift_candle_idx", "candle_idx"):
        if key in event:
            return event[key]
    raise KeyError(f"Event không có field chỉ số nến đã biết: {event}")


def _event_direction(event: Dict[str, Any]) -> Optional[str]:
    """Suy ra hướng (BULL/BEAR) từ type của event, None nếu không xác định được."""
    t = event.get("type", "")
    if "HIGH" in t or t == "BEAR":
        return "BEAR"   # sweep high / FVG bear thường đi cùng áp lực giảm
    if "LOW" in t or t == "BULL":
        return "BULL"
    return None


def _event_price_range(event: Dict[str, Any]) -> Optional[tuple]:
    """Vùng giá (bin) liên quan tới event, None nếu event không có vùng giá rõ (vd Shift đơn thuần)."""
    if "gap_low" in event and "gap_high" in event:
        return (event["gap_low"], event["gap_high"])
    if "swing_level" in event:
        return (event["swing_level"], event["swing_level"])
    return None


def build_relations(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Trả về list quan hệ giữa MỌI CẶP event (không trùng lặp, i < j theo
    thứ tự trong list đầu vào — KHÔNG phải thứ tự thời gian, caller cần
    tự sort `events` theo candle_idx trước nếu muốn relations theo đúng
    thứ tự thời gian).

    Mỗi quan hệ: {
        "event_a_idx", "event_b_idx"   — index trong list events đầu vào
        "order"                          — "A_BEFORE_B" | "B_BEFORE_A" | "SAME_CANDLE"
        "same_direction"                  — bool | None (None nếu 1 trong 2 event không xác định hướng)
        "overlap"                          — bool | None (None nếu 1 trong 2 event không có vùng giá)
    }

    LƯU Ý: case "SAME_CANDLE" (2 event cùng candle_idx) hiện chỉ được GÁN
    NHÃN, chưa có rule ưu tiên ai "trước" ai "sau" — đây là tie_breaking
    case còn để ngỏ trong spec, PHẢI chốt logic trước khi dùng kết quả
    "SAME_CANDLE" này cho mục đích sinh data thật.
    """
    relations = []
    n = len(events)

    for i in range(n):
        for j in range(i + 1, n):
            a, b = events[i], events[j]
            idx_a, idx_b = _event_candle_idx(a), _event_candle_idx(b)

            if idx_a < idx_b:
                order = "A_BEFORE_B"
            elif idx_a > idx_b:
                order = "B_BEFORE_A"
            else:
                order = "SAME_CANDLE"

            dir_a, dir_b = _event_direction(a), _event_direction(b)
            same_direction = (dir_a == dir_b) if (dir_a and dir_b) else None

            range_a, range_b = _event_price_range(a), _event_price_range(b)
            overlap = None
            if range_a and range_b:
                lo = max(range_a[0], range_b[0])
                hi = min(range_a[1], range_b[1])
                overlap = hi > lo   # strict — kề sát nhau (hi == lo) KHÔNG tính overlap

            relations.append({
                "event_a_idx"   : i,
                "event_b_idx"   : j,
                "order"         : order,
                "same_direction": same_direction,
                "overlap"       : overlap,
            })

    return relations