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

RULE TIE-BREAKING SAME_CANDLE (đã quyết định 1 phần):
    Swept và Shift trùng đúng 1 nến -> Swept đứng TRƯỚC Shift. Lý do: wick
    quét thanh khoản xảy ra TRONG LÚC nến hình thành, còn Shift xác nhận
    bằng giá ĐÓNG CỬA — về mặt thời gian hình thành nến, sweep "xảy ra
    trước" shift. Hướng Shift đại diện cho "bên thắng cuộc" sau cùng.

    FVG tham gia vào tie cùng nến -> CHƯA quyết định thứ tự, giữ nguyên
    nhãn "SAME_CANDLE" (không suy đoán thêm).
"""

from typing import List, Dict, Any, Optional


# Độ ưu tiên khi tie cùng candle_idx — số nhỏ hơn = xảy ra trước.
# CHỈ Swept và Shift đã được quyết định thứ tự (xem docstring module).
# FVG CỐ TÌNH không có trong dict này -> mọi tie liên quan tới FVG giữ
# nguyên nhãn "SAME_CANDLE", không suy đoán thứ tự.
_SAME_CANDLE_PRIORITY = {"SWEPT": 0, "SHIFT": 1}


def _event_kind(event: Dict[str, Any]) -> str:
    """Suy loại event từ field đặc trưng (swept/fvg/shift_candle_idx)."""
    if "swept_candle_idx" in event:
        return "SWEPT"
    if "shift_candle_idx" in event:
        return "SHIFT"
    if "fvg_candle_idx" in event:
        return "FVG"
    return "UNKNOWN"


def _event_candle_idx(event: Dict[str, Any]) -> int:
    """Lấy chỉ số nến đại diện cho 1 event, tùy loại event có field khác nhau."""
    for key in ("swept_candle_idx", "fvg_candle_idx", "shift_candle_idx", "candle_idx"):
        if key in event:
            return event[key]
    raise KeyError(f"Event không có field chỉ số nến đã biết: {event}")


def _event_direction(event: Dict[str, Any]) -> Optional[str]:
    """
    Suy ra hướng (BULL/BEAR) của 1 event.

    Ưu tiên đọc field "direction" TƯỜNG MINH nếu event có sẵn (Shift event
    từ scan_all_shift() luôn có field này: "direction": "BULL"|"BEAR").
    Chỉ suy từ "type" khi event KHÔNG có field "direction" (Swept dùng
    "SWEEP_HIGH"/"SWEEP_LOW", FVG dùng "BULL"/"BEAR" trực tiếp trong "type").

    BUG ĐÃ FIX (phát hiện qua tests/test_relations_integration.py — chạy
    build_relations() trên event dict THẬT do scan_all_shift() sinh ra,
    không phải dict tự tạo tay): version cũ chỉ đọc "type", khiến Shift
    event (type="BOS"/"CHoCH", không chứa "HIGH"/"LOW" và không phải chính
    xác "BULL"/"BEAR") luôn trả về None — làm same_direction sai thành
    None cho MỌI relation có Shift event tham gia, dù event đó có sẵn
    field "direction" tường minh không cần suy đoán gì cả.
    """
    if "direction" in event:
        return event["direction"]

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

    Case cùng candle_idx (idx_a == idx_b): ĐÃ QUYẾT ĐỊNH 1 PHẦN — nếu 1
    trong 2 event là Swept và bên kia là Shift, resolve thành A_BEFORE_B
    hoặc B_BEFORE_A theo rule Swept-trước-Shift (xem docstring module).
    Mọi tie khác (có FVG tham gia, hoặc cùng loại) giữ nguyên nhãn
    "SAME_CANDLE" — CHƯA quyết định thứ tự, không suy đoán thêm.
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
                # Cùng candle_idx — thử resolve theo rule Swept-trước-Shift
                kind_a, kind_b = _event_kind(a), _event_kind(b)
                prio_a = _SAME_CANDLE_PRIORITY.get(kind_a)
                prio_b = _SAME_CANDLE_PRIORITY.get(kind_b)

                if prio_a is not None and prio_b is not None and prio_a != prio_b:
                    order = "A_BEFORE_B" if prio_a < prio_b else "B_BEFORE_A"
                else:
                    order = "SAME_CANDLE"   # FVG tham gia hoặc cùng loại -> chưa quyết định

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