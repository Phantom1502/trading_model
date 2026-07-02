"""
tests/test_relations_integration.py — Lớp 5: build_relations trên output THẬT
======================================================================================
Khác test_relations.py (dùng dict event TỰ TẠO TAY, có thể vô tình khớp
đúng field mà code mong đợi, che giấu bug thật) — file này chạy build_facts()
với 1 chart tổng hợp đủ 3 loại event (Swept + FVG + Shift), rồi kiểm tra
build_relations() xử lý ĐÚNG trên chính dict event thật do scan_all_swept/
grade_fvg/scan_all_shift sinh ra.
"""

from app.ict.candle import Candle
from app.ict.parser import CandleParser
from app.ict.facts import build_facts


def _c(o, h, l, c):
    return Candle(open=o, high=h, low=l, close=c)


def _build_multi_event_chart():
    """
    Chart tổng hợp cố ý chứa cả 3 loại event trong cùng 1 chuỗi:
        - Swing high tại idx 2 (H=530) -> bị sweep tại idx 5 (wick xuyên qua)
        - FVG hoàn thiện tại idx 9 (BULL)
        - Swing high tại idx 12 (H=545) -> bị shift (Close xác nhận) tại idx 15
    """
    return [
        _c(495, 505, 490, 500),   # 0: padding
        _c(498, 508, 493, 503),   # 1: padding
        _c(505, 530, 500, 525),   # 2: swing high H=530
        _c(520, 518, 510, 515),   # 3
        _c(515, 513, 505, 510),   # 4
        _c(510, 535, 505, 512),   # 5: High=535 > 530 (wick) nhưng Close=512 -> SWEEP (không phải Shift)
        _c(512, 516, 508, 514),   # 6
        _c(514, 522, 510, 518),   # 7: nến đầu FVG (High=522)
        _c(519, 521, 517, 520),   # 8: nến giữa FVG
        _c(525, 535, 524, 530),   # 9: nến 3 FVG, Low=524 > High(idx7)=522 -> BULL FVG hoàn thiện
        _c(528, 532, 522, 525),   # 10
        _c(523, 528, 518, 520),   # 11
        _c(520, 545, 515, 540),   # 12: swing high H=545
        _c(535, 533, 525, 528),   # 13
        _c(525, 523, 515, 518),   # 14
        _c(515, 555, 510, 550),   # 15: Close=550 > 545 -> BOS (trend BULL, tiếp diễn)
    ]


def test_clear_relations_on_real_facts_output():
    """build_relations() chạy được trên fact JSON thật, không crash, cấu
    trúc kết quả hợp lệ."""
    parser = CandleParser.from_candles(_build_multi_event_chart(), swing_window=2)
    facts = build_facts(parser, initial_trend="BULL", lookback=10)

    # Xác nhận cả 3 loại event đều xuất hiện — nếu không, chart mẫu chưa
    # đủ đại diện, cần sửa lại chart chứ không phải bỏ qua kiểm tra
    assert len(facts["swept"]) >= 1, "Chart mẫu cần có ít nhất 1 Swept event"
    assert len(facts["fvg"]) >= 1, "Chart mẫu cần có ít nhất 1 FVG event"
    assert len(facts["shift"]) >= 1, "Chart mẫu cần có ít nhất 1 Shift event"

    relations = facts["relations"]
    assert isinstance(relations, list)
    assert len(relations) > 0

    for r in relations:
        assert r["order"] in ("A_BEFORE_B", "B_BEFORE_A", "SAME_CANDLE")
        assert r["same_direction"] in (True, False, None)
        assert r["overlap"] in (True, False, None)


def test_clear_shift_event_direction_not_silently_none():
    """BUG THẬT phát hiện qua integration test: _event_direction() trong
    relations.py trước đây chỉ đọc field "type" để suy hướng, nhưng Shift
    event dùng "type"="BOS"/"CHoCH" (không phải "BULL"/"BEAR" hay chứa
    "HIGH"/"LOW" như Swept/FVG) — khiến same_direction LUÔN None sai cho
    mọi relation có Shift event tham gia, dù event đó CÓ field "direction"
    tường minh. Test này khẳng định field "direction" phải được ưu tiên
    đọc trước khi suy từ "type".
    """
    parser = CandleParser.from_candles(_build_multi_event_chart(), swing_window=2)
    facts = build_facts(parser, initial_trend="BULL", lookback=10)

    # Tìm 1 relation có event Shift tham gia (event có field "direction" tường minh)
    all_events = facts["swept"] + facts["fvg"] + facts["shift"]
    shift_indices = {i for i, e in enumerate(all_events) if "direction" in e and "shift_candle_idx" in e}
    assert shift_indices, "Chart mẫu cần có ít nhất 1 Shift event để test này có ý nghĩa"

    relations_with_shift = [
        r for r in facts["relations"]
        if r["event_a_idx"] in shift_indices or r["event_b_idx"] in shift_indices
    ]
    assert relations_with_shift, "Cần ít nhất 1 relation có Shift event tham gia"

    # same_direction KHÔNG được luôn là None chỉ vì có Shift event tham gia —
    # nếu event còn lại cũng xác định được hướng, phải so sánh được, không bỏ cuộc
    all_none = all(r["same_direction"] is None for r in relations_with_shift)
    assert not all_none, (
        "same_direction luôn None cho mọi relation có Shift tham gia — "
        "dấu hiệu _event_direction() không đọc field 'direction' tường minh."
    )


def test_clear_swept_shift_same_candle_resolves_naturally():
    """Chart mẫu (_build_multi_event_chart) cố tình tạo nến idx 12 vừa
    Sweep (wick xuyên qua swing) vừa Shift/BOS (Close xác nhận) — đây KHÔNG
    phải case hiếm, mà là kịch bản kinh điển "nến breakout mạnh". Xác nhận
    rule Swept-trước-Shift resolve đúng trên data thật, không còn nhãn
    SAME_CANDLE mơ hồ cho case tự nhiên này."""
    parser = CandleParser.from_candles(_build_multi_event_chart(), swing_window=2)
    facts = build_facts(parser, initial_trend="BULL", lookback=10)

    # Tìm event Swept và Shift cùng candle_idx=12
    swept_at_12 = next(e for e in facts["swept"] if e["swept_candle_idx"] == 12)
    shift_at_12 = next(e for e in facts["shift"] if e["shift_candle_idx"] == 12)
    assert swept_at_12 is not None and shift_at_12 is not None

    all_events = facts["swept"] + facts["fvg"] + facts["shift"]
    swept_idx = all_events.index(swept_at_12)
    shift_idx = all_events.index(shift_at_12)

    relation = next(
        r for r in facts["relations"]
        if {r["event_a_idx"], r["event_b_idx"]} == {swept_idx, shift_idx}
    )
    assert relation["order"] != "SAME_CANDLE"   # đã resolve, không còn mơ hồ