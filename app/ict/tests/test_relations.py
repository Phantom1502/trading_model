"""
tests/test_relations.py — Lớp 5: build_relations
=====================================================
Rule tie-breaking SAME_CANDLE đã quyết định 1 phần (xem relations.py
docstring): Swept đứng TRƯỚC Shift khi trùng candle_idx. Case có FVG
tham gia vẫn CHƯA quyết định, giữ nguyên nhãn "SAME_CANDLE".
"""

from app.ict.relations import build_relations


def _swept_event(candle_idx, direction="HIGH"):
    return {
        "type"            : f"SWEEP_{direction}",
        "swept_candle_idx": candle_idx,
        "swing_level"     : 500,
    }


def _fvg_event(candle_idx, gap_low, gap_high, fvg_type="BULL"):
    return {
        "type"          : fvg_type,
        "fvg_candle_idx": candle_idx,
        "gap_low"       : gap_low,
        "gap_high"      : gap_high,
        "gap_size_bins" : gap_high - gap_low,
        "fill_pct"      : 0.0,
    }


def _shift_event(candle_idx, shift_type="BOS", direction="BULL"):
    return {
        "type"            : shift_type,
        "direction"       : direction,
        "shift_candle_idx": candle_idx,
        "swing_level"     : 500,
        "broken_type"     : "HIGH",
    }


def test_clear_sequential_order():
    """Event A tại nến 5, event B tại nến 10 -> order == A_BEFORE_B."""
    events = [_swept_event(5), _swept_event(10)]
    relations = build_relations(events)
    assert len(relations) == 1
    assert relations[0]["order"] == "A_BEFORE_B"


def test_clear_same_direction():
    """2 SWEEP_LOW (cùng hướng BULL - lực mua) -> same_direction == True."""
    events = [_swept_event(3, "LOW"), _swept_event(8, "LOW")]
    relations = build_relations(events)
    assert relations[0]["same_direction"] is True


def test_clear_opposite_direction():
    """SWEEP_HIGH và SWEEP_LOW -> same_direction == False."""
    events = [_swept_event(3, "HIGH"), _swept_event(8, "LOW")]
    relations = build_relations(events)
    assert relations[0]["same_direction"] is False


def test_clear_shift_direction_uses_explicit_field():
    """Shift event dùng field "direction" tường minh (không suy từ "type"
    như "BOS"/"CHoCH") — regression test cho bug đã fix qua integration test
    (xem test_relations_integration.py)."""
    events = [_shift_event(3, "BOS", "BULL"), _shift_event(8, "CHoCH", "BEAR")]
    relations = build_relations(events)
    assert relations[0]["same_direction"] is False   # BULL != BEAR, không phải None


def test_clear_overlap_zone():
    """FVG gap [510,525] và FVG gap [515,530] -> chồng lấp [515,525] -> overlap == True."""
    events = [
        _fvg_event(3, gap_low=510, gap_high=525),
        _fvg_event(8, gap_low=515, gap_high=530),
    ]
    relations = build_relations(events)
    assert relations[0]["overlap"] is True


def test_boundary_adjacent_no_overlap():
    """2 vùng giá kề sát nhau nhưng không chồng lấp (hi==lo, strict '>') -> False."""
    events = [
        _fvg_event(3, gap_low=510, gap_high=520),
        _fvg_event(8, gap_low=520, gap_high=530),   # kề sát: lo=max(510,520)=520, hi=min(520,530)=520 -> hi > lo là False
    ]
    relations = build_relations(events)
    assert relations[0]["overlap"] is False


def test_tie_breaking_swept_before_shift_same_candle():
    """Swept và Shift trùng candle_idx -> resolve thành A_BEFORE_B/B_BEFORE_A
    theo rule Swept-trước-Shift, KHÔNG còn là "SAME_CANDLE" (đã quyết định)."""
    events = [_swept_event(5, "HIGH"), _shift_event(5, "BOS", "BULL")]
    relations = build_relations(events)
    assert relations[0]["order"] == "A_BEFORE_B"   # Swept (A) trước Shift (B)


def test_tie_breaking_swept_before_shift_reversed_input_order():
    """Đảo thứ tự input (Shift trước Swept trong list) -> vẫn resolve ĐÚNG
    theo rule domain (Swept trước Shift), không phụ thuộc thứ tự list đầu vào."""
    events = [_shift_event(5, "BOS", "BULL"), _swept_event(5, "HIGH")]
    relations = build_relations(events)
    assert relations[0]["order"] == "B_BEFORE_A"   # Swept (B) trước Shift (A)


def test_tie_breaking_same_candle_index_fvg_still_undecided():
    """2 event cùng candle_idx, 1 trong 2 là FVG -> order == SAME_CANDLE
    (CHƯA quyết định thứ tự cho case có FVG tham gia, chỉ Swept/Shift đã
    được quyết định — xem relations.py docstring)."""
    events = [_swept_event(5, "HIGH"), _fvg_event(5, gap_low=510, gap_high=525)]
    relations = build_relations(events)
    assert relations[0]["order"] == "SAME_CANDLE"


def test_clear_no_events():
    """List rỗng -> list relation rỗng, không crash."""
    assert build_relations([]) == []


def test_clear_single_event():
    """1 event -> không có cặp nào -> list rỗng."""
    assert build_relations([_swept_event(5)]) == []


def test_clear_three_events_count():
    """3 event -> C(3,2) = 3 cặp relation."""
    events = [_swept_event(3), _swept_event(6), _swept_event(9)]
    relations = build_relations(events)
    assert len(relations) == 3