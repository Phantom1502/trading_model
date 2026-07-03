"""
tests/test_render.py — render.py: template engine, KHÔNG dùng GPT
========================================================================
Trọng tâm test: số liệu trong eval/explanation PHẢI khớp CHÍNH XÁC với
fact dict gốc (đảm bảo string interpolation không sai lệch, không hallucinate
— vì đây chính là lý do bỏ GPT). Dùng random.Random(seed) để test
deterministic, không phụ thuộc random module global.
"""

import random
import re

from app.ict.candle import Candle
from app.ict.parser import CandleParser
from app.ict.facts import build_facts
from app.ict.render import (
    render_swept_sample, render_fvg_sample, render_shift_sample,
    render_synthesis_sample, render_all_samples,
)


def _c(o, h, l, c):
    return Candle(open=o, high=h, low=l, close=c)


def _build_facts_and_raw(candles, initial_trend="BULL"):
    parser = CandleParser.from_candles(candles, swing_window=2)
    facts = build_facts(parser, initial_trend=initial_trend, lookback=10)
    raw_chart_text = parser.raw_text
    return facts, raw_chart_text


# ── Chart tối giản, ĐÚNG 1 event mỗi loại (để test interpolation số liệu) ──

def _single_swept_chart():
    return [
        _c(495, 505, 490, 500), _c(498, 508, 493, 503), _c(505, 530, 500, 525),   # 2: swing high
        _c(520, 518, 510, 515), _c(515, 513, 505, 510), _c(510, 535, 505, 520),   # 5: sweep
    ]


def _single_fvg_chart():
    return [
        _c(500, 510, 495, 505),
        _c(515, 525, 512, 520),
        _c(530, 540, 525, 535),   # idx2: FVG duy nhất (chỉ 3 nến, chỉ 1 vị trí check được)
    ]


def _single_shift_chart():
    return [
        _c(495, 505, 490, 500), _c(498, 508, 493, 503), _c(505, 530, 500, 525),   # 2: swing high
        _c(520, 518, 510, 515), _c(515, 513, 505, 510), _c(510, 540, 505, 535),   # 5: BOS (Close=535>530)
    ]


def _multi_event_chart():
    """Chart giống test_relations_integration.py — đủ 3 loại event (2 Swept, 4 FVG, 1 Shift)."""
    return [
        _c(495, 505, 490, 500), _c(498, 508, 493, 503), _c(505, 530, 500, 525),
        _c(520, 518, 510, 515), _c(515, 513, 505, 510), _c(510, 535, 505, 512),
        _c(512, 516, 508, 514), _c(514, 522, 510, 518), _c(519, 521, 517, 520),
        _c(525, 535, 524, 530), _c(528, 532, 522, 525), _c(523, 528, 518, 520),
        _c(520, 545, 515, 540), _c(535, 533, 525, 528), _c(525, 523, 515, 518),
        _c(515, 555, 510, 550),
    ]


def _two_fvg_chart():
    """Đúng 2 FVG tách biệt, không chồng lấp lẫn nhau (đã verify thủ công)."""
    return [
        _c(500, 510, 495, 505),
        _c(515, 525, 512, 520),
        _c(530, 540, 525, 535),   # FVG idx2
        _c(520, 524, 510, 515),   # pullback tránh idx3
        _c(515, 530, 505, 520),   # tránh idx4
        _c(540, 560, 545, 555),   # FVG idx5
    ]


# ══════════════════════════════════════════════════════════════════════
# Số liệu khớp CHÍNH XÁC fact dict
# ══════════════════════════════════════════════════════════════════════

def test_clear_swept_sample_numbers_match_fact_exactly():
    """Số liệu trong eval PHẢI khớp CHÍNH XÁC fact dict — kiểm tra từng
    field bằng cách parse ngược lại eval block."""
    facts, raw = _build_facts_and_raw(_single_swept_chart())
    assert len(facts["swept"]) == 1
    rng = random.Random(42)
    sample = render_swept_sample(facts, raw, rng=rng)
    assert sample is not None

    e = facts["swept"][0]
    fields = dict(re.findall(r"(\w+)=([\w.\-]+)", sample["eval"]))
    assert fields["TYPE"] == e["type"]
    assert int(fields["CANDLE"]) == e["swept_candle_idx"] + 1   # 1-based
    assert int(fields["SWING_CANDLE"]) == e["swing_idx"] + 1
    assert int(fields["SWING_LEVEL"]) == e["swing_level"]
    assert int(fields["DEPTH"]) == e["depth"]


def test_clear_fvg_sample_numbers_match_fact_exactly():
    facts, raw = _build_facts_and_raw(_single_fvg_chart())
    assert len(facts["fvg"]) == 1
    rng = random.Random(42)
    sample = render_fvg_sample(facts, raw, rng=rng)
    assert sample is not None

    e = facts["fvg"][0]
    fields = dict(re.findall(r"(\w+)=([\w.\-]+)", sample["eval"]))
    assert fields["TYPE"] == e["type"]
    assert int(fields["CANDLE"]) == e["fvg_candle_idx"] + 1
    assert int(fields["GAP_LOW"]) == e["gap_low"]
    assert int(fields["GAP_HIGH"]) == e["gap_high"]
    assert int(fields["GAP_SIZE"]) == e["gap_size_bins"]
    assert float(fields["FILL_PCT"]) == e["fill_pct"]


def test_clear_shift_sample_numbers_match_fact_exactly():
    facts, raw = _build_facts_and_raw(_single_shift_chart())
    assert len(facts["shift"]) == 1
    rng = random.Random(42)
    sample = render_shift_sample(facts, raw, rng=rng)
    assert sample is not None

    e = facts["shift"][0]
    fields = dict(re.findall(r"(\w+)=([\w.\-]+)", sample["eval"]))
    assert fields["TYPE"] == e["type"]
    assert fields["DIRECTION"] == e["direction"]
    assert int(fields["CANDLE"]) == e["shift_candle_idx"] + 1
    assert int(fields["SWING_CANDLE"]) == e["swing_idx"] + 1
    assert int(fields["SWING_LEVEL"]) == e["swing_level"]
    assert fields["BROKEN"] == e["broken_type"]


# ══════════════════════════════════════════════════════════════════════
# Format cơ bản
# ══════════════════════════════════════════════════════════════════════

def test_clear_chart_block_unchanged():
    """Phần [1. CHART] PHẢI giữ nguyên y hệt raw_chart_text đầu vào, không
    bị chỉnh sửa gì (nguyên tắc spec mục 5: không inline tag vào giữa chart)."""
    facts, raw = _build_facts_and_raw(_single_swept_chart())
    rng = random.Random(1)
    sample = render_swept_sample(facts, raw, rng=rng)
    assert sample["chart"] == raw
    assert sample["text"].startswith(raw)


def test_clear_single_event_no_numbering():
    """N==1 event -> field KHÔNG đánh số EVENT1_ (theo spec mục 4: chỉ đánh
    số khi nhiều sự kiện)."""
    facts, raw = _build_facts_and_raw(_single_shift_chart())
    rng = random.Random(7)
    sample = render_shift_sample(facts, raw, rng=rng)
    assert len(facts["shift"]) == 1
    assert "EVENT1_" not in sample["eval"]
    assert "TYPE=" in sample["eval"]


def test_boundary_multiple_events_use_numbering():
    """N>1 event cùng loại -> field PHẢI đánh số EVENT1_/EVENT2_."""
    facts, raw = _build_facts_and_raw(_two_fvg_chart())
    assert len(facts["fvg"]) >= 2

    rng = random.Random(3)
    sample = render_fvg_sample(facts, raw, rng=rng)
    assert "EVENT1_TYPE=" in sample["eval"]
    assert "EVENT2_TYPE=" in sample["eval"]


def test_clear_short_template_used_for_3plus_events():
    """>=3 event trong 1 mẫu -> dùng template "short" (câu ngắn gọn hơn),
    theo nguyên tắc co giãn độ dài spec mục 5. Dùng chart tổng hợp có sẵn
    4 FVG (đã verify tự nhiên >=3) thay vì cố ép đúng số lượng."""
    facts, raw = _build_facts_and_raw(_multi_event_chart())
    assert len(facts["fvg"]) >= 3

    rng = random.Random(5)
    sample = render_fvg_sample(facts, raw, rng=rng)
    assert sample["event_count"] == len(facts["fvg"])
    assert sample["event_count"] >= 3


# ══════════════════════════════════════════════════════════════════════
# Mẫu Tổng hợp
# ══════════════════════════════════════════════════════════════════════

def test_clear_synthesis_includes_sequence_when_multiple_events():
    """Mẫu Tổng hợp có >=2 event -> eval PHẢI có field SEQUENCE."""
    facts, raw = _build_facts_and_raw(_multi_event_chart())
    rng = random.Random(9)
    sample = render_synthesis_sample(facts, raw, rng=rng)
    assert sample is not None
    total_events = len(facts["swept"]) + len(facts["fvg"]) + len(facts["shift"])
    assert total_events >= 2
    assert "SEQUENCE=" in sample["eval"]


def test_clear_synthesis_event_count_matches_all_types():
    """event_count của mẫu Tổng hợp phải bằng tổng số event cả 3 loại
    (chart mẫu có <= FVG_TOP_K nên KHÔNG bị lọc — test riêng top-K bên dưới)."""
    facts, raw = _build_facts_and_raw(_multi_event_chart())
    rng = random.Random(11)
    sample = render_synthesis_sample(facts, raw, rng=rng)
    expected = len(facts["swept"]) + len(facts["fvg"]) + len(facts["shift"])
    assert sample["event_count"] == expected
    assert sample["total_event_count"] == expected   # không lọc -> 2 field bằng nhau


def _many_fvg_chart():
    """7 FVG rõ ràng (>FVG_TOP_K=4), tách biệt hoàn toàn theo nhóm base giá cách xa."""
    candles = []
    for k in range(3):
        base = 1000 + k * 300
        candles += [
            _c(base, base + 10, base - 5, base + 5),
            _c(base + 15, base + 25, base + 12, base + 20),
            _c(base + 30, base + 40, base + 25, base + 35),
            _c(base + 20, base + 24, base + 10, base + 15),
            _c(base + 15, base + 30, base + 5, base + 20),
        ]
    return candles


def test_boundary_fvg_top_k_reduces_event_count():
    """Chart có 7 FVG (>FVG_TOP_K=4) -> render_fvg_sample CHỈ hiển thị 4,
    nhưng total_event_count PHẢI vẫn phản ánh đúng 7 (minh bạch, không giấu
    thông tin đã lọc)."""
    from app.ict.render import FVG_TOP_K

    facts, raw = _build_facts_and_raw(_many_fvg_chart())
    assert len(facts["fvg"]) == 7
    assert len(facts["fvg"]) > FVG_TOP_K

    rng = random.Random(1)
    sample = render_fvg_sample(facts, raw, rng=rng)
    assert sample["event_count"] == FVG_TOP_K
    assert sample["total_event_count"] == 7


def test_clear_fvg_top_k_keeps_chronological_order():
    """Kết quả lọc top-K vẫn giữ đúng thứ tự thời gian khi hiển thị (không
    xáo trộn theo rank), chỉ QUYẾT ĐỊNH giữ/bỏ dựa trên rank."""
    facts, raw = _build_facts_and_raw(_many_fvg_chart())
    rng = random.Random(1)
    sample = render_fvg_sample(facts, raw, rng=rng)

    import re
    candles_mentioned = [int(x) for x in re.findall(r"CANDLE=(\d+)", sample["eval"])]
    assert candles_mentioned == sorted(candles_mentioned)   # tăng dần -> đúng thứ tự thời gian


def test_clear_synthesis_top_k_and_relations_remap_correctly():
    """Mẫu Tổng hợp với chart nhiều FVG -> event_count < total_event_count,
    và SEQUENCE vẫn hợp lệ (không lỗi index sau khi remap relations)."""
    from app.ict.render import FVG_TOP_K

    facts, raw = _build_facts_and_raw(_many_fvg_chart())
    rng = random.Random(1)
    sample = render_synthesis_sample(facts, raw, rng=rng)

    assert sample is not None
    expected_kept = len(facts["swept"]) + FVG_TOP_K + len(facts["shift"])
    expected_total = len(facts["swept"]) + len(facts["fvg"]) + len(facts["shift"])
    assert sample["event_count"] == expected_kept
    assert sample["total_event_count"] == expected_total
    assert sample["event_count"] < sample["total_event_count"]

    # SEQUENCE vẫn phải hợp lệ: đúng N-1 cặp theo N đã lọc (không phải N gốc)
    import re
    seq_match = re.search(r"SEQUENCE=([\d<~,]+)(?=</eval>)", sample["eval"])
    assert seq_match is not None
    pairs = len(seq_match.group(1).split(","))
    assert pairs == expected_kept - 1   # N-1 theo N ĐÃ LỌC, không phải N gốc

    # validate_no_leakage vẫn phải pass sau khi lọc (không có số liệu rác sót lại)
    from app.ict.validate import validate_no_leakage
    assert validate_no_leakage(sample) is True


def test_clear_sequence_is_linear_not_quadratic():
    """SEQUENCE PHẢI có đúng N-1 cặp (liền kề theo thời gian), KHÔNG PHẢI
    C(N,2) cặp tổ hợp đầy đủ — đây là fix quan trọng giảm token cho chart
    nhiều event (vd cửa sổ 29 nến từng phát sinh mẫu >1000 token)."""
    facts, raw = _build_facts_and_raw(_multi_event_chart())
    rng = random.Random(11)
    sample = render_synthesis_sample(facts, raw, rng=rng)

    n = sample["event_count"]
    assert n >= 4   # chart mẫu có 7 event — đủ để phân biệt O(n) vs O(n²)

    seq_match = re.search(r"SEQUENCE=([\d<~,]+)(?=</eval>)", sample["eval"])
    assert seq_match is not None
    pairs = len(seq_match.group(1).split(","))

    assert pairs == n - 1   # O(n): đúng N-1 cặp liền kề
    assert pairs != n * (n - 1) // 2   # KHÔNG PHẢI C(n,2) — xác nhận đã fix, không phải trùng hợp


def test_clear_sequence_pairs_are_chronologically_adjacent():
    """Mỗi cặp trong SEQUENCE phải là 2 event LIỀN KỀ theo candle_idx thời
    gian thực tế (không phải liền kề theo thứ tự list gốc swept+fvg+shift)."""
    facts, raw = _build_facts_and_raw(_multi_event_chart())
    rng = random.Random(11)
    sample = render_synthesis_sample(facts, raw, rng=rng)

    events = facts["swept"] + facts["fvg"] + facts["shift"]
    candle_indices = []
    for e in events:
        for key in ("swept_candle_idx", "fvg_candle_idx", "shift_candle_idx"):
            if key in e:
                candle_indices.append(e[key])
                break
    time_sorted_order = sorted(range(len(events)), key=lambda i: candle_indices[i])
    expected_adjacent_pairs = {
        frozenset((time_sorted_order[k] + 1, time_sorted_order[k + 1] + 1))
        for k in range(len(time_sorted_order) - 1)
    }

    seq_match = re.search(r"SEQUENCE=([\d<~,]+)(?=</eval>)", sample["eval"])
    actual_pairs = set()
    for token in seq_match.group(1).split(","):
        a, sep, b = re.split(r"([<~])", token)
        actual_pairs.add(frozenset((int(a), int(b))))

    assert actual_pairs == expected_adjacent_pairs


# ══════════════════════════════════════════════════════════════════════
# Case biên
# ══════════════════════════════════════════════════════════════════════

def test_near_miss_no_events_returns_none():
    """Chart hoàn toàn không có event nào -> render_*_sample trả về None
    (v1: SKIP, không sinh mẫu "không tìm thấy" — xem docstring render.py)."""
    candles = [
        _c(500, 502, 498, 501), _c(501, 503, 499, 502), _c(502, 504, 500, 503),
        _c(503, 505, 501, 504), _c(504, 506, 502, 505),
    ]
    facts, raw = _build_facts_and_raw(candles)

    assert render_swept_sample(facts, raw) is None
    assert render_fvg_sample(facts, raw) is None
    assert render_shift_sample(facts, raw) is None
    assert render_synthesis_sample(facts, raw) is None


def test_clear_render_all_samples_returns_all_4_when_all_types_present():
    """Chart đủ 3 loại event -> render_all_samples trả về đủ 4 mẫu
    (Swept, FVG, Shift, Tổng hợp)."""
    facts, raw = _build_facts_and_raw(_multi_event_chart())
    rng = random.Random(13)
    samples = render_all_samples(facts, raw, rng=rng)
    assert len(samples) == 4


def test_clear_no_gpt_dependency():
    """Xác nhận render.py không import bất kỳ thư viện gọi API/network nào
    — đảm bảo tính deterministic tuyệt đối, không có side-effect ngoài ý muốn."""
    import app.ict.render as render_module
    import inspect

    source = inspect.getsource(render_module)
    forbidden = ["openai", "requests", "httpx", "urllib", "anthropic"]
    for lib in forbidden:
        assert lib not in source.lower(), f"render.py không được import '{lib}' — phải 100% deterministic"


def test_clear_deterministic_with_same_seed():
    """Cùng seed -> kết quả render giống hệt nhau (reproducibility cho test/debug)."""
    facts, raw = _build_facts_and_raw(_multi_event_chart())
    sample1 = render_synthesis_sample(facts, raw, rng=random.Random(99))
    sample2 = render_synthesis_sample(facts, raw, rng=random.Random(99))
    assert sample1["text"] == sample2["text"]