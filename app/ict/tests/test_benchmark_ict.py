"""
tests/test_benchmark_ict.py — Verify logic sinh BenchItem cho ICT
========================================================================
KHÔNG test việc chấm điểm thật (avg_logprob_per_token cần model thật,
không có trong sandbox) — CHỈ verify:
    1. positive luôn khớp CHÍNH XÁC fact thật từ app.ict detector.
    2. negative khác positive ĐÚNG Ở ĐÚNG 1 CHỖ đã perturb, các field
       khác giữ nguyên (không vô tình làm sai lệch thêm chỗ khác).
    3. prompt KHÔNG rò rỉ giá trị negative (negative chỉ nằm trong
       completion, không lọt vào phần prompt).
    4. build_ict_bench_items() không crash trên chart không có event nào.
"""

import re

from app.ict.candle import Candle
from app.memlm.benchmark_ict import (
    build_ict_bench_items, _build_swept_items, _build_fvg_items,
    _build_shift_items, _replace_first_field, _extract_field,
)
from app.ict.parser import CandleParser
from app.ict.facts import build_facts
from app.ict.render import render_swept_sample, render_fvg_sample, render_shift_sample

import random


def _c(o, h, l, c):
    return Candle(open=o, high=h, low=l, close=c)


def _single_swept_chart():
    return [
        _c(495, 505, 490, 500), _c(498, 508, 493, 503), _c(505, 530, 500, 525),
        _c(520, 518, 510, 515), _c(515, 513, 505, 510), _c(510, 535, 505, 520),
    ]


def _single_fvg_chart():
    return [
        _c(500, 510, 495, 505),
        _c(515, 525, 512, 520),
        _c(530, 540, 525, 535),
    ]


def _single_shift_chart():
    return [
        _c(495, 505, 490, 500), _c(498, 508, 493, 503), _c(505, 530, 500, 525),
        _c(520, 518, 510, 515), _c(515, 513, 505, 510), _c(510, 540, 505, 535),
    ]


def _no_event_chart():
    return [
        _c(500, 502, 498, 501), _c(501, 503, 499, 502), _c(502, 504, 500, 503),
        _c(503, 505, 501, 504), _c(504, 506, 502, 505),
    ]


# ══════════════════════════════════════════════════════════════════════
# Helper string-level (_replace_first_field / _extract_field)
# ══════════════════════════════════════════════════════════════════════

def test_clear_extract_field_basic():
    assert _extract_field("T=SWEEP_HIGH C=6 SL=530", "T") == "SWEEP_HIGH"
    assert _extract_field("T=SWEEP_HIGH C=6 SL=530", "C") == "6"


def test_clear_extract_field_with_event_prefix():
    assert _extract_field("E1_T=BULL E1_C=4 E2_T=BEAR E2_C=9", "T") == "BULL"   # đầu tiên


def test_clear_replace_first_field_no_prefix():
    result = _replace_first_field("T=SWEEP_HIGH C=6 SL=530", "C", 99)
    assert result == "T=SWEEP_HIGH C=99 SL=530"


def test_clear_replace_first_field_with_prefix():
    result = _replace_first_field("E1_T=BULL E1_C=4", "T", "BEAR")
    assert result == "E1_T=BEAR E1_C=4"


def test_near_miss_replace_field_not_found_returns_none():
    assert _replace_first_field("T=SWEEP_HIGH", "NOTEXIST", 1) is None


# ══════════════════════════════════════════════════════════════════════
# Positive PHẢI khớp chính xác fact thật
# ══════════════════════════════════════════════════════════════════════

def test_clear_swept_positive_matches_real_fact():
    parser = CandleParser.from_candles(_single_swept_chart(), swing_window=2)
    facts = build_facts(parser, initial_trend="BULL", lookback=10)
    sample = render_swept_sample(facts, parser.raw_text)
    rng = random.Random(1)

    items = _build_swept_items(sample, rng)
    assert len(items) == 1

    e = facts["swept"][0]
    positive = items[0].positive[0]
    assert _extract_field(positive, "T") == e["type"]
    assert int(_extract_field(positive, "C")) == e["swept_candle_idx"] + 1
    assert int(_extract_field(positive, "SL")) == e["swing_level"]


def test_clear_fvg_positive_matches_real_fact():
    parser = CandleParser.from_candles(_single_fvg_chart(), swing_window=2)
    facts = build_facts(parser, initial_trend="BULL", lookback=10)
    sample = render_fvg_sample(facts, parser.raw_text)
    rng = random.Random(1)

    items = _build_fvg_items(sample, rng)
    assert len(items) == 1

    e = facts["fvg"][0]
    positive = items[0].positive[0]
    assert _extract_field(positive, "T") == e["type"]
    assert int(_extract_field(positive, "C")) == e["fvg_candle_idx"] + 1
    assert int(_extract_field(positive, "GS")) == e["gap_size_bins"]


def test_clear_shift_positive_matches_real_fact():
    parser = CandleParser.from_candles(_single_shift_chart(), swing_window=2)
    facts = build_facts(parser, initial_trend="BULL", lookback=10)
    sample = render_shift_sample(facts, parser.raw_text)
    rng = random.Random(1)

    items = _build_shift_items(sample, rng)
    assert len(items) == 1

    e = facts["shift"][0]
    positive = items[0].positive[0]
    assert _extract_field(positive, "T") == e["type"]
    assert _extract_field(positive, "DIR") == e["direction"]
    assert int(_extract_field(positive, "C")) == e["shift_candle_idx"] + 1


# ══════════════════════════════════════════════════════════════════════
# Negative khác positive ĐÚNG Ở ĐÚNG 1 CHỖ — không sai lệch thêm chỗ khác
# ══════════════════════════════════════════════════════════════════════

def test_clear_swept_negatives_differ_in_exactly_one_field():
    parser = CandleParser.from_candles(_single_swept_chart(), swing_window=2)
    facts = build_facts(parser, initial_trend="BULL", lookback=10)
    sample = render_swept_sample(facts, parser.raw_text)
    rng = random.Random(1)

    items = _build_swept_items(sample, rng)
    positive = items[0].positive[0]

    pos_fields = dict(re.findall(r"(\w+)=([\w.\-]+)", positive))

    for neg in items[0].negative:
        neg_fields = dict(re.findall(r"(\w+)=([\w.\-]+)", neg))
        assert set(pos_fields.keys()) == set(neg_fields.keys())   # không thêm/bớt field
        diff_keys = [k for k in pos_fields if pos_fields[k] != neg_fields[k]]
        assert len(diff_keys) == 1, f"Negative lệch nhiều hơn 1 field: {diff_keys}"


def test_clear_fvg_negatives_differ_in_exactly_one_field():
    parser = CandleParser.from_candles(_single_fvg_chart(), swing_window=2)
    facts = build_facts(parser, initial_trend="BULL", lookback=10)
    sample = render_fvg_sample(facts, parser.raw_text)
    rng = random.Random(1)

    items = _build_fvg_items(sample, rng)
    positive = items[0].positive[0]
    pos_fields = dict(re.findall(r"(\w+)=([\w.\-]+)", positive))

    for neg in items[0].negative:
        neg_fields = dict(re.findall(r"(\w+)=([\w.\-]+)", neg))
        diff_keys = [k for k in pos_fields if pos_fields[k] != neg_fields[k]]
        assert len(diff_keys) == 1, f"Negative lệch nhiều hơn 1 field: {diff_keys}"


def test_clear_shift_type_direction_negative_is_valid_pair():
    """Perturb loại+hướng của Shift phải đảo CẢ 2 cùng lúc (BOS+BULL <->
    CHoCH+BEAR là cặp hợp lệ đối xứng) — không được chỉ đảo 1 trong 2 (sẽ
    tạo ra tổ hợp T/DIR không tồn tại thật, không phải negative "khó" mà
    là nhiễu vô nghĩa)."""
    parser = CandleParser.from_candles(_single_shift_chart(), swing_window=2)
    facts = build_facts(parser, initial_trend="BULL", lookback=10)
    sample = render_shift_sample(facts, parser.raw_text)
    rng = random.Random(1)

    items = _build_shift_items(sample, rng)
    positive = items[0].positive[0]
    pos_t, pos_dir = _extract_field(positive, "T"), _extract_field(positive, "DIR")

    # Tìm negative có thay đổi T (đảo type+direction cùng lúc)
    type_dir_negatives = [
        n for n in items[0].negative
        if _extract_field(n, "T") != pos_t
    ]
    assert len(type_dir_negatives) == 1
    neg = type_dir_negatives[0]
    # Cả T và DIR đều phải đổi (không chỉ 1 trong 2)
    assert _extract_field(neg, "T") != pos_t
    assert _extract_field(neg, "DIR") != pos_dir


# ══════════════════════════════════════════════════════════════════════
# Prompt KHÔNG rò rỉ giá trị negative
# ══════════════════════════════════════════════════════════════════════

def test_clear_prompt_does_not_leak_negative_values():
    """Prompt (chart+request+explanation) không được vô tình chứa giá trị
    SAI (negative) — vì negative chỉ nên xuất hiện trong completion, không
    phải trong ngữ cảnh model đã thấy trước đó.

    Loại trừ trùng hợp ngẫu nhiên: số nến bị perturb có thể vô tình trùng
    với 1 số nến HỢP LỆ khác đã xuất hiện đúng trong Lý giải (vd trùng với
    swing_idx+1, vốn được nhắc tới hợp lệ) — đây không phải leak, chỉ là
    trùng số ngẫu nhiên do chart nhỏ trong test."""
    parser = CandleParser.from_candles(_single_swept_chart(), swing_window=2)
    facts = build_facts(parser, initial_trend="BULL", lookback=10)
    sample = render_swept_sample(facts, parser.raw_text)
    rng = random.Random(1)

    items = _build_swept_items(sample, rng)
    prompt = items[0].prompt

    # Các số nến HỢP LỆ được phép xuất hiện trong prompt (sweep + swing)
    e = facts["swept"][0]
    legitimate_candles = {e["swept_candle_idx"] + 1, e["swing_idx"] + 1}

    for neg in items[0].negative:
        neg_c = _extract_field(neg, "C")
        pos_c = _extract_field(items[0].positive[0], "C")
        if neg_c != pos_c and int(neg_c) not in legitimate_candles:
            assert f"nến thứ {neg_c}" not in prompt.lower()


def test_clear_prompt_ends_with_eval_tag():
    """Prompt phải dừng ngay sau '<eval>' — completion bắt đầu ngay sau đó,
    không có khoảng trắng/newline thừa gây lệch tokenize."""
    parser = CandleParser.from_candles(_single_swept_chart(), swing_window=2)
    facts = build_facts(parser, initial_trend="BULL", lookback=10)
    sample = render_swept_sample(facts, parser.raw_text)
    rng = random.Random(1)

    items = _build_swept_items(sample, rng)
    assert items[0].prompt.endswith("<eval>")


# ══════════════════════════════════════════════════════════════════════
# build_ict_bench_items — tích hợp toàn bộ
# ══════════════════════════════════════════════════════════════════════

def test_clear_build_ict_bench_items_all_types():
    """Chart chứa đủ 3 loại event -> đủ 3 nhóm BenchItem."""
    charts = [_single_swept_chart(), _single_fvg_chart(), _single_shift_chart()]
    items = build_ict_bench_items(charts, seed=42)

    assert len(items["swept"]) >= 1
    assert len(items["fvg"]) >= 1
    assert len(items["shift"]) >= 1


def test_near_miss_no_event_chart_produces_no_items():
    """Chart không có event nào -> KHÔNG sinh BenchItem nào cho chart đó
    (không crash, không tạo item rác)."""
    items = build_ict_bench_items([_no_event_chart()], seed=42)
    assert items["swept"] == []
    assert items["fvg"] == []
    assert items["shift"] == []


def test_clear_build_ict_bench_items_deterministic_with_same_seed():
    """Cùng seed -> sinh ra BenchItem giống hệt nhau (reproducibility)."""
    charts = [_single_swept_chart(), _single_fvg_chart()]
    items1 = build_ict_bench_items(charts, seed=7)
    items2 = build_ict_bench_items(charts, seed=7)

    assert items1["swept"][0].negative == items2["swept"][0].negative
    assert items1["fvg"][0].negative == items2["fvg"][0].negative