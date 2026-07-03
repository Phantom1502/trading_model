"""
tests/test_validate.py — validate.py: cross-consistency + no-leakage
============================================================================
"""

import random

from app.ict.candle import Candle
from app.ict.parser import CandleParser
from app.ict.facts import build_facts
from app.ict.render import (
    render_swept_sample, render_fvg_sample, render_shift_sample,
    render_synthesis_sample,
)
from app.ict.validate import validate_cross_consistency, validate_no_leakage, _parse_eval


def _c(o, h, l, c):
    return Candle(open=o, high=h, low=l, close=c)


def _multi_event_chart():
    return [
        _c(495, 505, 490, 500), _c(498, 508, 493, 503), _c(505, 530, 500, 525),
        _c(520, 518, 510, 515), _c(515, 513, 505, 510), _c(510, 535, 505, 512),
        _c(512, 516, 508, 514), _c(514, 522, 510, 518), _c(519, 521, 517, 520),
        _c(525, 535, 524, 530), _c(528, 532, 522, 525), _c(523, 528, 518, 520),
        _c(520, 545, 515, 540), _c(535, 533, 525, 528), _c(525, 523, 515, 518),
        _c(515, 555, 510, 550),
    ]


def _build_facts_and_raw(candles, initial_trend="BULL"):
    parser = CandleParser.from_candles(candles, swing_window=2)
    facts = build_facts(parser, initial_trend=initial_trend, lookback=10)
    return facts, parser.raw_text


def test_clear_parse_eval_basic():
    """_parse_eval() tách đúng KEY=VALUE từ block <eval>...</eval>."""
    fields = _parse_eval("<eval>TYPE=BULL CANDLE=3 GAP_SIZE=15</eval>")
    assert fields == {"TYPE": "BULL", "CANDLE": "3", "GAP_SIZE": "15"}


def test_clear_cross_consistency_passes_for_real_samples():
    """4 mẫu render từ CÙNG 1 fact JSON -> cross-consistency PHẢI pass
    (đây là baseline: render.py đúng theo construction, nên tự nhiên nhất
    quán, không cần "sửa" gì để pass)."""
    facts, raw = _build_facts_and_raw(_multi_event_chart())
    rng = random.Random(1)

    samples = [
        render_swept_sample(facts, raw, rng=rng),
        render_fvg_sample(facts, raw, rng=rng),
        render_shift_sample(facts, raw, rng=rng),
        render_synthesis_sample(facts, raw, rng=rng),
    ]
    samples = [s for s in samples if s is not None]
    assert len(samples) == 4

    assert validate_cross_consistency(samples) is True


def test_clear_cross_consistency_single_sample_trivially_true():
    """Chỉ 1 mẫu (không đủ để so sánh chéo) -> mặc định True, không lỗi."""
    facts, raw = _build_facts_and_raw(_multi_event_chart())
    sample = render_swept_sample(facts, raw, rng=random.Random(1))
    assert validate_cross_consistency([sample]) is True
    assert validate_cross_consistency([]) is True


def test_near_miss_cross_consistency_catches_mismatch():
    """Cố tình đưa vào 1 mẫu "giả" TRÙNG định danh (TYPE+CANDLE) với 1 event
    thật, nhưng field khác (SWING_LEVEL) SAI giá trị -> validate PHẢI False."""
    facts, raw = _build_facts_and_raw(_multi_event_chart())
    rng = random.Random(1)

    sample_shift = render_shift_sample(facts, raw, rng=rng)
    real_fields = _parse_eval(sample_shift["eval"])   # TYPE=BOS CANDLE=<thật>...

    fake_synthesis = {
        "chart": raw,
        "request": "fake",
        "explanation": "fake",
        # Cùng TYPE + CANDLE (cùng định danh event) nhưng SWING_LEVEL SAI
        "eval": f"<eval>TYPE={real_fields['TYPE']} CANDLE={real_fields['CANDLE']} SWING_LEVEL=999999</eval>",
        "event_count": 1,
    }
    assert validate_cross_consistency([sample_shift, fake_synthesis]) is False


def test_clear_no_leakage_passes_for_real_samples():
    """Mọi số trong Lý giải render từ template PHẢI truy được nguồn từ Eval
    (vì cả 2 đều lấy từ cùng fact dict qua string interpolation)."""
    facts, raw = _build_facts_and_raw(_multi_event_chart())
    rng = random.Random(2)

    for render_fn in (render_swept_sample, render_fvg_sample, render_shift_sample, render_synthesis_sample):
        sample = render_fn(facts, raw, rng=rng)
        if sample is not None:
            assert validate_no_leakage(sample) is True


def test_near_miss_no_leakage_catches_extra_number():
    """Explanation có số KHÔNG xuất hiện trong Eval -> validate PHẢI trả về False."""
    fake_sample = {
        "chart": "<chart></chart>",
        "request": "fake",
        "explanation": "Nến thứ 999 có điều gì đó bất thường.",   # số 999 không có trong eval
        "eval": "<eval>TYPE=BULL CANDLE=3</eval>",
        "event_count": 1,
    }
    assert validate_no_leakage(fake_sample) is False


def test_clear_no_leakage_every_template_variant():
    """
    Duyệt qua TOÀN BỘ template variant (full + short, mọi loại event) —
    không chỉ vài mẫu random. Bug thật đã tìm thấy trên data production
    quy mô lớn (5.7M dòng, ~35k fail): 2 template FVG có chữ số "1" tự
    nhiên trong câu tiếng Việt ("hình thành 1 khoảng trống") bị
    validate_no_leakage hiểu nhầm là leak số liệu — sandbox test trước đó
    dùng random seed nhỏ, không may mắn hit đúng 2 template này.

    Test này duyệt HẾT mọi template (deterministic, không phụ thuộc seed)
    để đảm bảo loại bug này không lọt qua lần sau.
    """
    from app.ict.render import _TEMPLATES

    # 1 event mẫu cho mỗi kind, đủ field để format mọi template
    sample_events = {
        "SWEEP_HIGH": {"swept_candle_idx": 5, "swing_idx": 2, "swing_level": 530, "depth": 10, "type": "SWEEP_HIGH"},
        "SWEEP_LOW" : {"swept_candle_idx": 5, "swing_idx": 2, "swing_level": 480, "depth": 10, "type": "SWEEP_LOW"},
        "BULL"      : {"fvg_candle_idx": 5, "gap_low": 510, "gap_high": 525, "gap_size_bins": 15, "fill_pct": 33.3, "type": "BULL"},
        "BEAR"      : {"fvg_candle_idx": 5, "gap_low": 510, "gap_high": 525, "gap_size_bins": 15, "fill_pct": 33.3, "type": "BEAR"},
        "BOS"       : {"shift_candle_idx": 5, "swing_idx": 2, "swing_level": 530, "direction": "BULL", "broken_type": "HIGH", "type": "BOS"},
        "CHoCH"     : {"shift_candle_idx": 5, "swing_idx": 2, "swing_level": 530, "direction": "BEAR", "broken_type": "HIGH", "type": "CHoCH"},
    }

    fail_count = 0
    fail_details = []

    for kind, event in sample_events.items():
        for use_short in (False, True):
            bank = _TEMPLATES[kind]["short" if use_short else "full"]
            for template_idx, template in enumerate(bank):
                rng = random.Random(0)
                # Ép rng chọn đúng template_idx bằng cách patch tạm bank về 1 phần tử
                # (đơn giản hơn mock random) — gọi trực tiếp .format() để test từng
                # template độc lập, không phụ thuộc random.choice.
                from app.ict.render import _swept_fields, _fvg_fields, _shift_fields

                if "swept_candle_idx" in event:
                    f = _swept_fields(event)
                    explanation = template.format(c=f["CANDLE"], s=f["SWING_CANDLE"], level=f["SWING_LEVEL"], depth=f["DEPTH"])
                    eval_block = f"<eval>TYPE={f['TYPE']} CANDLE={f['CANDLE']} SWING_CANDLE={f['SWING_CANDLE']} SWING_LEVEL={f['SWING_LEVEL']} DEPTH={f['DEPTH']}</eval>"
                elif "fvg_candle_idx" in event:
                    f = _fvg_fields(event)
                    explanation = template.format(c=f["CANDLE"], size=f["GAP_SIZE"], fill=f["FILL_PCT"])
                    eval_block = f"<eval>TYPE={f['TYPE']} CANDLE={f['CANDLE']} GAP_LOW={f['GAP_LOW']} GAP_HIGH={f['GAP_HIGH']} GAP_SIZE={f['GAP_SIZE']} FILL_PCT={f['FILL_PCT']}</eval>"
                else:
                    f = _shift_fields(event)
                    explanation = template.format(c=f["CANDLE"], s=f["SWING_CANDLE"], level=f["SWING_LEVEL"], dir=f["DIRECTION"])
                    eval_block = f"<eval>TYPE={f['TYPE']} DIRECTION={f['DIRECTION']} CANDLE={f['CANDLE']} SWING_CANDLE={f['SWING_CANDLE']} SWING_LEVEL={f['SWING_LEVEL']} BROKEN={f['BROKEN']}</eval>"

                sample = {"explanation": explanation, "eval": eval_block}
                if not validate_no_leakage(sample):
                    fail_count += 1
                    fail_details.append((kind, "short" if use_short else "full", template_idx, explanation))

    assert fail_count == 0, f"{fail_count} template gây false-positive leak: {fail_details}"