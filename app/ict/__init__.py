"""
app/ict — ICT Reward Model / Judge: detector + dataset pipeline
====================================================================
Package độc lập với app/memlm và app/utils, không import chéo.
Kế thừa Ý TƯỞNG từ app/utils/chart/candle_parser.py (Candle, CandleParser,
swing/FVG) nhưng viết lại riêng để package này tự chứa toàn bộ logic ICT
(Swept, FVG graded, Shift/MSS, relations) mà không phụ thuộc vào module
khác đang phục vụ mục đích khác (curriculum pretrain cơ bản).

Cấu trúc (đúng theo lộ trình trong spec, mục 10):
    candle.py        — Lớp 0: Candle, parse chuỗi token -> list Candle
    parser.py         — Lớp 0: CandleParser (slice, window helpers)
    basic.py            — Lớp 1-2: Bull/Bear, Pin Bar, Engulfing
    structure.py          — Lớp 3: Swing High/Low, FVG (binary)
    ict.py                  — Lớp 4: Swept, FVG graded, Shift/MSS
    relations.py               — Lớp 5: build_relations (Tầng 2 spec)
    facts.py                     — Gom toàn bộ detector -> 1 fact JSON / chart
    render.py                      — Fact JSON -> 4 dạng mẫu tin (curriculum)
    validate.py                      — Validate câu GPT sinh ra khớp fact JSON
    tests/                              — Golden test theo mục 9 trong spec
"""

from .candle import Candle, parse_candles
from .parser import CandleParser
from .basic import classify_direction, is_pin_bar, is_engulfing
from .structure import is_swing_high, is_swing_low, is_fvg
from .ict import is_swept, scan_all_swept, grade_fvg, is_shift, scan_all_shift
from .relations import build_relations
from .facts import build_facts

__all__ = [
    "Candle", "parse_candles",
    "CandleParser",
    "classify_direction", "is_pin_bar", "is_engulfing",
    "is_swing_high", "is_swing_low", "is_fvg",
    "is_swept", "scan_all_swept", "grade_fvg", "is_shift", "scan_all_shift",
    "build_relations",
    "build_facts",
]