"""
parser.py — Lớp 0: CandleParser
====================================
Bọc list[Candle], cung cấp slice() và các helper window dùng chung cho
toàn bộ detector ở các lớp trên. Theo đúng quy ước trong spec (mục 13):
    CandleParser.from_candles(candles, swing_window=2)  — dùng khi viết test
    CandleParser(raw_text, swing_window=2)               — dùng khi parse thật
"""

from typing import List, Optional

from .candle import Candle, parse_candles, build_raw_text


class CandleParser:
    def __init__(self, raw_text: str, swing_window: int = 2):
        self.raw_text     = raw_text
        self.swing_window = swing_window
        self.candles: List[Candle] = parse_candles(raw_text)

    @classmethod
    def from_candles(cls, candles: List[Candle], swing_window: int = 2) -> "CandleParser":
        """Dựng trực tiếp từ list Candle có sẵn (không parse lại text) — dùng
        chủ yếu trong golden test, nơi case được khai báo bằng Candle(...) trực tiếp."""
        obj = cls.__new__(cls)
        obj.swing_window = swing_window
        obj.candles      = list(candles)
        obj.raw_text      = build_raw_text(candles)
        return obj

    def slice(self, start: int, end: int) -> "CandleParser":
        """Cắt [start, end) thành CandleParser mới, raw_text tự sinh lại đúng
        theo đoạn đã cắt (test_slice.py, case tie_breaking_raw_text_rebuild)."""
        return CandleParser.from_candles(self.candles[start:end], swing_window=self.swing_window)

    def __len__(self):
        return len(self.candles)

    def __getitem__(self, idx):
        return self.candles[idx]

    # ── Window helper dùng chung cho structure.py / ict.py ──────────────
    def window(self, center: int, w: Optional[int] = None) -> List[Candle]:
        """Trả về list Candle trong [center-w, center+w], rỗng nếu không đủ context
        2 bên (đúng hành vi edge_position_window_start/end trong spec mục 9)."""
        w = w if w is not None else self.swing_window
        n = len(self.candles)
        if center - w < 0 or center + w >= n:
            return []
        return self.candles[center - w: center + w + 1]