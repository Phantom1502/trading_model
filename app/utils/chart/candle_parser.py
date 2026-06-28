"""
candle_parser.py  (đồng bộ với bản người dùng đã sửa)
================
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Dict


@dataclass
class Candle:
    open:  float
    high:  float
    low:   float
    close: float

    def __repr__(self):
        return f"Candle(O={self.open}, H={self.high}, L={self.low}, C={self.close})"

    def is_bull(self) -> bool:
        return self.close > self.open + 1

    def is_bear(self) -> bool:
        return self.close < self.open - 1

    def tag(self) -> str:
        return f"<chart> O_{self.open:g} H_{self.high:g} L_{self.low:g} C_{self.close:g} </chart>"

    def description(self, index: int) -> str:
        lines: List[str] = []
        lines.append(f"\tCây nến thứ {index} {self.tag()} là ",)
        if self.is_bull():
            lines.append("nến tăng giá")
        elif self.is_bear():
            lines.append("nến giảm giá")
        else:
            lines.append("nến doji (giá không đổi)")
        lines.append(f" có giá mở cửa {self.open:g}, giá đóng cửa {self.close:g}, giá cao nhất {self.high:g}, giá thấp nhất {self.low:g}.")
        return "".join(lines)


_CANDLE_PATTERN = re.compile(
    r"O_(-?\d+(?:\.\d+)?)\s+"
    r"H_(-?\d+(?:\.\d+)?)\s+"
    r"L_(-?\d+(?:\.\d+)?)\s+"
    r"C_(-?\d+(?:\.\d+)?)"
)


class CandleParser:
    def __init__(self, raw_text: str, swing_window: int = 2):
        self.raw_text     = raw_text
        self.swing_window = swing_window
        self.candles: List[Candle] = self._parse(raw_text)

    @staticmethod
    def _parse(raw_text: str) -> List[Candle]:
        candles: List[Candle] = []
        for match in _CANDLE_PATTERN.finditer(raw_text):
            o, h, l, c = match.groups()
            candles.append(Candle(open=float(o), high=float(h), low=float(l), close=float(c)))
        return candles

    def __len__(self): return len(self.candles)
    def __getitem__(self, idx): return self.candles[idx]

    def to_dicts(self) -> List[dict]:
        return [{"Open": c.open, "High": c.high, "Low": c.low, "Close": c.close} for c in self.candles]

    def is_swing_high(self, index: int, window: Optional[int] = None) -> bool:
        window = window if window is not None else self.swing_window
        n = len(self.candles)
        if index - window < 0 or index + window >= n:
            return False
        target = self.candles[index].high
        window_highs = [self.candles[i].high for i in range(index - window, index + window + 1)]
        return target == max(window_highs)

    def is_swing_low(self, index: int, window: Optional[int] = None) -> bool:
        window = window if window is not None else self.swing_window
        n = len(self.candles)
        if index - window < 0 or index + window >= n:
            return False
        target = self.candles[index].low
        window_lows = [self.candles[i].low for i in range(index - window, index + window + 1)]
        return target == min(window_lows)

    def is_fvg(self, index: int) -> Optional[str]:
        if index - 2 < 0:
            return None
        c0 = self.candles[index - 2]
        c2 = self.candles[index]
        if c2.low > c0.high:
            return "BULL"
        if c2.high < c0.low:
            return "BEAR"
        return None

    def is_bull_bear(self, index: int) -> str:
        c = self.candles[index]
        if c.is_bull(): return "BULL"
        if c.is_bear(): return "BEAR"
        return "DOJI"