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

    @classmethod
    def from_candles(cls, candles: List[Candle], swing_window: int = 2) -> "CandleParser":
        """
        Tạo CandleParser trực tiếp từ 1 list Candle đã có sẵn (không parse lại text).
        Dùng khi đã CẮT một đoạn con từ list candles của 1 parser lớn hơn — `raw_text`
        sẽ được TỰ SINH LẠI đúng theo các Candle này, để header hiển thị trong dataset
        khớp chính xác với nội dung đang được phân tích (tránh lệch giữa raw_text gốc
        100 nến và nội dung thực tế chỉ còn 20-30 nến).
        """
        obj = cls.__new__(cls)
        obj.swing_window = swing_window
        obj.candles      = list(candles)
        obj.raw_text      = obj._build_raw_text(candles)
        return obj

    @staticmethod
    def _build_raw_text(candles: List[Candle]) -> str:
        """Sinh lại chuỗi <chart> O_.. H_.. L_.. C_.. ... </chart> từ list Candle."""
        tokens = [
            f"O_{c.open:g} H_{c.high:g} L_{c.low:g} C_{c.close:g}"
            for c in candles
        ]
        return "<chart> " + " ".join(tokens) + " </chart>"

    def slice(self, start: int, end: int) -> "CandleParser":
        """
        Cắt một đoạn con [start, end) từ parser hiện tại, trả về CandleParser mới
        độc lập (raw_text tự sinh lại đúng theo đoạn đã cắt).

        Lưu ý: cắt SAU khi đã parse xong (trên List[Candle]), không cắt trên text thô —
        tránh việc cắt giữa 1 token O_/H_/L_/C_ làm hỏng dữ liệu.
        """
        return CandleParser.from_candles(self.candles[start:end], swing_window=self.swing_window)

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

    # ──────────────────────────────────────────────
    # PATTERN HELPERS (nội bộ): body / wick của 1 nến
    # ──────────────────────────────────────────────
    @staticmethod
    def _body(c: Candle) -> float:
        return abs(c.close - c.open)

    @staticmethod
    def _upper_wick(c: Candle) -> float:
        return c.high - max(c.open, c.close)

    @staticmethod
    def _lower_wick(c: Candle) -> float:
        return min(c.open, c.close) - c.low

    @staticmethod
    def _range(c: Candle) -> float:
        r = c.high - c.low
        return r if r > 0 else 1e-9   # tránh chia 0 với nến range=0

    # ──────────────────────────────────────────────
    # MẪU HÌNH 1 NẾN: Pin Bar / Hammer / Shooting Star
    # ──────────────────────────────────────────────
    def is_pin_bar(
        self, index: int,
        wick_ratio: float = 0.6,
        body_ratio: float = 0.3,
    ) -> Optional[str]:
        """
        Phát hiện mẫu hình 1 nến dạng Pin Bar (bóng nến dài 1 bên, thân nhỏ).

        - "HAMMER"         : bóng dưới dài (>= wick_ratio * range), thân nhỏ
                              (<= body_ratio * range), nằm ở phần trên của range
                              → thường xuất hiện sau giảm, gợi ý khả năng đảo chiều TĂNG.
        - "SHOOTING_STAR"  : bóng trên dài, thân nhỏ, nằm ở phần dưới của range
                              → thường xuất hiện sau tăng, gợi ý khả năng đảo chiều GIẢM.
        - None             : không phải Pin Bar.

        Đây là phát hiện ĐỘC LẬP theo hình dạng nến, không xét đến vị trí
        (không cần đang ở swing high/low) — chỉ mô tả hình dạng và ý nghĩa
        thống kê thường gặp của mẫu hình đó.
        """
        c = self.candles[index]
        rng  = self._range(c)
        body = self._body(c)
        upper = self._upper_wick(c)
        lower = self._lower_wick(c)

        if body > body_ratio * rng:
            return None   # thân quá lớn, không phải Pin Bar

        if lower >= wick_ratio * rng and upper < lower * 0.5:
            return "HAMMER"
        if upper >= wick_ratio * rng and lower < upper * 0.5:
            return "SHOOTING_STAR"
        return None

    # ──────────────────────────────────────────────
    # MẪU HÌNH 2 NẾN: Bullish / Bearish Engulfing
    # ──────────────────────────────────────────────
    def is_engulfing(self, index: int) -> Optional[str]:
        """
        Phát hiện mẫu hình Engulfing dùng 2 nến: (index-1, index).

        - "BULLISH_ENGULFING" : nến (index-1) giảm, nến (index) tăng và
          thân nến (index) "nuốt trọn" thân nến (index-1)
          (Open[index] <= Close[index-1] và Close[index] >= Open[index-1])
          → thường gợi ý khả năng đảo chiều TĂNG.
        - "BEARISH_ENGULFING" : nến (index-1) tăng, nến (index) giảm và
          thân nến (index) nuốt trọn thân nến (index-1)
          → thường gợi ý khả năng đảo chiều GIẢM.
        - None : không phải Engulfing.
        """
        if index - 1 < 0:
            return None

        prev = self.candles[index - 1]
        curr = self.candles[index]

        prev_dir = self.is_bull_bear(index - 1)
        curr_dir = self.is_bull_bear(index)

        if prev_dir == "BEAR" and curr_dir == "BULL":
            if curr.open <= prev.close and curr.close >= prev.open:
                return "BULLISH_ENGULFING"

        if prev_dir == "BULL" and curr_dir == "BEAR":
            if curr.open >= prev.close and curr.close <= prev.open:
                return "BEARISH_ENGULFING"

        return None
    
    # ── Swing ──────────────────────────────────────────────────────
    def is_swing_high(self, i: int, window: int = None) -> bool:
        w = window or self.swing_window
        if i - w < 0 or i + w >= len(self.candles): return False
        target = self.candles[i].high
        return target == max(self.candles[j].high for j in range(i - w, i + w + 1))
 
    def is_swing_low(self, i: int, window: int = None) -> bool:
        w = window or self.swing_window
        if i - w < 0 or i + w >= len(self.candles): return False
        target = self.candles[i].low
        return target == min(self.candles[j].low for j in range(i - w, i + w + 1))
 
    # ── Sweep helpers ──────────────────────────────────────────────
    def _find_active_swing_high(self, i: int, lookback: int, w: int) -> Optional[tuple[int, float]]:
        start = max(0, i - lookback)
        for j in range(i - w - 1, start - 1, -1):
            if j - w < 0:
                continue
            if not self.is_swing_high(j, w):
                continue
            level = self.candles[j].high
            already_broken = any(self.candles[k].close > level for k in range(j + 1, i))
            if already_broken:
                continue
            return j, level
        return None
 
    def _find_active_swing_low(self, i: int, lookback: int, w: int) -> Optional[tuple[int, float]]:
        start = max(0, i - lookback)
        for j in range(i - w - 1, start - 1, -1):
            if j - w < 0:
                continue
            if not self.is_swing_low(j, w):
                continue
            level = self.candles[j].low
            already_broken = any(self.candles[k].close < level for k in range(j + 1, i))
            if already_broken:
                continue
            return j, level
        return None
 
    # ── Sweep (Liquidity Grab) ───────────────────────────────────────
    def is_swept(self, i: int, lookback: int = 20, swing_window: int = None) -> Optional[dict]:
        """
        Trả về dict số hóa đầy đủ thay vì chỉ string, để unit test/đo lường dễ dàng:
        {
          "type": "BEARISH_SWEEP" | "BULLISH_SWEEP",
          "swept_candle_idx": i,
          "swing_idx": j,
          "swing_level": float,
          "depth": float,          # độ sâu xuyên qua (đơn vị giá/bin, tuỳ input)
        }
        Trả về None nếu không có sweep.
        """
        w = swing_window or self.swing_window
        if i - w < 0:
            return None
        curr = self.candles[i]
 
        high_swing = self._find_active_swing_high(i, lookback, w)
        if high_swing is not None:
            j, level = high_swing
            if curr.high > level and curr.close < level:
                return {
                    "type": "BEARISH_SWEEP",
                    "swept_candle_idx": i,
                    "swing_idx": j,
                    "swing_level": level,
                    "depth": round(curr.high - level, 6),
                }
 
        low_swing = self._find_active_swing_low(i, lookback, w)
        if low_swing is not None:
            j, level = low_swing
            if curr.low < level and curr.close > level:
                return {
                    "type": "BULLISH_SWEEP",
                    "swept_candle_idx": i,
                    "swing_idx": j,
                    "swing_level": level,
                    "depth": round(level - curr.low, 6),
                }
 
        return None