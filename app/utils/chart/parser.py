"""
candle_parser.py
================
Class CandleParser: parse chuỗi dạng
    "<chart> O_512 H_521 L_500 C_500 ... </chart>"
thành list Candle, đồng thời cung cấp sẵn các hàm phân tích ICT
(swing high/low, FVG, bull/bear) và hàm generate dataset text mô tả
từng cây nến — dùng để tạo training data / mô tả tự nhiên cho LLM.

Cách dùng nhanh:
    parser = CandleParser(raw_text)
    print(parser.to_dataset_text())
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
        """Nến tăng: Close > Open."""
        return self.close > self.open + 1 # threshold 1 để tránh nến doji (giá không đổi)

    def is_bear(self) -> bool:
        """Nến giảm: Close < Open."""
        return self.close < self.open - 1 # threshold 1 để tránh nến doji (giá không đổi)

    def tag(self) -> str:
        """Token gọn để in trong dataset text, ví dụ <O_512 H_521 L_500 C_500>."""
        return f"<chart> O_{self.open:g} H_{self.high:g} L_{self.low:g} C_{self.close:g} </chart>"

    def description(self, index: int) -> str:
        """Mô tả tự nhiên cho nến này: "nến tăng/giảm/doji"."""
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

# Regex: bắt 4 token O_<số> H_<số> L_<số> C_<số> liên tiếp theo đúng thứ tự.
_CANDLE_PATTERN = re.compile(
    r"O_(-?\d+(?:\.\d+)?)\s+"
    r"H_(-?\d+(?:\.\d+)?)\s+"
    r"L_(-?\d+(?:\.\d+)?)\s+"
    r"C_(-?\d+(?:\.\d+)?)"
)


class CandleParser:
    """
    Parse chuỗi OHLC thô và cung cấp các hàm phân tích ICT trên list Candle.

    Attributes
    ----------
    candles : List[Candle]   — danh sách nến đã parse, theo đúng thứ tự xuất hiện
    """

    def __init__(self, raw_text: str, swing_window: int = 2):
        """
        Parameters
        ----------
        raw_text     : chuỗi thô chứa các token O_/H_/L_/C_
        swing_window : số nến xét mỗi bên khi tính swing high/low (mặc định 2)
        """
        self.raw_text     = raw_text
        self.swing_window = swing_window
        self.candles: List[Candle] = self._parse(raw_text)

    # ──────────────────────────────────────────────
    # PARSE
    # ──────────────────────────────────────────────
    @staticmethod
    def _parse(raw_text: str) -> List[Candle]:
        candles: List[Candle] = []
        for match in _CANDLE_PATTERN.finditer(raw_text):
            o, h, l, c = match.groups()
            candles.append(Candle(
                open=float(o), high=float(h),
                low=float(l),  close=float(c),
            ))
        return candles

    def __len__(self) -> int:
        return len(self.candles)

    def __getitem__(self, idx) -> Candle:
        return self.candles[idx]

    def to_dicts(self) -> List[dict]:
        """Trả về list[dict] tiện cho json hoặc pandas.DataFrame(...)."""
        return [
            {"Open": c.open, "High": c.high, "Low": c.low, "Close": c.close}
            for c in self.candles
        ]

    # ──────────────────────────────────────────────
    # ICT HELPERS
    # ──────────────────────────────────────────────
    def is_swing_high(self, index: int, window: Optional[int] = None) -> bool:
        """
        Swing High: High tại `index` là cao nhất trong cụm [index-window, index+window].
        """
        window = window if window is not None else self.swing_window
        n = len(self.candles)
        if index - window < 0 or index + window >= n:
            return False
        target = self.candles[index].high
        window_highs = [self.candles[i].high for i in range(index - window, index + window + 1)]
        return target == max(window_highs)

    def is_swing_low(self, index: int, window: Optional[int] = None) -> bool:
        """
        Swing Low: Low tại `index` là thấp nhất trong cụm [index-window, index+window].
        """
        window = window if window is not None else self.swing_window
        n = len(self.candles)
        if index - window < 0 or index + window >= n:
            return False
        target = self.candles[index].low
        window_lows = [self.candles[i].low for i in range(index - window, index + window + 1)]
        return target == min(window_lows)

    def is_fvg(self, index: int) -> Optional[str]:
        """
        FVG 3-nến kết thúc tại `index` (dùng index - 2, index - 1, index).
        Returns "BULL" / "BEAR" / None.
        """
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
        """Phân loại nến tại `index`: "BULL" / "BEAR" / "DOJI"."""
        c = self.candles[index]
        if c.close > c.open:
            return "BULL"
        if c.close < c.open:
            return "BEAR"
        return "DOJI"

    # ──────────────────────────────────────────────
    # PHÂN TÍCH TOÀN BỘ (dùng nội bộ cho dataset generator)
    # ──────────────────────────────────────────────
    def analyze_index(self, index: int) -> Dict:
        """
        Trả về toàn bộ thông tin phân tích ICT của 1 nến tại `index`:
        bull/bear, swing high/low, fvg bắt đầu tại nến này,
        và fvg mà nến này là "nến thứ 3" hoàn thiện gap (index-2).
        """
        info = {
            "index":        index,
            "candle":       self.candles[index],
            "direction":    self.is_bull_bear(index),
            "swing_high":   self.is_swing_high(index),
            "swing_low":    self.is_swing_low(index),
            "fvg_start":    self.is_fvg(index),       # FVG bắt đầu TẠI nến này (dùng index, index+1, index+2)
            "fvg_complete": self.is_fvg(index - 2) if index - 2 >= 0 else None,  # FVG hoàn thiện BỞI nến này
        }
        return info

    # ──────────────────────────────────────────────
    # GENERATE DATASET TEXT
    # ──────────────────────────────────────────────
    def to_dataset_text(self, include_raw_chart: bool = True) -> str:
        """
        Sinh text mô tả tự nhiên cho từng cây nến, ví dụ:

            Đây là chart <chart> O_512 H_521 ... </chart>

            Cây nến thứ 1 <O_512 H_521 L_500 C_500> là nến giảm giá.
            Cây nến thứ 4 <O_508 H_511 L_499 C_503> là nến giảm giá.
            Cây nến thứ 4 cùng cây nến 2, 3 tạo thành FVG giảm (Bearish FVG).
            Cây nến thứ 5 <O_502 H_502 L_475 C_491> là swing low.
            ...

        Returns
        -------
        str — toàn bộ đoạn text dataset.
        """
        lines: List[str] = []

        if include_raw_chart:
            lines.append(f"Phân tích chart: {self.raw_text.strip()}")
            lines.append("")


        lines.append("Từ trên chart, phân tích từng cây nến như sau:")
        n = len(self.candles)
        for i in range(n):
            c = self.candles[i]
            ordinal = i + 1   # đánh số nến từ 1 cho tự nhiên

            lines.append(c.description(ordinal))

            # Swing high / low
            if self.is_swing_high(i):
                lines.append(f"Cây nến thứ {ordinal} là swing high (đỉnh cục bộ).")
            if self.is_swing_low(i):
                lines.append(f"Cây nến thứ {ordinal} là swing low (đáy cục bộ).")

            # FVG bắt đầu tại nến i (hoàn thiện bởi nến i+2)
            fvg = self.is_fvg(i)
            if fvg:
                third_ordinal = i - 1  # nến index-2, đánh số từ 1
                fvg_vn = "tăng (Bullish FVG)" if fvg == "BULL" else "giảm (Bearish FVG)"
                lines.append(
                    f"Cây nến thứ {ordinal} cùng cây nến thứ {i} và {third_ordinal} "
                    f"tạo thành FVG {fvg_vn}."
                )

            lines.append("")  # ngăn cách giữa các nến

        return "\n".join(lines).strip() + "\n"


# ══════════════════════════════════════════════════════════════════
# DEMO / SELF-TEST
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    sample = (
        "<chart> O_512 H_521 L_500 C_500 O_500 H_521 L_500 C_514 "
        "O_514 H_516 L_500 C_510 O_508 H_511 L_499 C_503 "
        "O_502 H_502 L_475 C_491 O_490 H_490 L_478 C_489 "
        "O_494 H_507 L_481 C_481 O_479 H_484 L_473 C_473 </chart>"
    )

    parser = CandleParser(sample, swing_window=2)
    print(f"Tổng số nến: {len(parser)}\n")

    print("=== Dataset text ===\n")
    print(parser.to_dataset_text())