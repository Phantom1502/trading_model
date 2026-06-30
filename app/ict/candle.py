"""
candle.py — Lớp 0: Candle + parse token text -> list Candle
==================================================================
Giá trị open/high/low/close ở đây LUÔN LÀ BIN INDEX (số nguyên parse trực
tiếp từ token O_xxx/H_xxx/L_xxx/C_xxx), KHÔNG PHẢI giá thật. Mọi detector
trong package này tính toán hoàn toàn trên bin index — đây là nguyên tắc
khóa cứng đã thống nhất (xem spec mục 4): label luôn nhất quán với chính
token mà model nhìn thấy, không leak thông tin giá thật mà model không có.
"""

import re
from dataclasses import dataclass
from typing import List


_CANDLE_TOKEN_RE = re.compile(
    r"O_(-?\d+)\s+"
    r"H_(-?\d+)\s+"
    r"L_(-?\d+)\s+"
    r"C_(-?\d+)"
)


@dataclass
class Candle:
    """1 nến, 4 giá trị đều là BIN INDEX (không phải giá thật)."""
    open : int
    high : int
    low  : int
    close: int

    def __repr__(self):
        return f"Candle(O={self.open}, H={self.high}, L={self.low}, C={self.close})"

    def body(self) -> int:
        return abs(self.close - self.open)

    def range(self) -> int:
        r = self.high - self.low
        return r if r > 0 else 1   # tránh chia 0 cho nến range=0 (toàn bộ giá trị bằng nhau)

    def upper_wick(self) -> int:
        return self.high - max(self.open, self.close)

    def lower_wick(self) -> int:
        return min(self.open, self.close) - self.low

    def tag(self) -> str:
        """Đóng gói lại thành <chart>...</chart> 1 nến — đi đúng price_vocab khi
        trích dẫn lại nến trong câu mô tả (xem spec mục 6)."""
        return f"<chart> O_{self.open} H_{self.high} L_{self.low} C_{self.close} </chart>"


def parse_candles(raw_text: str) -> List[Candle]:
    """Parse chuỗi '<chart> O_.. H_.. L_.. C_.. ... </chart>' -> list Candle (bin index)."""
    candles: List[Candle] = []
    for o, h, l, c in _CANDLE_TOKEN_RE.findall(raw_text):
        candles.append(Candle(open=int(o), high=int(h), low=int(l), close=int(c)))
    return candles


def build_raw_text(candles: List[Candle]) -> str:
    """Sinh lại chuỗi token chuẩn từ list Candle — dùng khi CandleParser.slice()
    cần raw_text khớp đúng đoạn đã cắt (xem test_slice.py, case tie_breaking_raw_text_rebuild)."""
    tokens = [f"O_{c.open} H_{c.high} L_{c.low} C_{c.close}" for c in candles]
    return "<chart> " + " ".join(tokens) + " </chart>"