"""
candle_parser.py
================
Parse chuỗi dạng:
    "<chart> O_512 H_521 L_500 C_500 O_500 H_521 L_500 C_514 ... </chart>"
thành list các Candle (OHLC) theo đúng thứ tự xuất hiện.

Cách dùng:
    candles = parse_candles(raw_string)
    for c in candles:
        print(c)
"""

import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Candle:
    open:  int
    high:  int
    low:   int
    close: int

    def __repr__(self):
        return f"Candle(O={self.open}, H={self.high}, L={self.low}, C={self.close})"


# Regex: bắt 4 token O_<số> H_<số> L_<số> C_<số> liên tiếp (đúng thứ tự).
# Số có thể có dấu âm hoặc phần thập phân, nên cho phép \-?\d+(\.\d+)?
_CANDLE_PATTERN = re.compile(
    r"O_(-?\d+(?:\.\d+)?)\s+"
    r"H_(-?\d+(?:\.\d+)?)\s+"
    r"L_(-?\d+(?:\.\d+)?)\s+"
    r"C_(-?\d+(?:\.\d+)?)"
)


def parse_candles(raw_text: str) -> List[Candle]:
    """
    Parse chuỗi chứa các token O_/H_/L_/C_ thành list Candle theo thứ tự xuất hiện.

    Hàm tự bỏ qua các tag bao ngoài như <chart> ... </chart> vì regex chỉ
    tìm đúng pattern O_x H_x L_x C_x, không quan tâm phần text khác.

    Parameters
    ----------
    raw_text : str
        Chuỗi đầu vào, ví dụ:
        "<chart> O_512 H_521 L_500 C_500 O_500 H_521 L_500 C_514 </chart>"

    Returns
    -------
    List[Candle]
        Danh sách Candle theo đúng thứ tự xuất hiện trong chuỗi.
    """
    candles: List[Candle] = []
    for match in _CANDLE_PATTERN.finditer(raw_text):
        o, h, l, c = match.groups()
        candles.append(Candle(
            open=int(o),
            high=int(h),
            low=int(l),
            close=int(c),
        ))
    return candles


def parse_candles_to_dicts(raw_text: str) -> List[dict]:
    """Giống parse_candles nhưng trả về list[dict] (tiện cho json/pandas)."""
    return [
        {"Open": c.open, "High": c.high, "Low": c.low, "Close": c.close}
        for c in parse_candles(raw_text)
    ]


# ══════════════════════════════════════════════════════════════════
# DEMO / SELF-TEST
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    sample = """<chart> O_512 H_521 L_500 C_500 O_500 H_521 L_500 C_514 O_514 H_516 L_500 C_510 O_508 H_511 L_499 C_503 O_502 H_502 L_475 C_491 </chart>"""

    result = parse_candles(sample)
    print(f"Tổng số nến parse được: {len(result)}\n")
    for i, candle in enumerate(result):
        print(f"  [{i}] {candle}")