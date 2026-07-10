import re
from dataclasses import dataclass
from typing import List, Optional

_CANDLE_PATTERN = re.compile(
    r"O_(-?\d+(?:\.\d+)?)\s+"
    r"H_(-?\d+(?:\.\d+)?)\s+"
    r"L_(-?\d+(?:\.\d+)?)\s+"
    r"C_(-?\d+(?:\.\d+)?)"
)

@dataclass
class Candle:
    open: int
    high: int
    low: int
    close: int
    
class CandleParser:
    def __init__(self, text: str):
        self.text = text
        self.candles: List[Candle] = self.parse_candles(text)
        
    @staticmethod
    def parse_candles(text: str) -> List[Candle]:
        candles = []
        for match in _CANDLE_PATTERN.finditer(text):
            o, h, l, c = map(int, match.groups())
            candles.append(Candle(open=o, high=h, low=l, close=c))
        return candles
    
if __name__ == "__main__":
    sample_text = "O_100 H_110 L_90 C_105 O_105 H_115 L_95 C_110"
    parser = CandleParser(sample_text)
    for candle in parser.candles:
        print(candle)