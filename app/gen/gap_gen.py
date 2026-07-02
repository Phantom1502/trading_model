import json

from app.gen.base_gen import BaseGenerator
from ict.candle import Candle, build_raw_text
from typing import List, Dict
import random


CANDLESTICK_GAP_TEMPLATES = {
    "GAP_UP": [
        "Cặp nến {chart} tạo ra gap up tại {low} - {high}",
        "Hai nến {chart} hình thành gap up {low} - {high}",
        "Hai nến {chart} tạo khoảng trống giá {low} - {high}",
        "Hai nến {chart} tạo khoảng trống {low} - {high}",
    ],
    "GAP_DOWN": [
        "Cặp nến {chart} tạo ra gap down tại {low} - {high}",
        "Hai nến {chart} hình thành gap down {low} - {high}",
        "Hai nến {chart} tạo khoảng trống giá {low} - {high}",
        "Hai nến {chart} tạo khoảng trống {low} - {high}",
    ]
}

class GapGenerator(BaseGenerator):
    def __init__(self, tokenizer=None, num_samples: int = 1024, sample_per_count = 5):
        super().__init__(tokenizer)
        self.num_samples = num_samples
        self.sample_per_count = sample_per_count
        self._count = 0

    def __next__(self) -> List[Dict]:
        if self._count >= self.num_samples:
            raise StopIteration

        batch = []
        while len(batch) < self.batch_size:
            if self._count >= self.num_samples:
                break

            for _ in range(self.sample_per_count): # self.sample_per_count samples per count
                for gap_type in ["GAP_UP", "GAP_DOWN"]:
                    high_distance = random.randint(0, 20)
                    low_distance = random.randint(0, 20)
                    high_clip = min(1023, self._count + high_distance)
                    low_clip = max(0, self._count - low_distance)
                    close_price = random.randint(low_clip, high_clip)
                    candle1 = Candle(
                        open=self._count,
                        high=high_clip,
                        low=low_clip,
                        close=close_price
                    )
                    gap = random.randint(1, 10)
                    open_price2 = close_price + gap if gap_type == "GAP_UP" else close_price - gap

                    if open_price2 < 0 or open_price2 > 1023:
                        continue

                    high_distance2 = random.randint(0, 20)
                    low_distance2 = random.randint(0, 20)
                    high_clip2 = min(1023, open_price2 + high_distance2)
                    low_clip2 = max(0, open_price2 - low_distance2)
                    close_price2 = random.randint(low_clip2, high_clip2)
                    candle2 = Candle(
                        open=open_price2,
                        high=high_clip2,
                        low=low_clip2,
                        close=close_price2
                    )

                    template = random.choice(CANDLESTICK_GAP_TEMPLATES[gap_type])
                    chart = build_raw_text([candle1, candle2])
                    gap_low = min(candle1.close, candle2.open)
                    gap_high = max(candle1.close, candle2.open)
                    gen_text = template.format(chart=chart, low=gap_low, high=gap_high)
                    meta = {
                        "type": "gap"
                    }
                    token_length = len(self.tokenizer.encode(gen_text)) if self.tokenizer else 0
                    batch.append({
                        "text": gen_text,
                        "source": "trading",
                        "token_length": token_length,
                        "meta": json.dumps(meta, ensure_ascii=False)
                    })
            self._count += 1

        return batch