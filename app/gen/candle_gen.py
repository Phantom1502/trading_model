
import json

from app.gen.base_gen import BaseGenerator
from ict.candle import Candle
from typing import List, Dict
import random

CANDLESTICK_TEMPLATES = {
    "FULL_CANDLE": [
        "Cây nến {chart} có giá mở cửa {o}, cao nhất {h}, thấp nhất {l}, đóng cửa {c}. Thân nến dao động từ {body_start} đến {body_end}, râu trên từ {upper_wick_start} đến {upper_wick_end}, và râu dưới từ {lower_wick_start} đến {lower_wick_end}.",
        "Với nến {chart}, giá mở cửa là {o}, giá cao nhất là {h}, giá thấp nhất là {l} và giá đóng cửa là {c}. Phần thân nến trải dài từ {body_start} đến {body_end}, có râu trên vươn tới {upper_wick_end} từ {upper_wick_start} và râu dưới kéo dài từ {lower_wick_start} đến {lower_wick_end}.",
        "Chi tiết nến {chart}: mở cửa tại {o}, đỉnh {h}, đáy {l}, đóng cửa {c}. Thân nến biểu thị sự biến động từ {body_start} đến {body_end}, với râu trên từ {upper_wick_start} lên {upper_wick_end} và râu dưới từ {lower_wick_start} xuống {lower_wick_end}.",
        "Mô tả nến {chart}: giá {o} khi mở, {h} cao nhất, {l} thấp nhất, {c} khi đóng. Thân nến nằm giữa {body_start} và {body_end}, râu trên từ {upper_wick_start} đến {upper_wick_end}, râu dưới từ {lower_wick_start} đến {lower_wick_end}.",
        "Cây nến {chart} ghi nhận giá mở {o}, cao {h}, thấp {l}, đóng {c}. Thân nến hình thành từ {body_start} đến {body_end}, râu trên là khoảng từ {upper_wick_start} tới {upper_wick_end}, và râu dưới là từ {lower_wick_start} tới {lower_wick_end}.",
        "Phân tích nến {chart}: Mở cửa {o}, cao nhất {h}, thấp nhất {l}, đóng cửa {c}. Thân nến cho thấy vùng giá từ {body_start} đến {body_end}, râu trên kéo dài từ {upper_wick_start} lên {upper_wick_end}, râu dưới từ {lower_wick_start} xuống {lower_wick_end}.",
        "Cây nến {chart} có các mức giá quan trọng: mở {o}, cao {h}, thấp {l}, đóng {c}. Thân nến bao trọn từ {body_start} đến {body_end}, với bóng trên từ {upper_wick_start} đến {upper_wick_end} và bóng dưới từ {lower_wick_start} đến {lower_wick_end}.",
        "Đối với nến {chart}, giá mở cửa là {o}, giá cao nhất là {h}, giá thấp nhất là {l}, và giá đóng cửa là {c}. Thân nến thể hiện biên độ giao dịch chính từ {body_start} đến {body_end}, râu trên từ {upper_wick_start} đến {upper_wick_end}, râu dưới từ {lower_wick_start} đến {lower_wick_end}.",
        "Cây nến {chart} có {o} là giá mở, {h} là giá cao nhất, {l} là giá thấp nhất, {c} là giá đóng cửa. Thân nến thể hiện vùng giá từ {body_start} đến {body_end}, râu trên từ {upper_wick_start} đến {upper_wick_end}, và râu dưới từ {lower_wick_start} đến {lower_wick_end}.",
        "Tổng quan nến {chart}: giá mở {o}, cao {h}, thấp {l}, đóng {c}. Thân nến chính từ {body_start} đến {body_end}, phần râu trên từ {upper_wick_start} đến {upper_wick_end}, và phần râu dưới từ {lower_wick_start} đến {lower_wick_end}."
    ]
}

class CandleGenerator(BaseGenerator):
    def __init__(self, tokenizer=None, num_samples: int = 1024, sample_per_count = 50):
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
                high_distance = random.randint(0, 20)
                low_distance = random.randint(0, 20)
                high_clip = min(1023, self._count + high_distance)
                low_clip = max(0, self._count - low_distance)
                close_price = random.randint(low_clip, high_clip)
                candle = Candle(
                    open=self._count,
                    high=high_clip,
                    low=low_clip,
                    close=close_price
                )

                template = random.choice(CANDLESTICK_TEMPLATES["FULL_CANDLE"])
                gen_text = template.format(
                    chart=candle.tag(),
                    o=candle.open,
                    h=candle.high,
                    l=candle.low,
                    c=candle.close,
                    body_start=min(candle.open, candle.close),
                    body_end=max(candle.open, candle.close),
                    upper_wick_start=max(candle.open, candle.close),
                    upper_wick_end=candle.high,
                    lower_wick_start=candle.low,
                    lower_wick_end=min(candle.open, candle.close)
                )
                meta = {
                    "type": "full_candlestick"
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