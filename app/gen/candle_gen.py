
import json

from app.gen.base_gen import BaseGenerator

from typing import List, Dict
import random

CANDLESTICK_TEMPLATES = {
    "O": [
        "Cây nến {chart} có giá mở cửa là {i}",
        "{i} là mức giá mở cửa của cây nến {chart}",
        "Giá mở cửa của cây nến {chart} ghi nhận tại {i}",
        "Bắt đầu phiên, cây nến {chart} đứng ở mức {i}",
        "Phiên giao dịch khởi đầu với cây nến {chart} tại {i}",
        "Mức giá đầu tiên của cây nến {chart} là {i}",
        "Cây nến {chart} mở màn tại ngưỡng {i}",
        "Tại thời điểm mở cửa, cây nến {chart} có giá {i}",
        "Ghi nhận mức giá mở phiên {i} cho cây nến {chart}",
        "Cây nến {chart} xuất phát ở mức giá {i}"
    ],
    "H": [
        "Cây nến {chart} có giá cao nhất {i}",
        "Mức cao kỷ lục của cây nến {chart} là {i}",
        "Cây nến {chart} đạt đỉnh tại {i}",
        "Giá cao nhất ghi nhận được cho cây nến {chart} là {i}",
        "Trong phiên, cây nến {chart} vươn tới mức {i}",
        "Ngưỡng cao nhất mà cây nến {chart} đạt được là {i}",
        "Cây nến {chart} thiết lập đỉnh ở {i}",
        "Mức giá trần của cây nến {chart} trong phiên là {i}",
        "Cây nến {chart} tăng cao nhất lên đến {i}",
        "Đỉnh điểm của phiên giao dịch cây nến {chart} là {i}"
    ],
    "L": [
        "Cây nến {chart} chạm đáy tại {i}",
        "Mức thấp nhất của cây nến {chart} là {i}",
        "Cây nến {chart} giảm xuống còn {i}",
        "Giá thấp nhất trong phiên của cây nến {chart} là {i}",
        "Cây nến {chart} rơi về vùng giá thấp nhất {i}",
        "Điểm thấp nhất ghi nhận được cho cây nến {chart} là {i}",
        "Cây nến {chart} thoái lui về mức {i}",
        "Trong đợt sụt giảm, cây nến {chart} chạm ngưỡng {i}",
        "Đáy của phiên giao dịch cây nến {chart} nằm tại {i}",
        "Mức giá sàn mà cây nến {chart} đã đi qua là {i}"
    ],
    "C": [
        "Cây nến {chart} chốt phiên tại {i}",
        "Giá đóng cửa của cây nến {chart} là {i}",
        "Kết thúc phiên, cây nến {chart} đứng ở mức {i}",
        "Phiên giao dịch dừng lại với cây nến {chart} giá {i}",
        "Cây nến {chart} kết thúc ngày giao dịch ở {i}",
        "Mức giá đóng cửa cuối cùng của cây nến {chart} là {i}",
        "Sau cùng, cây nến {chart} giữ mức {i}",
        "Giá chốt cuối phiên cho cây nến {chart} ghi nhận {i}",
        "Cây nến {chart} hoàn tất phiên tại ngưỡng {i}",
        "Giao dịch cuối cùng của cây nến {chart} được khớp ở {i}"
    ]
}

class CandleGenerator(BaseGenerator):
    def __init__(self, tokenizer=None, num_samples: int = 1024):
        super().__init__(tokenizer)
        self.num_samples = num_samples
        self._count = 0

    def __next__(self) -> List[Dict]:
        if self._count >= self.num_samples:
            raise StopIteration
        batch = []
        while len(batch) < self.batch_size:
            if self._count >= self.num_samples:
                break

            for candle_type in ["O", "H", "L", "C"]:
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
                price_type = candle.open if candle_type == "O" else candle.high if candle_type == "H" else candle.low if candle_type == "L" else candle.close
                template = random.choice(CANDLESTICK_TEMPLATES[candle_type])
                gen_text = template.format(chart=candle.tag(), i=price_type)
                meta = {
                    "type": "candlestick"
                }
                token_length = len(self.tokenizer.encode(gen_text)) if self.tokenizer else 0
                batch.append({
                    "text": gen_text,
                    "source": "candlestick",
                    "token_length": token_length,
                    "meta": json.dumps(meta, ensure_ascii=False)
                })
            self._count += 1

        return batch