
import json

from app.gen.base_gen import BaseGenerator

from typing import List, Dict
import random

BASIC_TEMPLATES_MAP = {
    "O": [
        "<chart>O_{i}</chart> có giá mở cửa là {i}",
        "{i} là mức giá mở cửa của <chart>O_{i}</chart>",
        "Giá mở cửa <chart>O_{i}</chart> ghi nhận tại {i}",
        "Bắt đầu phiên, <chart>O_{i}</chart> đứng ở mức {i}",
        "Phiên giao dịch khởi đầu với <chart>O_{i}</chart> tại {i}",
        "Mức giá đầu tiên của <chart>O_{i}</chart> là {i}",
        "<chart>O_{i}</chart> mở màn tại ngưỡng {i}",
        "Tại thời điểm mở cửa, <chart>O_{i}</chart> có giá {i}",
        "Ghi nhận mức giá mở phiên {i} cho <chart>O_{i}</chart>",
        "<chart>O_{i}</chart> xuất phát ở mức giá {i}"
    ],
    "H": [
        "<chart>H_{i}</chart> có giá cao nhất {i}",
        "Mức cao kỷ lục của <chart>H_{i}</chart> là {i}",
        "<chart>H_{i}</chart> đạt đỉnh tại {i}",
        "Giá cao nhất ghi nhận được cho <chart>H_{i}</chart> là {i}",
        "Trong phiên, <chart>H_{i}</chart> vươn tới mức {i}",
        "Ngưỡng cao nhất mà <chart>H_{i}</chart> đạt được là {i}",
        "<chart>H_{i}</chart> thiết lập đỉnh ở {i}",
        "Mức giá trần của <chart>H_{i}</chart> trong phiên là {i}",
        "<chart>H_{i}</chart> tăng cao nhất lên đến {i}",
        "Đỉnh điểm của phiên giao dịch <chart>H_{i}</chart> là {i}"
    ],
    "L": [
        "<chart>L_{i}</chart> chạm đáy tại {i}",
        "Mức thấp nhất của <chart>L_{i}</chart> là {i}",
        "<chart>L_{i}</chart> giảm xuống còn {i}",
        "Giá thấp nhất trong phiên của <chart>L_{i}</chart> là {i}",
        "<chart>L_{i}</chart> rơi về vùng giá thấp nhất {i}",
        "Điểm thấp nhất ghi nhận được cho <chart>L_{i}</chart> là {i}",
        "<chart>L_{i}</chart> thoái lui về mức {i}",
        "Trong đợt sụt giảm, <chart>L_{i}</chart> chạm ngưỡng {i}",
        "Đáy của phiên giao dịch <chart>L_{i}</chart> nằm tại {i}",
        "Mức giá sàn mà <chart>L_{i}</chart> đã đi qua là {i}"
    ],
    "C": [
        "<chart>C_{i}</chart> chốt phiên tại {i}",
        "Giá đóng cửa của <chart>C_{i}</chart> là {i}",
        "Kết thúc phiên, <chart>C_{i}</chart> đứng ở mức {i}",
        "Phiên giao dịch dừng lại với <chart>C_{i}</chart> giá {i}",
        "<chart>C_{i}</chart> kết thúc ngày giao dịch ở {i}",
        "Mức giá đóng cửa cuối cùng của <chart>C_{i}</chart> là {i}",
        "Sau cùng, <chart>C_{i}</chart> giữ mức {i}",
        "Giá chốt cuối phiên cho <chart>C_{i}</chart> ghi nhận {i}",
        "<chart>C_{i}</chart> hoàn tất phiên tại ngưỡng {i}",
        "Giao dịch cuối cùng của <chart>C_{i}</chart> được khớp ở {i}"
    ]
}

class BasicGenerator(BaseGenerator):
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
            for key in ["O", "H", "L", "C"]:
                template = random.choice(BASIC_TEMPLATES_MAP[key])
                gen_text = template.format(i=self._count)
                token_length = len(self.tokenizer.encode(gen_text)) if self.tokenizer else 0

                meta = {
                    "type": "basic_chart"
                }

                batch.append({
                    "text": gen_text,
                    "source": "trading",
                    "token_length": token_length,
                    "meta": json.dumps(meta, ensure_ascii=False)
                })
            self._count += 1
        return batch