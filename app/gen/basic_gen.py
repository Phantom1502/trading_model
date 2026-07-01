
from app.gen.base_gen import BaseGenerator

from typing import List, Dict
import random

BASIC_TEMPLATES_MAP = {
    "O": [
        "{tag} có giá mở cửa là {val}",
        "{val} là mức giá mở cửa của {tag}",
        "Giá mở cửa {tag} ghi nhận tại {val}",
        "Bắt đầu phiên, {tag} đứng ở mức {val}",
        "Phiên giao dịch khởi đầu with {tag} tại {val}"
    ],
    "H": [
        "{tag} có giá cao nhất {val}",
        "Mức cao kỷ lục của {tag} là {val}",
        "{tag} đạt đỉnh tại {val}",
        "Trong phiên, {tag} vươn tới mức {val}"
    ],
    "L": [
        "{tag} chạm đáy tại {val}",
        "Mức thấp nhất của {tag} là {val}",
        "{tag} giảm xuống còn {val}",
        "Điểm thấp nhất ghi nhận được cho {tag} là {val}"
    ],
    "C": [
        "{tag} chốt phiên tại {val}",
        "Giá đóng cửa của {tag} là {val}",
        "Kết thúc phiên, {tag} đứng ở mức {val}",
        "Giao dịch cuối cùng của {tag} được khớp ở {val}"
    ]
}

class BasicGenerator(BaseGenerator):
    def __init__(self, tokenizer=None, num_range: int = 1024):
        super().__init__(tokenizer)
        self.num_range = num_range
        self._current_i = 0

    def __next__(self) -> List[Dict]:
        if self._current_i >= self.num_range:
            raise StopIteration
        records = []
        i = self._current_i
        for _ in range(400):
            for key in ["O", "H", "L", "C"]:
                tag = f"<chart>{key}_{i}</chart>"
                template = random.choice(BASIC_TEMPLATES_MAP[key])
                gen_text = template.format(tag=tag, val=i)
                records.append({
                    "text": gen_text,
                    "source": "synthetic_trading_data",
                    "token_length": len(self.tokenizer.encode(gen_text)) if self.tokenizer else 0,
                    "meta": f"type_{key}"
                })
        self._current_i += 1
        return records