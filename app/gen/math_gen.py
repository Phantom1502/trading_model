import json

from app.gen.base_gen import BaseGenerator
from typing import List, Dict
import random

MATH_REASONING_TEMPLATES = {
    "sub": [
        "Để tính {a} trừ {b}, ta lấy {a} bớt đi {b} đơn vị. Kết quả cuối cùng là {res}.",
        "Bài toán: {a} - {b}. Suy luận: Bắt đầu từ {a}, lùi lại {b} bước ta dừng ở {res}. Vậy {a} - {b} = {res}.",
        "Muốn tìm hiệu của {a} và {b}, ta thực hiện phép trừ {a} - {b}. Ta có {a} trừ {b} bằng {res}."
    ],
    "dist": [
        "Khoảng cách giữa {a} và {b} được tính bằng trị tuyệt đối của hiệu hai số: |{a} - {b}|. Ta có |{a} - {b}| = {res}.",
        "Để tìm khoảng cách từ {a} đến {b}, ta lấy số lớn trừ số bé. Kết quả độ lệch là {res} đơn vị.",
        "Nhìn trên trục số, khoảng cách từ điểm {a} đến điểm {b} là {res}."
    ]
}

class MathGenerator(BaseGenerator):
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
            for op_type in ["sub", "dist"]:
                a = self._count
                b = random.randint(0, 20)
                c = a - b if a > b else b - a

                min_num = min(a, c)
                max_num = max(a, c)

                res = max_num - min_num

                template = random.choice(MATH_REASONING_TEMPLATES[op_type])
                gen_text = template.format(a=max_num, b=min_num, res=res)

                meta = {
                    "type": "basic_math"
                }
                token_length = len(self.tokenizer.encode(gen_text)) if self.tokenizer else 0
                batch.append({
                    "text": gen_text,
                    "source": "math",
                    "token_length": token_length,
                    "meta": json.dumps(meta, ensure_ascii=False)
                })
            self._count += 1

        return batch