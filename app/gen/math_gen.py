from app.gen.base_gen import BaseGenerator
from typing import List, Dict
import random

MATH_REASONING_TEMPLATES = {
    "sub": [
        "Để tính {a} trừ {b}, ta lấy {a} bớt đi {b} đơn vị. Kết quả cuối cùng là {res}.",
        "Bài toán: {a} - {b}. Suy luận: Bắt đầu từ {a}, lùi lại {b} bước ta dừng ở {res}. Vậy {a} - {b} = {res}."
    ],
    "dist": [
        "Khoảng cách giữa {a} và {b} được tính bằng trị tuyệt đối của hiệu hai số: |{a} - {b}|. Ta có |{a} - {b}| = {res}.",
        "Để tìm khoảng cách từ {a} đến {b}, ta lấy số lớn trừ số bé. Kết quả độ lệch là {res} đơn vị."
    ]
}

class MathGenerator(BaseGenerator):
    def __init__(self, tokenizer=None, num_range: int = 1024):
        super().__init__(tokenizer)
        self.num_range = num_range
        self._current_i = 0

    def __next__(self) -> List[Dict]:
        if self._current_i >= self.num_range:
            raise StopIteration
        records = []
        for _ in range(400):
            op_type = random.choice(["sub", "dist"])
            a, b = random.randint(0, 1023), random.randint(0, 1023)
            res = a - b if op_type == "sub" else abs(a - b)
            template = random.choice(MATH_REASONING_TEMPLATES[op_type])
            gen_text = template.format(a=a, b=b, res=res)
            records.append({
                "text": gen_text,
                "source": "synthetic_math_reasoning",
                "token_length": len(self.tokenizer.encode(gen_text)) if self.tokenizer else 0,
                "meta": f"math_reasoning_{op_type}"
            })
        self._current_i += 1
        return records