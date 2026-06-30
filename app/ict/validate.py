"""
validate.py — STUB, chưa triển khai (Giai đoạn 5 trong lộ trình)
=======================================================================
Dự kiến 3 nhóm validate, đúng nguyên tắc spec mục 8-9:

    validate_numbers_match(sample, fact)
        Đối chiếu số liệu trong câu GPT sinh (phần Lý giải) khớp đúng
        fact JSON gốc — bắt case GPT tự đổi số liệu khi diễn đạt.

    validate_cross_consistency(samples_same_chart)
        Đối chiếu field giữa mẫu đơn (Swept/FVG/Shift) và mẫu Tổng hợp
        PHẢI khớp 100% — lệch nghĩa là lỗi ở bước generate.

    validate_no_leakage(sample)
        Xóa phần [3. LÝ GIẢI], thử tái dựng câu chuyện CHỈ từ phần
        [4. CHẤM ĐIỂM] — nếu không tái dựng được, Eval đang thiếu field
        mà Lý giải có nhắc tới (dấu hiệu GPT tự thêm thắt).

Chưa triển khai vì phụ thuộc render.py xong trước (cần có sample thật để
validate).
"""

from typing import Dict, Any, List


def validate_numbers_match(sample: Dict[str, Any], fact: Dict[str, Any]) -> bool:
    raise NotImplementedError("Giai đoạn 5 chưa triển khai — xem spec mục 10.")


def validate_cross_consistency(samples_same_chart: List[Dict[str, Any]]) -> bool:
    raise NotImplementedError("Giai đoạn 5 chưa triển khai — xem spec mục 10.")


def validate_no_leakage(sample: Dict[str, Any]) -> bool:
    raise NotImplementedError("Giai đoạn 5 chưa triển khai — xem spec mục 10.")