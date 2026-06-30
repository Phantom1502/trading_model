"""
render.py — STUB, chưa triển khai (Giai đoạn 5 trong lộ trình)
=====================================================================
Dự kiến: nhận fact JSON (từ facts.build_facts) -> render 4 dạng mẫu tin
(Swept / FVG / Shift / Tổng hợp) theo format 4 phần chuẩn (spec mục 5):
    [1. CHART] [2. YÊU CẦU] [3. LÝ GIẢI] [4. CHẤM ĐIỂM]

Phần [3. LÝ GIẢI] cần GPT diễn đạt (spec mục 8) — module này chỉ chịu
trách nhiệm render phần [1], [2], [4] (đều suy ra thẳng từ fact JSON,
không cần GPT) và đóng gói prompt gửi GPT cho phần [3].

Chưa triển khai vì phụ thuộc Giai đoạn 3-4 xong trước (cần is_shift() và
build_relations() đã có golden test, để fact JSON đáng tin trước khi
render thành data training).
"""

from typing import Dict, Any, List


def render_swept_sample(fact: Dict[str, Any], raw_chart_text: str) -> Dict[str, Any]:
    raise NotImplementedError("Giai đoạn 5 chưa triển khai — xem spec mục 10.")


def render_fvg_sample(fact: Dict[str, Any], raw_chart_text: str) -> Dict[str, Any]:
    raise NotImplementedError("Giai đoạn 5 chưa triển khai — xem spec mục 10.")


def render_shift_sample(fact: Dict[str, Any], raw_chart_text: str) -> Dict[str, Any]:
    raise NotImplementedError("Giai đoạn 5 chưa triển khai — xem spec mục 10, cần is_shift() xong trước.")


def render_synthesis_sample(fact: Dict[str, Any], raw_chart_text: str) -> Dict[str, Any]:
    raise NotImplementedError("Giai đoạn 5 chưa triển khai — xem spec mục 10.")


def render_all_samples(fact: Dict[str, Any], raw_chart_text: str) -> List[Dict[str, Any]]:
    """Render cả 4 dạng từ 1 fact JSON (spec mục 7) — gọi 4 hàm trên."""
    raise NotImplementedError("Giai đoạn 5 chưa triển khai — xem spec mục 10.")