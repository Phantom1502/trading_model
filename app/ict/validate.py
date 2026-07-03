"""
validate.py — Validate sample đã render (Giai đoạn 5)
================================================================
GIẢN LƯỢC so với thiết kế ban đầu (spec mục 8): vì render.py không còn
dùng GPT (template engine deterministic, số liệu chèn trực tiếp từ fact
dict), KHÔNG CẦN bước "đối chiếu số liệu với GPT output" nữa — số liệu
đúng theo construction, đã được test trong test_render.py.

2 nhóm validate còn giữ lại — vẫn CÓ THỂ xảy ra do bug trong chính
template/logic render (không phải do GPT):

    validate_cross_consistency(samples_same_chart)
        Đối chiếu field giữa mẫu đơn (Swept/FVG/Shift) và mẫu Tổng hợp
        PHẢI khớp 100% CHO CÙNG 1 EVENT. Định danh 1 event bằng cặp
        (TYPE, CANDLE) — nếu event đó xuất hiện ở nhiều mẫu, mọi field
        chung (SWING_LEVEL, DEPTH, GAP_SIZE...) phải giống hệt nhau.

    validate_no_leakage(sample)
        Xóa phần [3. LÝ GIẢI], thử tái dựng câu chuyện CHỈ từ phần
        [4. CHẤM ĐIỂM] — nếu không tái dựng được, Eval đang thiếu field
        mà Lý giải có nhắc tới.
"""

import re
from collections import defaultdict
from typing import Dict, Any, List, Optional


# Regex bắt toàn bộ KEY=VALUE trong 1 block <eval>...</eval>
_EVAL_FIELD_RE = re.compile(r"(\w+)=([\w.\-]+)")

# Regex trích số — CHỈ bắt phần thập phân khi có ÍT NHẤT 1 chữ số sau dấu
# chấm (\.?\d* cũ từng nuốt nhầm dấu chấm cuối câu, vd "...nến 10." bị
# bắt thành "10." thay vì "10" — đã fix bằng (?:\.\d+)? thay vì \.?\d*).
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


def _parse_eval(eval_block: str) -> Dict[str, str]:
    """Parse <eval>KEY=VAL KEY2=VAL2...</eval> thành dict phẳng (KHÔNG tách theo event)."""
    inner = eval_block.strip()
    if inner.startswith("<eval>"):
        inner = inner[len("<eval>"):]
    if inner.endswith("</eval>"):
        inner = inner[: -len("</eval>")]
    return dict(_EVAL_FIELD_RE.findall(inner))


def _parse_eval_events(eval_block: str) -> List[Dict[str, str]]:
    """
    Parse eval thành LIST các dict, mỗi dict = 1 event riêng biệt.

    N==1 (không có tiền tố EVENTk_) -> list 1 phần tử.
    N>1 (có tiền tố EVENT1_/EVENT2_/...) -> tách theo từng số thứ tự,
    field "SEQUENCE" (nếu có, chỉ xuất hiện ở mẫu Tổng hợp N>=2) KHÔNG
    phải 1 event, tự động bị loại khỏi kết quả.
    """
    flat = _parse_eval(eval_block)
    grouped: Dict[int, Dict[str, str]] = defaultdict(dict)
    unprefixed: Dict[str, str] = {}

    for k, v in flat.items():
        m = re.match(r"^EVENT(\d+)_(.+)$", k)
        if m:
            grouped[int(m.group(1))][m.group(2)] = v
        elif k != "SEQUENCE":
            unprefixed[k] = v

    if grouped:
        return [grouped[i] for i in sorted(grouped.keys())]
    return [unprefixed] if unprefixed else []


def _event_identity(event: Dict[str, str]) -> Optional[tuple]:
    """Định danh 1 event bằng (TYPE, CANDLE) — None nếu thiếu 1 trong 2 field."""
    if "TYPE" in event and "CANDLE" in event:
        return (event["TYPE"], event["CANDLE"])
    return None


def validate_cross_consistency(samples_same_chart: List[Dict[str, Any]]) -> bool:
    """
    Đối chiếu field giữa các mẫu ĐƠN (Swept/FVG/Shift) và mẫu TỔNG HỢP
    render từ CÙNG 1 fact JSON — CÙNG 1 EVENT (định danh bằng TYPE+CANDLE)
    xuất hiện ở nhiều mẫu thì mọi field chung phải khớp giá trị.

    `samples_same_chart`: list dict trả về từ render_*_sample(), PHẢI cùng
    render từ 1 fact JSON (caller đảm bảo, hàm không tự kiểm tra điều đó).

    Returns True nếu nhất quán, False nếu phát hiện event trùng định danh
    (TYPE+CANDLE) nhưng field khác lệch giá trị giữa 2 mẫu bất kỳ.
    """
    if len(samples_same_chart) < 2:
        return True

    by_identity: Dict[tuple, List[Dict[str, str]]] = defaultdict(list)
    for sample in samples_same_chart:
        for event in _parse_eval_events(sample["eval"]):
            identity = _event_identity(event)
            if identity is not None:
                by_identity[identity].append(event)

    for identity, events in by_identity.items():
        if len(events) < 2:
            continue
        reference = events[0]
        for other in events[1:]:
            shared_keys = set(reference) & set(other)
            for k in shared_keys:
                if reference[k] != other[k]:
                    return False

    return True


def validate_no_leakage(sample: Dict[str, Any]) -> bool:
    """
    Kiểm tra mọi SỐ xuất hiện trong phần [3. LÝ GIẢI] đều truy được nguồn
    gốc từ phần [4. CHẤM ĐIỂM] — nếu Lý giải nhắc tới 1 con số không có
    trong Eval, đó là dấu hiệu template thêm thắt chi tiết không kiểm
    soát được.
    """
    explanation_numbers = set(_NUMBER_RE.findall(sample["explanation"]))
    eval_numbers = set(_NUMBER_RE.findall(sample["eval"]))

    missing = explanation_numbers - eval_numbers
    return len(missing) == 0