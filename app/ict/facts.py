"""
facts.py — Gom toàn bộ detector thành 1 fact JSON / chart
================================================================
Đây là entrypoint nối Lớp 1-5 lại với nhau: 1 chart -> chạy mọi detector
-> 1 dict fact đầy đủ, dùng làm input cho render.py (sinh 4 dạng mẫu tin,
spec mục 7) ở Giai đoạn 5.

CHƯA gọi is_shift() (chưa triển khai) — để field "shift" rỗng list, tự
động bổ sung khi ict.is_shift() xong logic thật + golden test.
"""

from typing import Dict, Any

from .parser import CandleParser
from .ict import scan_all_swept, grade_fvg
from .structure import is_fvg
from .relations import build_relations


def build_facts(parser: CandleParser, lookback: int = 10) -> Dict[str, Any]:
    """
    Chạy toàn bộ detector trên 1 CandleParser, trả về fact JSON:
        {
            "n_candles": int,
            "swept"    : [ ... list dict từ scan_all_swept ... ],
            "fvg"      : [ ... list dict từ grade_fvg, mỗi nến có FVG ... ],
            "shift"    : [],  # rỗng cho tới khi is_shift() triển khai xong
            "relations": [ ... list dict từ build_relations ... ],
        }

    Đây là JSON fact "đã tính chính xác, KHÔNG được đổi số liệu" nhắc tới
    trong prompt khung cho GPT (spec mục 8) — mọi bước diễn đạt văn xuôi
    sau này PHẢI bám đúng field trong dict này, không suy luận thêm.
    """
    swept_events = scan_all_swept(parser, lookback=lookback)

    fvg_events = []
    for i in range(len(parser)):
        if is_fvg(parser, i) is not None:
            graded = grade_fvg(parser, i)
            if graded:
                fvg_events.append(graded)

    shift_events: list = []   # placeholder — is_shift() chưa triển khai

    all_events = swept_events + fvg_events + shift_events
    relations = build_relations(all_events)

    return {
        "n_candles": len(parser),
        "swept"    : swept_events,
        "fvg"      : fvg_events,
        "shift"    : shift_events,
        "relations": relations,
    }