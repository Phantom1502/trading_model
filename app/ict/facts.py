"""
facts.py — Gom toàn bộ detector thành 1 fact JSON / chart
================================================================
Đây là entrypoint nối Lớp 1-5 lại với nhau: 1 chart -> chạy mọi detector
-> 1 dict fact đầy đủ, dùng làm input cho render.py (sinh 4 dạng mẫu tin,
spec mục 7) ở Giai đoạn 5.
"""

from typing import Dict, Any

from .parser import CandleParser
from .ict import scan_all_swept, grade_fvg, scan_all_shift, SWEPT_LOOKBACK_DEFAULT
from .structure import is_fvg
from .relations import build_relations


def build_facts(
    parser: CandleParser,
    initial_trend: str,
    lookback: int = SWEPT_LOOKBACK_DEFAULT,
) -> Dict[str, Any]:
    """
    Chạy toàn bộ detector trên 1 CandleParser, trả về fact JSON:
        {
            "n_candles": int,
            "swept"    : [ ... list dict từ scan_all_swept ... ],
            "fvg"      : [ ... list dict từ grade_fvg, mỗi nến có FVG ... ],
            "shift"    : [ ... list dict từ scan_all_shift ... ],
            "relations": [ ... list dict từ build_relations ... ],
        }

    `initial_trend`: "BULL" | "BEAR" — trend giả định TẠI NẾN ĐẦU chart,
    PHẢI truyền vào tường minh (vd suy từ HTF bias, hoặc quy ước riêng
    của pipeline gen data) — build_facts() KHÔNG tự đoán trend, vì đây là
    input nghiệp vụ nằm ngoài phạm vi thông tin có trong đúng 20 nến của
    chart hiện tại (xem ict.py::scan_all_shift docstring).

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

    shift_events = scan_all_shift(parser, initial_trend=initial_trend, lookback=lookback)

    all_events = swept_events + fvg_events + shift_events
    relations = build_relations(all_events)

    return {
        "n_candles": len(parser),
        "swept"    : swept_events,
        "fvg"      : fvg_events,
        "shift"    : shift_events,
        "relations": relations,
    }