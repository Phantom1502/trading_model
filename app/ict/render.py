"""
render.py — Template engine: fact JSON -> 4 dạng mẫu tin (Giai đoạn 5)
==============================================================================
KHÔNG dùng GPT (quyết định đã đổi, xem spec mục 8) — toàn bộ văn xuôi sinh
ra từ template cố định + ngân hàng biến thể (random.choice), số liệu chèn
trực tiếp từ fact dict qua string interpolation. 100% deterministic, không
có khả năng hallucinate số liệu.

Format 4 phần chuẩn (spec mục 5):
    [1. CHART]      raw_chart_text, giữ nguyên, không đổi
    [2. YÊU CẦU]     chọn ngẫu nhiên 1 trong nhiều cách diễn đạt câu hỏi
    [3. LÝ GIẢI]      văn xuôi từ template, co giãn độ dài theo số sự kiện
    [4. CHẤM ĐIỂM]     block <eval>...</eval>, KEY=VALUE, EVENT1_/EVENT2_ nếu >1 event

QUY ƯỚC ĐÁNH SỐ NẾN: 1-based (nến đầu tiên = "nến 1"), khớp convention đã
dùng trong candle_parser.py (`ordinal = i + 1`). Index trong fact dict
(swept_candle_idx, fvg_candle_idx, shift_candle_idx, swing_idx) đều là
0-based (index Python thật) — PHẢI +1 khi hiển thị trong text/eval.

GIỚI HẠN PHẠM VI (v1, CHƯA quyết định, không tự ý làm):
    - Chart có 0 event (không phát hiện Swept/FVG/Shift nào) -> HIỆN TẠI
      SKIP hoàn toàn, không sinh mẫu "không tìm thấy pattern". Đây có thể
      là data âm (negative example) hữu ích cho training, nhưng là quyết
      định domain cần bàn riêng trước khi thêm — không tự ý sinh loại mẫu
      này.
    - Định dạng chuỗi SEQUENCE trong eval của mẫu Tổng hợp là lựa chọn
      implementation của module này (không phải hằng số cứng từ spec) —
      xem docstring _build_sequence_field() để biết chi tiết, có thể đổi
      nếu cần format khác dễ parse hơn.
"""

import random
from typing import Dict, Any, List, Optional


# ══════════════════════════════════════════════════════════════════════
# NGÂN HÀNG TEMPLATE — mỗi loại event có "full" (<=2 event/mẫu) và
# "short" (>=3 event/mẫu), theo đúng nguyên tắc co giãn độ dài spec mục 5.
# ══════════════════════════════════════════════════════════════════════

_TEMPLATES = {
    "SWEEP_HIGH": {
        "full": [
            "Nến thứ {c} quét qua đỉnh cũ được thiết lập tại nến thứ {s} (mức {level}), vượt lên {depth} bin trước khi bị từ chối — dấu hiệu Liquidity Sweep phía trên.",
            "Tại nến thứ {c}, giá xuyên qua vùng đỉnh ở nến thứ {s} (mức {level}) với độ sâu {depth} bin, cho thấy thanh khoản phía trên đã bị quét.",
            "Đỉnh cũ tại nến thứ {s} (mức {level}) bị nến thứ {c} phá vỡ tạm thời, vượt {depth} bin — một Sweep High điển hình.",
        ],
        "short": [
            "Nến {c} quét đỉnh nến {s} (mức {level}, sâu {depth} bin).",
            "Sweep High tại nến {c}, quét đỉnh cũ ở nến {s}.",
        ],
    },
    "SWEEP_LOW": {
        "full": [
            "Nến thứ {c} quét qua đáy cũ được thiết lập tại nến thứ {s} (mức {level}), xuyên xuống {depth} bin trước khi bị từ chối — dấu hiệu Liquidity Sweep phía dưới.",
            "Tại nến thứ {c}, giá xuyên qua vùng đáy ở nến thứ {s} (mức {level}) với độ sâu {depth} bin, cho thấy thanh khoản phía dưới đã bị quét.",
            "Đáy cũ tại nến thứ {s} (mức {level}) bị nến thứ {c} phá vỡ tạm thời, xuyên xuống {depth} bin — một Sweep Low điển hình.",
        ],
        "short": [
            "Nến {c} quét đáy nến {s} (mức {level}, sâu {depth} bin).",
            "Sweep Low tại nến {c}, quét đáy cũ ở nến {s}.",
        ],
    },
    "BULL": {   # FVG Bullish
        "full": [
            "Từ nến thứ {c} hình thành khoảng trống giá tăng (Bullish FVG), rộng {size} bin, hiện đã lấp {fill}%.",
            "Nến thứ {c} hoàn thiện Fair Value Gap tăng với kích thước {size} bin — mức lấp hiện tại là {fill}%.",
            "Khoảng trống giá tăng hình thành tại nến thứ {c}, độ rộng {size} bin, đã được lấp {fill}% tính đến thời điểm hiện tại.",
        ],
        "short": [
            "FVG tăng tại nến {c} ({size} bin, lấp {fill}%).",
            "Nến {c}: Bullish FVG {size} bin, lấp {fill}%.",
        ],
    },
    "BEAR": {   # FVG Bearish
        "full": [
            "Từ nến thứ {c} hình thành khoảng trống giá giảm (Bearish FVG), rộng {size} bin, hiện đã lấp {fill}%.",
            "Nến thứ {c} hoàn thiện Fair Value Gap giảm với kích thước {size} bin — mức lấp hiện tại là {fill}%.",
            "Khoảng trống giá giảm hình thành tại nến thứ {c}, độ rộng {size} bin, đã được lấp {fill}% tính đến thời điểm hiện tại.",
        ],
        "short": [
            "FVG giảm tại nến {c} ({size} bin, lấp {fill}%).",
            "Nến {c}: Bearish FVG {size} bin, lấp {fill}%.",
        ],
    },
    "BOS": {
        "full": [
            "Nến thứ {c} đóng cửa vượt qua mức {level} (đỉnh/đáy tại nến thứ {s}), xác nhận cấu trúc tiếp diễn theo hướng {dir} — Break of Structure.",
            "Tại nến thứ {c}, giá đóng cửa phá vỡ mốc {level} thiết lập ở nến thứ {s} — BOS, tiếp diễn hướng {dir}.",
        ],
        "short": [
            "BOS tại nến {c}, tiếp diễn hướng {dir}.",
            "Nến {c}: phá cấu trúc tiếp diễn ({dir}), mốc {level} tại nến {s}.",
        ],
    },
    "CHoCH": {
        "full": [
            "Nến thứ {c} đóng cửa vượt qua mức {level} (đỉnh/đáy tại nến thứ {s}), đánh dấu sự thay đổi cấu trúc theo hướng {dir} — Change of Character.",
            "Tại nến thứ {c}, giá đóng cửa phá vỡ mốc {level} thiết lập ở nến thứ {s} — CHoCH, đảo chiều sang hướng {dir}.",
        ],
        "short": [
            "CHoCH tại nến {c}, đảo chiều sang {dir}.",
            "Nến {c}: đổi cấu trúc ({dir}), mốc {level} tại nến {s}.",
        ],
    },
}

_REQUEST_TEMPLATES = {
    "swept": [
        "Phân tích Liquidity Sweep trong chart này.",
        "Xác định các vùng thanh khoản đã bị quét trong đoạn nến trên.",
        "Chart này có Sweep nào không? Nếu có, mô tả chi tiết.",
    ],
    "fvg": [
        "Phân tích Fair Value Gap trong chart này.",
        "Xác định các khoảng trống giá (FVG) xuất hiện trong đoạn nến trên.",
        "Chart này có Fair Value Gap nào không? Nếu có, mô tả chi tiết.",
    ],
    "shift": [
        "Phân tích Market Structure Shift trong chart này.",
        "Xác định các điểm phá vỡ cấu trúc (BOS/CHoCH) trong đoạn nến trên.",
        "Chart này có Shift nào không? Nếu có, mô tả chi tiết.",
    ],
    "synthesis": [
        "Phân tích toàn bộ setup trong chart này.",
        "Đánh giá tổng thể các yếu tố ICT xuất hiện trong đoạn nến trên.",
        "Mô tả đầy đủ cấu trúc thị trường thể hiện trong chart này.",
    ],
}


# ══════════════════════════════════════════════════════════════════════
# HELPER — field extraction (0-based -> 1-based), KHÔNG qua GPT
# ══════════════════════════════════════════════════════════════════════

def _n(idx0: int) -> int:
    """0-based index (trong fact dict) -> 1-based (hiển thị trong text/eval)."""
    return idx0 + 1


def _swept_fields(e: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "TYPE"       : e["type"],
        "CANDLE"     : _n(e["swept_candle_idx"]),
        "SWING_CANDLE": _n(e["swing_idx"]),
        "SWING_LEVEL": e["swing_level"],
        "DEPTH"      : e["depth"],
    }


def _fvg_fields(e: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "TYPE"    : e["type"],
        "CANDLE"  : _n(e["fvg_candle_idx"]),
        "GAP_LOW" : e["gap_low"],
        "GAP_HIGH": e["gap_high"],
        "GAP_SIZE": e["gap_size_bins"],
        "FILL_PCT": e["fill_pct"],
    }


def _shift_fields(e: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "TYPE"        : e["type"],
        "DIRECTION"   : e["direction"],
        "CANDLE"      : _n(e["shift_candle_idx"]),
        "SWING_CANDLE": _n(e["swing_idx"]),
        "SWING_LEVEL" : e["swing_level"],
        "BROKEN"      : e["broken_type"],
    }


def _event_kind_for_template(e: Dict[str, Any]) -> str:
    """Khóa để tra _TEMPLATES — Swept dùng type SWEEP_HIGH/LOW, FVG dùng
    BULL/BEAR trực tiếp, Shift dùng BOS/CHoCH."""
    return e["type"]


def _sentence_for_event(e: Dict[str, Any], use_short: bool, rng: random.Random) -> str:
    """Chọn ngẫu nhiên 1 template full/short phù hợp, điền số liệu từ event dict."""
    kind = _event_kind_for_template(e)
    bank = _TEMPLATES[kind]["short" if use_short else "full"]
    template = rng.choice(bank)

    if "swept_candle_idx" in e:
        f = _swept_fields(e)
        return template.format(c=f["CANDLE"], s=f["SWING_CANDLE"], level=f["SWING_LEVEL"], depth=f["DEPTH"])
    if "fvg_candle_idx" in e:
        f = _fvg_fields(e)
        return template.format(c=f["CANDLE"], size=f["GAP_SIZE"], fill=f["FILL_PCT"])
    if "shift_candle_idx" in e:
        f = _shift_fields(e)
        return template.format(c=f["CANDLE"], s=f["SWING_CANDLE"], level=f["SWING_LEVEL"], dir=f["DIRECTION"])
    raise KeyError(f"Event không xác định được loại: {e}")


def _fields_for_event(e: Dict[str, Any]) -> Dict[str, Any]:
    if "swept_candle_idx" in e:
        return _swept_fields(e)
    if "fvg_candle_idx" in e:
        return _fvg_fields(e)
    if "shift_candle_idx" in e:
        return _shift_fields(e)
    raise KeyError(f"Event không xác định được loại: {e}")


def _build_eval_block(events: List[Dict[str, Any]]) -> str:
    """
    Ghép field của N event thành block <eval>...</eval>, KEY=VALUE cách
    nhau bằng space (spec mục 5). N==1 -> field KHÔNG đánh số (TYPE=...).
    N>1 -> đánh số EVENT1_/EVENT2_/... theo đúng thứ tự trong `events`.
    """
    if len(events) == 1:
        fields = _fields_for_event(events[0])
        parts = [f"{k}={v}" for k, v in fields.items()]
    else:
        parts = []
        for i, e in enumerate(events, start=1):
            fields = _fields_for_event(e)
            parts.extend(f"EVENT{i}_{k}={v}" for k, v in fields.items())
    return "<eval>" + " ".join(parts) + "</eval>"


def _raw_candle_idx(e: Dict[str, Any]) -> int:
    """Lấy candle_idx GỐC (0-based) của 1 event — dùng để sắp xếp theo thời
    gian, KHÔNG dùng field CANDLE đã +1 (chỉ để hiển thị)."""
    for key in ("swept_candle_idx", "fvg_candle_idx", "shift_candle_idx"):
        if key in e:
            return e[key]
    raise KeyError(f"Event không xác định được candle_idx: {e}")


def _build_sequence_field(events: List[Dict[str, Any]], relations: List[Dict[str, Any]]) -> Optional[str]:
    """
    Mã hoá "chuỗi trigger sự kiện" — CHỈ cặp LIỀN KỀ theo thời gian thực tế
    (candle_idx), KHÔNG PHẢI toàn bộ C(n,2) cặp tổ hợp. Với N event, tạo ra
    N-1 cặp (event 1 -> event 2 -> event 3 -> ...), giảm từ O(n²) xuống
    O(n) — quan trọng vì SEQUENCE kiểu tổ hợp đầy đủ vừa tốn token vô ích
    (nhiều cặp event cách xa nhau, không liên quan trực tiếp), vừa không
    đúng ý nghĩa "chuỗi trigger" (điều gì xảy ra NGAY SAU điều gì).

    Nếu <2 event, không có cặp nào, trả về None.

    ĐỊNH DẠNG (lựa chọn implementation, không phải hằng số cứng từ spec):
        "1<2,2~3,3<4"
    Số là thứ tự event (1-based, khớp EVENT1/EVENT2 trong eval — GIỮ
    NGUYÊN thứ tự gốc trong `events`, KHÔNG sắp xếp lại), "<" nghĩa là bên
    trái xảy ra TRƯỚC, "~" là SAME_CANDLE (chưa quyết định, vd có FVG tham
    gia — xem relations.py).
    """
    if len(events) < 2:
        return None

    # Sắp xếp CHỈ SỐ GỐC (0-based trong `events`) theo candle_idx thời gian
    # thực — dùng sort ỔN ĐỊNH (stable) để 2 event cùng candle_idx giữ
    # nguyên thứ tự xuất hiện gốc trong `events` (Swept trước FVG trước
    # Shift, do facts.py ghép all_events = swept+fvg+shift) — TRÙNG HỢP
    # khớp đúng rule đã quyết định (Swept trước Shift), nhưng case có FVG
    # tham gia tie vẫn chỉ là artifact thứ tự list, KHÔNG PHẢI quyết định
    # có chủ đích (xem relations.py để biết rule thật đã chốt phần nào).
    time_order = sorted(range(len(events)), key=lambda i: _raw_candle_idx(events[i]))

    # Tra cứu nhanh order/tie giữa 2 chỉ số gốc bất kỳ từ relations đã có
    relation_lookup = {}
    for r in relations:
        relation_lookup[frozenset((r["event_a_idx"], r["event_b_idx"]))] = r

    parts = []
    for k in range(len(time_order) - 1):
        orig_a, orig_b = time_order[k], time_order[k + 1]
        r = relation_lookup.get(frozenset((orig_a, orig_b)))
        a_num, b_num = orig_a + 1, orig_b + 1   # 1-based khớp EVENT1/EVENT2

        if r is None:
            # Không tìm thấy relation tương ứng (không nên xảy ra nếu
            # `relations` được tính đúng từ chính `events`) — fallback an
            # toàn theo thứ tự candle_idx đã sort, không suy đoán tie.
            parts.append(f"{a_num}<{b_num}")
        elif r["order"] == "A_BEFORE_B":
            parts.append(f"{a_num}<{b_num}" if r["event_a_idx"] == orig_a else f"{b_num}<{a_num}")
        elif r["order"] == "B_BEFORE_A":
            parts.append(f"{b_num}<{a_num}" if r["event_a_idx"] == orig_a else f"{a_num}<{b_num}")
        else:
            parts.append(f"{a_num}~{b_num}")

    return "SEQUENCE=" + ",".join(parts) if parts else None


# ══════════════════════════════════════════════════════════════════════
# RENDER — 4 dạng mẫu tin
# ══════════════════════════════════════════════════════════════════════

def _render_single_type(
    events: List[Dict[str, Any]],
    request_key: str,
    raw_chart_text: str,
    rng: random.Random,
) -> Optional[Dict[str, Any]]:
    """Dùng chung cho Swept/FVG/Shift — chỉ khác request_key và events list."""
    if not events:
        return None   # v1: SKIP hoàn toàn khi không có event loại này (xem docstring module)

    use_short = len(events) >= 3
    sentences = [_sentence_for_event(e, use_short, rng) for e in events]
    explanation = " ".join(sentences)

    request = rng.choice(_REQUEST_TEMPLATES[request_key])
    eval_block = _build_eval_block(events)

    text = f"{raw_chart_text}\n{request}\n{explanation}\n{eval_block}"

    return {
        "chart"      : raw_chart_text,
        "request"    : request,
        "explanation": explanation,
        "eval"       : eval_block,
        "text"       : text,
        "event_count": len(events),
    }


def render_swept_sample(fact: Dict[str, Any], raw_chart_text: str, rng: Optional[random.Random] = None) -> Optional[Dict[str, Any]]:
    """Dạng 1 (spec mục 7): chỉ nói về Swept, bỏ qua FVG/Shift dù có xuất hiện."""
    rng = rng or random
    return _render_single_type(fact["swept"], "swept", raw_chart_text, rng)


def render_fvg_sample(fact: Dict[str, Any], raw_chart_text: str, rng: Optional[random.Random] = None) -> Optional[Dict[str, Any]]:
    """Dạng 2: chỉ nói về FVG."""
    rng = rng or random
    return _render_single_type(fact["fvg"], "fvg", raw_chart_text, rng)


def render_shift_sample(fact: Dict[str, Any], raw_chart_text: str, rng: Optional[random.Random] = None) -> Optional[Dict[str, Any]]:
    """Dạng 3: chỉ nói về Shift."""
    rng = rng or random
    return _render_single_type(fact["shift"], "shift", raw_chart_text, rng)


def render_synthesis_sample(fact: Dict[str, Any], raw_chart_text: str, rng: Optional[random.Random] = None) -> Optional[Dict[str, Any]]:
    """
    Dạng 4: nói về CẢ 3 loại + quan hệ thứ tự (Tầng 2, field SEQUENCE).

    Thứ tự event trong `events` khớp ĐÚNG cách facts.build_facts() ghép
    all_events (swept + fvg + shift) -> đảm bảo EVENT1/EVENT2/... trong
    eval khớp đúng event_a_idx/event_b_idx trong facts["relations"].
    """
    rng = rng or random
    events = fact["swept"] + fact["fvg"] + fact["shift"]
    if not events:
        return None   # v1: SKIP khi chart hoàn toàn không có event nào (xem docstring module)

    use_short = len(events) >= 3
    sentences = [_sentence_for_event(e, use_short, rng) for e in events]
    explanation = " ".join(sentences)

    request = rng.choice(_REQUEST_TEMPLATES["synthesis"])
    eval_block = _build_eval_block(events)

    sequence_field = _build_sequence_field(events, fact["relations"])
    if sequence_field:
        # Chèn SEQUENCE vào trong block <eval>...</eval>, trước dấu đóng
        eval_block = eval_block[:-len("</eval>")] + " " + sequence_field + "</eval>"

    text = f"{raw_chart_text}\n{request}\n{explanation}\n{eval_block}"

    return {
        "chart"      : raw_chart_text,
        "request"    : request,
        "explanation": explanation,
        "eval"       : eval_block,
        "text"       : text,
        "event_count": len(events),
    }


def render_all_samples(fact: Dict[str, Any], raw_chart_text: str, rng: Optional[random.Random] = None) -> List[Dict[str, Any]]:
    """
    Render cả 4 dạng từ 1 fact JSON (spec mục 7). CHỈ trả về những dạng có
    ít nhất 1 event (v1: bỏ qua dạng "không tìm thấy pattern", xem docstring
    module) — số phần tử trả về có thể từ 0 đến 4.
    """
    rng = rng or random
    samples = [
        render_swept_sample(fact, raw_chart_text, rng),
        render_fvg_sample(fact, raw_chart_text, rng),
        render_shift_sample(fact, raw_chart_text, rng),
        render_synthesis_sample(fact, raw_chart_text, rng),
    ]
    return [s for s in samples if s is not None]