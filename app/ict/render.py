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
    [4. CHẤM ĐIỂM]     block <eval>...</eval>, KEY=VALUE, E1_/E2_ nếu >1 event

QUY ƯỚC ĐÁNH SỐ NẾN: 1-based (nến đầu tiên = "nến 1"), khớp convention đã
dùng trong candle_parser.py (`ordinal = i + 1`). Index trong fact dict
(swept_candle_idx, fvg_candle_idx, shift_candle_idx, swing_idx) đều là
0-based (index Python thật) — PHẢI +1 khi hiển thị trong text/eval.

BẢNG VIẾT TẮT KEY TRONG <eval> (đã rút gọn để giảm token — tokenizer BPE
nhỏ (~16k vocab, chủ yếu train tiếng Việt/code) không có sẵn token nguyên
khối cho chuỗi ALLCAPS_GẠCH_DƯỚI hiếm gặp, phải tách nhỏ nhiều mảnh, cộng
dồn đáng kể qua nhiều field/event. Xem README.md "Case study 3" để biết
số liệu thật đã dẫn tới quyết định này):

    Tiền tố nhiều event : EVENT -> E       (vd E1_, E2_)
    TYPE                : T
    CANDLE              : C
    SWING_CANDLE        : SC
    SWING_LEVEL         : SL
    DEPTH               : D
    GAP_LOW             : GL
    GAP_HIGH            : GH
    GAP_SIZE            : GS
    FILL_PCT            : FP
    DIRECTION           : DIR
    BROKEN              : BR

GIỚI HẠN PHẠM VI (v1, CHƯA quyết định, không tự ý làm):
    - Chart có 0 event (không phát hiện Swept/FVG/Shift nào) -> HIỆN TẠI
      SKIP hoàn toàn, không sinh mẫu "không tìm thấy pattern". Đây có thể
      là data âm (negative example) hữu ích cho training, nhưng là quyết
      định domain cần bàn riêng trước khi thêm — không tự ý sinh loại mẫu
      này.

MẪU TỔNG HỢP — KHÔNG còn field SEQUENCE (đã bỏ, xem quyết định dưới):
    Trước đây có field SEQUENCE riêng mã hoá quan hệ thứ tự giữa các event.
    ĐÃ BỎ vì dư thừa: field C (CANDLE) đã có sẵn trong TỪNG event, model có
    thể tự so sánh 2 giá trị C để suy ra thứ tự thời gian mà không cần 1
    field riêng liệt kê lại quan hệ đó — SEQUENCE chỉ thêm token mà không
    thêm thông tin thật sự mới, ngoại trừ rule tie-break "Swept trước
    Shift" khi trùng candle, và điều đó giờ được truyền đạt đơn giản hơn:
    mẫu Tổng hợp SẮP XẾP event theo đúng thứ tự THỜI GIAN (candle_idx) khi
    hiển thị (sort ổn định — 2 event cùng candle giữ nguyên thứ tự
    Swept/FVG/Shift đã quyết định), nên thứ tự xuất hiện trong text tự nó
    đã là tín hiệu thứ tự, không cần field phụ.

    Hệ quả: render.py KHÔNG CÒN cần `facts["relations"]` cho bất kỳ việc
    gì — toàn bộ logic remap index (từng cần để giữ SEQUENCE đúng sau khi
    lọc top-K) cũng biến mất theo, code đơn giản hẳn.

TOP-K CHO FVG (đã quyết định, dựa trên dữ liệu thật):
    Validate quy mô lớn (5.7M dòng, cửa sổ 20 nến) cho thấy mật độ FVG
    (~25%/nến) đủ cao để MỘT MÌNH giới hạn cửa sổ 20 nến KHÔNG đủ tránh
    vượt max_seq=512 — mẫu FVG đơn lẻ đã vượt ngân sách ngay ở p50 (536
    token), mẫu Tổng hợp vượt còn nặng hơn (87.99% > 512). Swept/Shift
    KHÔNG cần giới hạn (0% vượt ngân sách theo cùng thống kê).

    FVG_TOP_K giới hạn số FVG event hiển thị trong 1 mẫu — CHỌN theo kết
    hợp 2 tiêu chí (không phải chỉ magnitude thuần):
        1. Gần vùng giá HIỆN TẠI (Close của nến cuối cửa sổ)
        2. Gần về THỜI GIAN (candle_idx gần cuối cửa sổ hơn)
    Kết hợp bằng RANK (không phải giá trị thô, vì 2 đại lượng khác đơn vị)
    — xem _select_top_k_fvg() để biết chi tiết thuật toán.

    K=4 là giá trị mặc định BAN ĐẦU — đã xác nhận qua validate thật: FVG
    đơn lẻ giờ 0% vượt max_seq (p50=450). Mẫu Tổng hợp VẪN còn vượt nhiều
    (81.98% > 512, do cộng thêm Swept/Shift không lọc + field verbosity) —
    đây là lý do trực tiếp dẫn tới quyết định bỏ SEQUENCE + rút gọn key ở
    trên, xem README.md "Case study 3".
"""

import random
from typing import Dict, Any, List, Optional

from .candle import parse_candles


# FVG_TOP_K — xem docstring module "TOP-K CHO FVG" để biết lý do/căn cứ.
FVG_TOP_K = 4


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
# Key đã RÚT GỌN — xem bảng viết tắt ở docstring module.
# ══════════════════════════════════════════════════════════════════════

def _n(idx0: int) -> int:
    """0-based index (trong fact dict) -> 1-based (hiển thị trong text/eval)."""
    return idx0 + 1


def _swept_fields(e: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "T" : e["type"],
        "C" : _n(e["swept_candle_idx"]),
        "SC": _n(e["swing_idx"]),
        "SL": e["swing_level"],
        "D" : e["depth"],
    }


def _fvg_fields(e: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "T" : e["type"],
        "C" : _n(e["fvg_candle_idx"]),
        "GL": e["gap_low"],
        "GH": e["gap_high"],
        "GS": e["gap_size_bins"],
        "FP": e["fill_pct"],
    }


def _shift_fields(e: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "T"  : e["type"],
        "DIR": e["direction"],
        "C"  : _n(e["shift_candle_idx"]),
        "SC" : _n(e["swing_idx"]),
        "SL" : e["swing_level"],
        "BR" : e["broken_type"],
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
        return template.format(c=f["C"], s=f["SC"], level=f["SL"], depth=f["D"])
    if "fvg_candle_idx" in e:
        f = _fvg_fields(e)
        return template.format(c=f["C"], size=f["GS"], fill=f["FP"])
    if "shift_candle_idx" in e:
        f = _shift_fields(e)
        return template.format(c=f["C"], s=f["SC"], level=f["SL"], dir=f["DIR"])
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
    nhau bằng space (spec mục 5). N==1 -> field KHÔNG đánh số (T=...).
    N>1 -> đánh số E1_/E2_/... theo đúng thứ tự trong `events`.
    """
    if len(events) == 1:
        fields = _fields_for_event(events[0])
        parts = [f"{k}={v}" for k, v in fields.items()]
    else:
        parts = []
        for i, e in enumerate(events, start=1):
            fields = _fields_for_event(e)
            parts.extend(f"E{i}_{k}={v}" for k, v in fields.items())
    return "<eval>" + " ".join(parts) + "</eval>"


def _raw_candle_idx(e: Dict[str, Any]) -> int:
    """Lấy candle_idx GỐC (0-based) của 1 event — dùng để sắp xếp theo thời
    gian, KHÔNG dùng field C đã +1 (chỉ để hiển thị)."""
    for key in ("swept_candle_idx", "fvg_candle_idx", "shift_candle_idx"):
        if key in e:
            return e[key]
    raise KeyError(f"Event không xác định được candle_idx: {e}")


# ══════════════════════════════════════════════════════════════════════
# TOP-K CHO FVG — xem docstring module để biết căn cứ dữ liệu thật
# ══════════════════════════════════════════════════════════════════════

def _current_price_and_n(raw_chart_text: str) -> tuple:
    """Trích Close của nến CUỐI CÙNG (đại diện 'giá hiện tại') + tổng số
    nến trong chart, bằng cách parse lại raw_chart_text (không cần truyền
    thêm parser/candles vào mọi hàm render, giữ API ổn định)."""
    candles = parse_candles(raw_chart_text)
    if not candles:
        return None, 0
    return candles[-1].close, len(candles)


def _select_top_k_fvg(
    fvg_events: List[Dict[str, Any]],
    raw_chart_text: str,
    k: int = FVG_TOP_K,
) -> tuple:
    """
    Chọn K FVG event ĐÁNG CHÚ Ý NHẤT theo 2 tiêu chí kết hợp:
        1. Gần vùng giá hiện tại (khoảng cách |trung điểm gap - Close nến cuối|)
        2. Gần về thời gian (candle_idx gần cuối cửa sổ hơn)

    Kết hợp bằng RANK (mỗi tiêu chí xếp hạng riêng, 1=tốt nhất, cộng 2
    rank lại, chọn K event tổng rank nhỏ nhất) — tránh phải chọn trọng số
    giữa 2 đại lượng khác đơn vị (giá tính bin, thời gian tính số nến).

    Trả về (list event đã lọc, list index GỐC trong fvg_events tương ứng)
    — trả kèm index để caller (render_synthesis_sample) biết chính xác vị
    trí gốc nếu cần, tránh tra cứu lại bằng .index() dễ vỡ nếu có event
    trùng giá trị (dict so sánh bằng value, không phải identity).

    Nếu len(fvg_events) <= k, trả về NGUYÊN VẸN (không cắt, không rank).
    Kết quả LUÔN giữ đúng thứ tự thời gian gốc (chronological), không xáo
    trộn theo rank — rank chỉ dùng để QUYẾT ĐỊNH giữ/bỏ, không phải thứ tự
    hiển thị.
    """
    if len(fvg_events) <= k:
        return fvg_events, list(range(len(fvg_events)))

    current_price, n_candles = _current_price_and_n(raw_chart_text)
    if current_price is None:
        keep = list(range(len(fvg_events) - k, len(fvg_events)))   # fallback: giữ K event gần cuối nhất
        return [fvg_events[i] for i in keep], keep

    price_dists = [abs((e["gap_low"] + e["gap_high"]) / 2 - current_price) for e in fvg_events]
    time_dists  = [(n_candles - 1) - e["fvg_candle_idx"] for e in fvg_events]

    price_rank = {i: r for r, i in enumerate(sorted(range(len(fvg_events)), key=lambda i: price_dists[i]))}
    time_rank  = {i: r for r, i in enumerate(sorted(range(len(fvg_events)), key=lambda i: time_dists[i]))}

    combined_order = sorted(range(len(fvg_events)), key=lambda i: price_rank[i] + time_rank[i])
    keep_indices = sorted(combined_order[:k])   # sort lại theo index gốc -> giữ thứ tự thời gian khi hiển thị

    return [fvg_events[i] for i in keep_indices], keep_indices


# ══════════════════════════════════════════════════════════════════════
# RENDER — 4 dạng mẫu tin
# ══════════════════════════════════════════════════════════════════════

def _render_single_type(
    events: List[Dict[str, Any]],
    request_key: str,
    raw_chart_text: str,
    rng: random.Random,
    total_count: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """
    Dùng chung cho Swept/FVG/Shift — chỉ khác request_key và events list.

    `total_count`: tổng số event THẬT SỰ phát hiện được (trước khi lọc
    top-K, nếu có) — mặc định = len(events) khi không lọc gì. Field
    "total_event_count" trong kết quả LUÔN phản ánh số thật, "event_count"
    phản ánh số ĐÃ HIỂN THỊ — minh bạch khi có lọc bớt, không giấu thông
    tin đã bỏ.
    """
    if not events:
        return None   # v1: SKIP hoàn toàn khi không có event loại này (xem docstring module)

    use_short = len(events) >= 3
    sentences = [_sentence_for_event(e, use_short, rng) for e in events]
    explanation = " ".join(sentences)

    request = rng.choice(_REQUEST_TEMPLATES[request_key])
    eval_block = _build_eval_block(events)

    text = f"{raw_chart_text}\n{request}\n{explanation}\n{eval_block}"

    return {
        "chart"            : raw_chart_text,
        "request"          : request,
        "explanation"      : explanation,
        "eval"             : eval_block,
        "text"             : text,
        "event_count"      : len(events),
        "total_event_count": total_count if total_count is not None else len(events),
    }


def render_swept_sample(fact: Dict[str, Any], raw_chart_text: str, rng: Optional[random.Random] = None) -> Optional[Dict[str, Any]]:
    """Dạng 1 (spec mục 7): chỉ nói về Swept, bỏ qua FVG/Shift dù có xuất hiện."""
    rng = rng or random
    return _render_single_type(fact["swept"], "swept", raw_chart_text, rng)


def render_fvg_sample(fact: Dict[str, Any], raw_chart_text: str, rng: Optional[random.Random] = None) -> Optional[Dict[str, Any]]:
    """
    Dạng 2: chỉ nói về FVG. Áp dụng FVG_TOP_K nếu số FVG thật sự phát hiện
    vượt ngưỡng — xem docstring module "TOP-K CHO FVG" để biết căn cứ.
    """
    rng = rng or random
    all_fvg = fact["fvg"]
    selected, _ = _select_top_k_fvg(all_fvg, raw_chart_text)
    return _render_single_type(selected, "fvg", raw_chart_text, rng, total_count=len(all_fvg))


def render_shift_sample(fact: Dict[str, Any], raw_chart_text: str, rng: Optional[random.Random] = None) -> Optional[Dict[str, Any]]:
    """Dạng 3: chỉ nói về Shift."""
    rng = rng or random
    return _render_single_type(fact["shift"], "shift", raw_chart_text, rng)


def render_synthesis_sample(fact: Dict[str, Any], raw_chart_text: str, rng: Optional[random.Random] = None) -> Optional[Dict[str, Any]]:
    """
    Dạng 4: nói về CẢ 3 loại. KHÔNG CÒN field SEQUENCE (đã bỏ, xem docstring
    module) — thứ tự thời gian truyền đạt bằng cách SẮP XẾP event theo
    candle_idx trước khi hiển thị (sort ổn định giữ đúng rule Swept-trước-
    Shift khi trùng candle, vì list gốc đã theo thứ tự swept+fvg+shift).

    Áp dụng FVG_TOP_K cho phần FVG (giống render_fvg_sample) — Swept/Shift
    KHÔNG lọc (dữ liệu thật xác nhận không cần, xem docstring module).

    KHÔNG CÒN cần facts["relations"] — đã bỏ cùng với SEQUENCE, code đơn
    giản hơn nhiều so với bản trước (không cần remap index sau khi lọc).
    """
    rng = rng or random
    swept_events = fact["swept"]
    all_fvg = fact["fvg"]
    shift_events = fact["shift"]

    total_count = len(swept_events) + len(all_fvg) + len(shift_events)
    if total_count == 0:
        return None   # v1: SKIP khi chart hoàn toàn không có event nào (xem docstring module)

    selected_fvg, _ = _select_top_k_fvg(all_fvg, raw_chart_text)

    # Sắp xếp theo THỜI GIAN (candle_idx) — sort ổn định nên 2 event cùng
    # candle_idx giữ nguyên thứ tự tương đối gốc (Swept trước FVG trước
    # Shift, do nối list theo thứ tự đó trước khi sort) -> tự động khớp
    # đúng rule đã quyết định (Swept trước Shift) mà KHÔNG cần field riêng.
    events = sorted(
        swept_events + selected_fvg + shift_events,
        key=_raw_candle_idx,
    )

    use_short = len(events) >= 3
    sentences = [_sentence_for_event(e, use_short, rng) for e in events]
    explanation = " ".join(sentences)

    request = rng.choice(_REQUEST_TEMPLATES["synthesis"])
    eval_block = _build_eval_block(events)

    text = f"{raw_chart_text}\n{request}\n{explanation}\n{eval_block}"

    return {
        "chart"            : raw_chart_text,
        "request"          : request,
        "explanation"      : explanation,
        "eval"             : eval_block,
        "text"             : text,
        "event_count"      : len(events),
        "total_event_count": total_count,
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