"""
curriculum_generator.py
=======================
Sinh dataset TEXT THUẦN (continued pretraining) theo curriculum 6 tầng,
dạy model nhận thức về phân tích nến ICT theo đúng thứ tự cơ bản → phức tạp:

    Tầng 0 — KHÁI NIỆM       : Nến là gì? Gồm yếu tố nào? Yếu tố nào quan trọng nhất?
    Tầng 1 — PHÂN LOẠI ĐƠN    : Áp dụng khái niệm để gọi tên từng nến (Bull/Bear/Doji)
    Tầng 2 — GIÁ HIỆN TẠI     : Giá hiện tại = Close của nến cuối cùng trong chart
    Tầng 3 — CẤU TRÚC SWING   : Quan hệ NHIỀU nến → đỉnh/đáy cục bộ (Swing High/Low)
    Tầng 4 — FAIR VALUE GAP   : Quan hệ 3 nến → khoảng trống giá (khái niệm khó nhất)
    Tầng 5 — TỔNG HỢP         : Liên kết toàn bộ — 1 nến có thể mang nhiều vai trò

Mỗi tầng sinh ra 1 đoạn TEXT THUẦN độc lập, không có Q&A, không JSON —
đúng định dạng "continued pretraining": model đọc và ngấm kiến thức,
không cần cấu trúc instruction/response.

RANDOM HÓA: không phải mẫu nào cũng cần giới thiệu đủ 6 tầng. Dùng
`generate_random_subset()` (1 chart) hoặc `build_pretrain_dataset(randomize=True)`
(nhiều chart) để mỗi sample chỉ chứa một tổ hợp tầng ngẫu nhiên — có thể là
một TẬP CON rời rạc (vd: tầng 0, 2, 4) hoặc một DẢI LIÊN TỤC giữ thứ tự
(vd: tầng 1→3). Cả hai kiểu được trộn lẫn để dataset đa dạng nhất.

Cách dùng:
    from candle_parser import CandleParser
    parser = CandleParser(raw_text)
    gen = CurriculumGenerator(parser)

    full_text   = gen.generate_full_curriculum()                  # đủ 6 tầng
    random_text = gen.generate_random_subset()                    # tổ hợp ngẫu nhiên
"""

import random
from typing import List, Optional, Dict, Sequence
from candle_parser import CandleParser, Candle


class CurriculumGenerator:
    """
    Sinh text pretrain theo curriculum 6 tầng từ 1 CandleParser đã parse sẵn.
    """

    # Tên hiển thị của từng tầng (dùng khi log / debug, không xuất ra text)
    LAYER_NAMES = [
        "concept",            # 0
        "classify_candle",    # 1
        "current_price",      # 2
        "swing_structure",    # 3
        "fvg",                # 4
        "synthesis",          # 5
    ]

    def __init__(self, parser: CandleParser):
        self.parser  = parser
        self.candles = parser.candles
        self.n       = len(parser)

        # Danh sách method theo ĐÚNG thứ tự tầng 0→5, dùng chung cho
        # generate_full_curriculum / generate_layers_separately / random subset.
        self._layer_methods = [
            self.layer0_concept,
            self.layer1_classify_each_candle,
            self.layer2_current_price,
            self.layer3_swing_structure,
            self.layer4_fvg,
            self.layer5_synthesis,
        ]

    @property
    def num_layers(self) -> int:
        return len(self._layer_methods)

    # ══════════════════════════════════════════════════════════════
    # TẦNG 0 — KHÁI NIỆM CƠ BẢN VỀ NẾN
    # ══════════════════════════════════════════════════════════════
    def layer0_concept(self) -> str:
        """
        Dạy khái niệm nền tảng: nến là gì, gồm yếu tố nào, yếu tố nào quan trọng nhất.
        Không nhắc đến chart cụ thể — đây là kiến thức tổng quát, lặp lại để model ghi nhớ.
        """
        return """\
=== KHÁI NIỆM CƠ BẢN: CÂY NẾN (CANDLE) ===

Một cây nến trong biểu đồ giá đại diện cho biến động giá trong một khoảng thời gian.
Mỗi cây nến gồm 4 yếu tố:
- Open  (giá mở cửa): giá tại thời điểm cây nến bắt đầu hình thành.
- High  (giá cao nhất): mức giá cao nhất đạt được trong suốt thời gian của cây nến.
- Low   (giá thấp nhất): mức giá thấp nhất đạt được trong suốt thời gian của cây nến.
- Close (giá đóng cửa): giá tại thời điểm cây nến kết thúc.

Trong 4 yếu tố trên, hai yếu tố quan trọng nhất để xác định HƯỚNG của cây nến là
Open và Close, vì chúng cho biết giá đã đi lên hay đi xuống trong khoảng thời gian đó:
- Nếu Close lớn hơn Open: giá đã tăng trong suốt cây nến, đây là nến TĂNG (Bull/Bullish).
- Nếu Close nhỏ hơn Open: giá đã giảm trong suốt cây nến, đây là nến GIẢM (Bear/Bearish).
- Nếu Close gần như bằng Open: giá không đổi nhiều, đây là nến DOJI, thể hiện sự lưỡng lự.

Hai yếu tố High và Low ít quan trọng hơn khi xét HƯỚNG của riêng một cây nến,
nhưng lại trở nên rất quan trọng khi so sánh NHIỀU cây nến với nhau, vì chúng
cho biết mức giá xa nhất mà thị trường đã chạm tới trong khoảng thời gian đó.
"""

    # ══════════════════════════════════════════════════════════════
    # TẦNG 1 — PHÂN LOẠI TỪNG NẾN ĐƠN LẺ
    # ══════════════════════════════════════════════════════════════
    def layer1_classify_each_candle(self) -> str:
        """
        Áp dụng khái niệm Tầng 0 vào chart thực tế: gọi tên từng nến.
        Đây là bước đầu tiên model THỰC HÀNH khái niệm vừa học, trên dữ liệu cụ thể.
        """
        lines: List[str] = []
        lines.append("=== ÁP DỤNG: PHÂN LOẠI TỪNG CÂY NẾN TRONG CHART ===\n")
        lines.append(f"Chart này có tổng cộng {self.n} cây nến. "
                      f"Dựa vào khái niệm Open và Close, phân loại từng cây nến như sau:\n")

        for i in range(self.n):
            c = self.candles[i]
            ordinal = i + 1
            lines.append(c.description(ordinal))

        return "\n".join(lines).strip() + "\n"

    # ══════════════════════════════════════════════════════════════
    # TẦNG 2 — GIÁ HIỆN TẠI (Close của nến cuối cùng)
    # ══════════════════════════════════════════════════════════════
    def layer2_current_price(self) -> str:
        """
        Dạy khái niệm "giá hiện tại": là giá Close của cây nến CUỐI CÙNG trong chart
        (nến mới nhất). Chỉ nêu giá trị, không suy luận thêm — đúng yêu cầu giữ đơn giản.
        """
        last_candle = self.candles[-1]
        ordinal     = self.n

        lines: List[str] = []
        lines.append("=== KHÁI NIỆM: GIÁ HIỆN TẠI ===\n")
        lines.append(
            "Giá hiện tại của một chart là giá Close (giá đóng cửa) của cây nến CUỐI CÙNG "
            "trong chart đó, vì đây là cây nến mới nhất, gần với thời điểm hiện tại nhất.\n"
        )
        lines.append(
            f"Trong chart này, cây nến cuối cùng là cây nến thứ {ordinal} {last_candle.tag()}. "
            f"Vậy giá hiện tại của chart này là {last_candle.close:g}."
        )

        return "\n".join(lines).strip() + "\n"

    # ══════════════════════════════════════════════════════════════
    # TẦNG 3 — CẤU TRÚC SWING (QUAN HỆ NHIỀU NẾN)
    # ══════════════════════════════════════════════════════════════
    def layer3_swing_structure(self) -> str:
        """
        Dạy khái niệm Swing High/Low: cần XEM NHIỀU NẾN mới nhận biết được,
        khác với Tầng 1 chỉ cần nhìn 1 nến. Giải thích TẠI SAO nó quan trọng hơn
        việc phân loại đơn lẻ: nó cho biết đỉnh/đáy của thị trường.
        """
        lines: List[str] = []
        lines.append("=== KHÁI NIỆM NÂNG CAO: SWING HIGH / SWING LOW ===\n")
        lines.append(
            "Khác với việc phân loại một cây nến đơn lẻ (chỉ cần nhìn Open và Close của "
            "chính nó), để nhận biết Swing High hoặc Swing Low, ta cần SO SÁNH một cây nến "
            "với NHIỀU cây nến xung quanh nó (cả bên trái và bên phải).\n"
        )
        lines.append(
            "- Swing High (đỉnh cục bộ): một cây nến có giá High CAO HƠN tất cả các cây nến "
            "lân cận xung quanh nó. Đây là điểm mà giá đã đảo chiều đi xuống.\n"
            "- Swing Low (đáy cục bộ): một cây nến có giá Low THẤP HƠN tất cả các cây nến "
            "lân cận xung quanh nó. Đây là điểm mà giá đã đảo chiều đi lên.\n"
        )
        lines.append(
            "Swing High và Swing Low quan trọng hơn việc phân loại tăng/giảm đơn lẻ, vì chúng "
            "đánh dấu các vùng mà thị trường đã thay đổi hướng đi — đây là nền tảng để xác định "
            "cấu trúc thị trường (market structure) và các vùng thanh khoản (liquidity).\n"
        )

        found_any = False
        for i in range(self.n):
            c = self.candles[i]
            ordinal = i + 1
            if self.parser.is_swing_high(i):
                lines.append(
                    f"Cây nến thứ {ordinal} {c.tag()} có giá cao nhất {c.high:g}, "
                    f"cao hơn tất cả các cây nến lân cận xung quanh nó "
                    f"→ đây là một SWING HIGH."
                )
                found_any = True
            if self.parser.is_swing_low(i):
                lines.append(
                    f"Cây nến thứ {ordinal} {c.tag()} có giá thấp nhất {c.low:g}, "
                    f"thấp hơn tất cả các cây nến lân cận xung quanh nó "
                    f"→ đây là một SWING LOW."
                )
                found_any = True

        if not found_any:
            lines.append("Trong chart này không tìm thấy Swing High hoặc Swing Low rõ ràng.")

        return "\n".join(lines).strip() + "\n"

    # ══════════════════════════════════════════════════════════════
    # TẦNG 4 — FAIR VALUE GAP (KHÁI NIỆM KHÓ NHẤT, CẦN 3 NẾN)
    # ══════════════════════════════════════════════════════════════
    def layer4_fvg(self) -> str:
        """
        Dạy khái niệm Fair Value Gap: phức tạp hơn Swing vì cần đúng 3 nến liên tiếp
        và phải so sánh nến đầu với nến thứ ba (bỏ qua nến giữa).
        Giải thích vì sao đây là khái niệm QUAN TRỌNG NHẤT trong các khái niệm đã học,
        vì nó thể hiện sự mất cân bằng giá (price imbalance) mà thị trường có xu hướng
        quay lại lấp đầy.
        """
        lines: List[str] = []
        lines.append("=== KHÁI NIỆM QUAN TRỌNG NHẤT: FAIR VALUE GAP (FVG) ===\n")
        lines.append(
            "Fair Value Gap (FVG) là khái niệm phức tạp hơn Swing High/Low, vì nó không "
            "chỉ so sánh một cây nến với các nến lân cận, mà xét MỘT CỤM ĐÚNG 3 CÂY NẾN "
            "LIÊN TIẾP, và so sánh nến ĐẦU TIÊN với nến THỨ BA (bỏ qua nến ở giữa).\n"
        )
        lines.append(
            "- Bullish FVG (FVG tăng): khi giá Low của nến thứ ba CAO HƠN giá High của "
            "nến đầu tiên. Điều này nghĩa là giá đã tăng quá nhanh, bỏ lại một khoảng "
            "trống (gap) chưa có giao dịch nào diễn ra ở vùng đó.\n"
            "- Bearish FVG (FVG giảm): khi giá High của nến thứ ba THẤP HƠN giá Low của "
            "nến đầu tiên. Điều này nghĩa là giá đã giảm quá nhanh, bỏ lại một khoảng "
            "trống chưa có giao dịch ở vùng đó.\n"
        )
        lines.append(
            "FVG là khái niệm QUAN TRỌNG NHẤT trong số các khái niệm đã học, vì nó thể hiện "
            "sự MẤT CÂN BẰNG GIÁ (price imbalance) — thị trường thường có xu hướng quay lại "
            "để 'lấp đầy' vùng giá còn trống này trước khi tiếp tục di chuyển theo xu hướng cũ. "
            "Vì vậy, vùng FVG thường được dùng làm điểm tham chiếu để dự đoán nơi giá sẽ quay "
            "lại trong tương lai.\n"
        )

        found_any = False
        for i in range(self.n):
            fvg = self.parser.is_fvg(i)
            if fvg:
                first_idx  = i - 2
                middle_idx = i - 1
                third_idx  = i
                c_first = self.candles[first_idx]
                c_third = self.candles[third_idx]
                fvg_vn  = "TĂNG (Bullish FVG)" if fvg == "BULL" else "GIẢM (Bearish FVG)"

                if fvg == "BULL":
                    detail = (
                        f"giá High của nến thứ {first_idx + 1} là {c_first.high:g}, "
                        f"thấp hơn giá Low của nến thứ {third_idx + 1} là {c_third.low:g}, "
                        f"tạo ra một khoảng trống giá chưa được giao dịch."
                    )
                else:
                    detail = (
                        f"giá Low của nến thứ {first_idx + 1} là {c_first.low:g}, "
                        f"cao hơn giá High của nến thứ {third_idx + 1} là {c_third.high:g}, "
                        f"tạo ra một khoảng trống giá chưa được giao dịch."
                    )

                lines.append(
                    f"Cây nến thứ {first_idx + 1} {c_first.tag()}, cây nến thứ {middle_idx + 1}, "
                    f"và cây nến thứ {third_idx + 1} {c_third.tag()} tạo thành một FVG {fvg_vn}: "
                    f"{detail}"
                )
                found_any = True

        if not found_any:
            lines.append("Trong chart này không tìm thấy Fair Value Gap nào.")

        return "\n".join(lines).strip() + "\n"

    # ══════════════════════════════════════════════════════════════
    # TẦNG 5 — TỔNG HỢP: LIÊN KẾT TOÀN BỘ KHÁI NIỆM
    # ══════════════════════════════════════════════════════════════
    def layer5_synthesis(self) -> str:
        """
        Liên kết lại toàn bộ: một cây nến có thể đồng thời mang nhiều vai trò
        (vừa là Bull/Bear, vừa là Swing High/Low, vừa thuộc về một FVG).
        Đây là bước tổng hợp giúp model thấy được BỨC TRANH TOÀN CẢNH,
        thay vì chỉ học từng khái niệm riêng lẻ.
        """
        lines: List[str] = []
        lines.append("=== TỔNG HỢP: LIÊN KẾT CÁC KHÁI NIỆM TRÊN CÙNG MỘT CÂY NẾN ===\n")
        lines.append(
            "Một cây nến có thể đồng thời mang nhiều vai trò khác nhau cùng lúc: "
            "nó vừa có hướng tăng/giảm/doji, vừa có thể là một Swing High hoặc "
            "Swing Low, vừa có thể là một phần của Fair Value Gap. "
            "Hiểu được sự kết hợp này giúp đọc đúng bức tranh toàn cảnh của thị trường, "
            "vì các khái niệm không tồn tại độc lập mà bổ sung ý nghĩa cho nhau.\n"
        )

        any_combo = False
        for i in range(self.n):
            c = self.candles[i]
            ordinal = i + 1
            roles: List[str] = []

            direction = self.parser.is_bull_bear(i)
            direction_vn = {"BULL": "nến tăng", "BEAR": "nến giảm", "DOJI": "nến doji"}[direction]
            roles.append(direction_vn)

            if self.parser.is_swing_high(i):
                roles.append("Swing High (đỉnh cục bộ)")
            if self.parser.is_swing_low(i):
                roles.append("Swing Low (đáy cục bộ)")

            fvg_as_third = self.parser.is_fvg(i)
            if fvg_as_third:
                fvg_vn = "Bullish FVG" if fvg_as_third == "BULL" else "Bearish FVG"
                roles.append(f"nến hoàn thiện một {fvg_vn} (cùng nến {i - 1} và {i})")

            # Chỉ liệt kê những nến có NHIỀU HƠN 1 vai trò đáng chú ý (direction + ít nhất 1 cấu trúc)
            if len(roles) > 1:
                extra_roles = roles[1:]
                if len(extra_roles) == 1:
                    extra_text = extra_roles[0]
                else:
                    extra_text = ", ".join(extra_roles[:-1]) + f", và {extra_roles[-1]}"
                lines.append(
                    f"Cây nến thứ {ordinal} {c.tag()} là {roles[0]}, "
                    f"đồng thời cũng là {extra_text}."
                )
                any_combo = True

        if not any_combo:
            lines.append(
                "Trong chart này, không có cây nến nào mang đồng thời nhiều vai trò cấu trúc "
                "đáng chú ý — các khái niệm Swing và FVG xuất hiện ở những cây nến khác nhau."
            )

        return "\n".join(lines).strip() + "\n"

    # ══════════════════════════════════════════════════════════════
    # GHÉP TOÀN BỘ CURRICULUM (ĐỦ 6 TẦNG)
    # ══════════════════════════════════════════════════════════════
    def generate_full_curriculum(self, include_chart_header: bool = True) -> str:
        """
        Sinh toàn bộ text pretrain theo đúng thứ tự 6 tầng, từ khái niệm cơ bản
        đến tổng hợp phức tạp. Dùng khi muốn 1 sample đầy đủ, không random.
        """
        return self._render(list(range(self.num_layers)), include_chart_header)

    def generate_layers_separately(self) -> List[str]:
        """
        Trả về list 6 đoạn text riêng biệt (1 đoạn / 1 tầng) thay vì 1 văn bản dài.
        Hữu ích nếu muốn mỗi tầng là 1 sample riêng trong dataset pretrain
        (giúp model học từng khái niệm độc lập, tránh 1 sample quá dài).
        """
        return [method() for method in self._layer_methods]

    # ══════════════════════════════════════════════════════════════
    # RANDOM SUBSET — không phải sample nào cũng cần đủ 6 tầng
    # ══════════════════════════════════════════════════════════════
    def generate_random_subset(
        self,
        mode: Optional[str] = None,         # "subset" | "range" | None (tự random chọn 1 trong 2)
        min_layers: int = 2,
        max_layers: Optional[int] = None,
        include_chart_header: bool = True,
        rng: Optional[random.Random] = None,
    ) -> str:
        """
        Sinh 1 đoạn text pretrain từ MỘT TỔ HỢP TẦNG NGẪU NHIÊN, không cần đủ 6 tầng.

        Có 2 kiểu chọn:
        - "subset" : chọn ngẫu nhiên k tầng BẤT KỲ (không cần liên tục), nhưng vẫn
                     SẮP XẾP LẠI theo đúng thứ tự gốc 0→5 khi render — vì nội dung
                     mỗi tầng được viết để đọc theo thứ tự tăng dần độ khó, đảo
                     ngược thứ tự sẽ làm văn bản không còn mạch lạc về nhận thức.
        - "range"  : chọn một DẢI LIÊN TỤC [start, end] (vd: tầng 1→3), giữ nguyên
                     tính tuần tự tăng dần độ khó trong dải đó.

        Parameters
        ----------
        mode        : "subset", "range", hoặc None (ngẫu nhiên chọn 1 trong 2 kiểu)
        min_layers  : số tầng tối thiểu trong 1 sample
        max_layers  : số tầng tối đa (mặc định = num_layers, tức có thể ra đủ 6 tầng)
        rng         : random.Random tùy chọn, để tái lập kết quả (testing/seed)

        Returns
        -------
        str — đoạn text pretrain ứng với tổ hợp tầng đã chọn.
        """
        rng = rng or random
        max_layers = max_layers if max_layers is not None else self.num_layers
        max_layers = min(max_layers, self.num_layers)
        min_layers = max(1, min(min_layers, max_layers))

        chosen_mode = mode or rng.choice(["subset", "range"])

        if chosen_mode == "range":
            # Chọn 1 dải liên tục [start, end], độ dài trong [min_layers, max_layers]
            length = rng.randint(min_layers, max_layers)
            start_max = self.num_layers - length
            start = rng.randint(0, max(0, start_max))
            indices = list(range(start, start + length))

        elif chosen_mode == "subset":
            # Chọn k tầng bất kỳ, không cần liên tục, rồi sort lại theo thứ tự gốc
            k = rng.randint(min_layers, max_layers)
            indices = sorted(rng.sample(range(self.num_layers), k))

        else:
            raise ValueError(f"mode không hợp lệ: {chosen_mode}. Dùng 'subset' hoặc 'range'.")

        return self._render(indices, include_chart_header)

    # ══════════════════════════════════════════════════════════════
    # HELPER NỘI BỘ: render 1 danh sách index tầng thành text
    # ══════════════════════════════════════════════════════════════
    def _render(self, layer_indices: Sequence[int], include_chart_header: bool) -> str:
        parts: List[str] = []

        if include_chart_header:
            parts.append(f"Dữ liệu chart đang phân tích: {self.parser.raw_text.strip()}\n")

        for idx in layer_indices:
            parts.append(self._layer_methods[idx]())

        return "\n\n".join(p.strip() for p in parts) + "\n"


# ══════════════════════════════════════════════════════════════════
# BATCH BUILDER — sinh dataset từ NHIỀU chart cùng lúc
# ══════════════════════════════════════════════════════════════════

def build_pretrain_dataset(
    raw_charts: List[str],
    swing_window: int = 2,
    mode: str = "full",          # "full" | "layers" | "random"
    randomize: bool = False,     # tương đương mode="random" (giữ tương thích cũ)
    min_layers: int = 2,
    max_layers: Optional[int] = None,
    separator: str = "\n\n<|endofsample|>\n\n",
    seed: Optional[int] = None,
) -> str:
    """
    Sinh dataset pretrain TEXT THUẦN từ nhiều chuỗi chart thô.

    Parameters
    ----------
    raw_charts : list các chuỗi "<chart> O_.. H_.. L_.. C_.. ... </chart>"
    mode       : "full"   → mỗi chart sinh 1 văn bản đầy đủ 6 tầng (liên tục)
                 "layers" → mỗi tầng của mỗi chart là 1 sample riêng
                 "random" → mỗi chart sinh 1 văn bản với TỔ HỢP TẦNG NGẪU NHIÊN
                            (trộn lẫn cả kiểu "subset" rời rạc và "range" liên tục)
    randomize  : nếu True, ép buộc dùng mode="random" (tiện khi gọi nhanh)
    min_layers, max_layers : giới hạn số tầng mỗi sample khi mode="random"
    separator  : chuỗi phân tách giữa các sample khi ghép thành 1 file lớn
    seed       : seed cố định để tái lập dataset (testing / reproducibility)

    Returns
    -------
    str — toàn bộ dataset, các sample nối bằng `separator`.
    """
    if randomize:
        mode = "random"

    rng = random.Random(seed) if seed is not None else random
    samples: List[str] = []

    for raw in raw_charts:
        parser = CandleParser(raw, swing_window=swing_window)
        gen    = CurriculumGenerator(parser)

        if mode == "full":
            samples.append(gen.generate_full_curriculum())
        elif mode == "layers":
            samples.extend(gen.generate_layers_separately())
        elif mode == "random":
            samples.append(gen.generate_random_subset(
                min_layers=min_layers, max_layers=max_layers, rng=rng
            ))
        else:
            raise ValueError(f"mode không hợp lệ: {mode}. Dùng 'full', 'layers', hoặc 'random'.")

    return separator.join(s.strip() for s in samples) + "\n"


# ══════════════════════════════════════════════════════════════════
# DEMO / SELF-TEST
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    sample = (
        "<chart> O_512 H_521 L_500 C_500 O_500 H_521 L_500 C_514 "
        "O_514 H_516 L_500 C_510 O_508 H_511 L_499 C_503 "
        "O_502 H_502 L_475 C_491 O_490 H_490 L_478 C_489 "
        "O_494 H_507 L_481 C_481 O_479 H_484 L_473 C_473 </chart>"
    )

    parser = CandleParser(sample, swing_window=2)
    gen    = CurriculumGenerator(parser)

    print("=== FULL (đủ 6 tầng) ===\n")
    full_text = gen.generate_full_curriculum()
    print(full_text)
    print(f"[Độ dài: {len(full_text)} ký tự]\n")

    print("\n=== RANDOM SUBSET (kiểu 'subset', rời rạc) ===\n")
    rng1 = random.Random(42)
    print(gen.generate_random_subset(mode="subset", rng=rng1))

    print("\n=== RANDOM SUBSET (kiểu 'range', liên tục) ===\n")
    rng2 = random.Random(42)
    print(gen.generate_random_subset(mode="range", rng=rng2))