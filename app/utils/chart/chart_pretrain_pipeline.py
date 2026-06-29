"""
chart_pretrain_pipeline.py
==========================
Nhánh 1: Gen data pretrain dạy model hiểu chart cơ bản.

Pipeline:
    DataFrame OHLCV (từ ChartDatasetBuilder / nhánh 2)
    hoặc raw chart text (<chart> O_.. H_.. ... </chart>)
    → CandleParser   : parse token text → list Candle
    → CurriculumGenerator : sinh text pretrain 7 tầng
    → parquet lẻ (cùng schema với nhánh 2 và 3)

Schema output:
    text          : string  — nội dung pretrain
    source        : string  — tên nguồn (vd "XAUUSD_1Min")
    token_length  : int64   — 0 (không tokenize ở bước này)
    meta          : string  — JSON {source_chart_index, slice_start,
                                    slice_end, num_candles, num_layers}

Cách dùng:
    from chart_pretrain_pipeline import ChartPretrainPipeline

    pipeline = ChartPretrainPipeline(source_name="XAUUSD_1Min")
    pipeline.build_from_parquet(
        input_path  = "data/chart_XAUUSD_dataset_1Min.parquet",
        output_path = "data/pretrain_XAUUSD_1Min.parquet",
    )
"""

import json
import random
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    _HAS_PYARROW = True
except ImportError:
    _HAS_PYARROW = False


# ══════════════════════════════════════════════════════════════════════
# CANDLE
# ══════════════════════════════════════════════════════════════════════

@dataclass
class Candle:
    open : float
    high : float
    low  : float
    close: float

    def is_bull(self) -> bool: return self.close > self.open + 1
    def is_bear(self) -> bool: return self.close < self.open - 1

    def tag(self) -> str:
        return (f"<chart> O_{self.open:g} H_{self.high:g} "
                f"L_{self.low:g} C_{self.close:g} </chart>")

    def direction(self) -> str:
        if self.is_bull(): return "BULL"
        if self.is_bear(): return "BEAR"
        return "DOJI"

    def description(self, index: int) -> str:
        dir_vn = {
            "BULL": "nến tăng giá",
            "BEAR": "nến giảm giá",
            "DOJI": "nến doji (giá không đổi)",
        }[self.direction()]
        return (
            f"\tCây nến thứ {index} {self.tag()} là {dir_vn}"
            f" có giá mở cửa {self.open:g}, giá đóng cửa {self.close:g},"
            f" giá cao nhất {self.high:g}, giá thấp nhất {self.low:g}."
        )


# ══════════════════════════════════════════════════════════════════════
# CANDLE PARSER
# ══════════════════════════════════════════════════════════════════════

_CANDLE_RE = re.compile(
    r"O_(-?\d+(?:\.\d+)?)\s+"
    r"H_(-?\d+(?:\.\d+)?)\s+"
    r"L_(-?\d+(?:\.\d+)?)\s+"
    r"C_(-?\d+(?:\.\d+)?)"
)


class CandleParser:
    """Parse chuỗi chart token thành list Candle, hỗ trợ slice."""

    def __init__(self, raw_text: str, swing_window: int = 2):
        self.raw_text     = raw_text
        self.swing_window = swing_window
        self.candles      = self._parse(raw_text)

    @staticmethod
    def _parse(text: str) -> List[Candle]:
        return [
            Candle(float(o), float(h), float(l), float(c))
            for o, h, l, c in _CANDLE_RE.findall(text)
        ]

    @classmethod
    def from_candles(cls, candles: List[Candle], swing_window: int = 2) -> "CandleParser":
        obj              = cls.__new__(cls)
        obj.swing_window = swing_window
        obj.candles      = list(candles)
        obj.raw_text     = cls._build_raw(candles)
        return obj

    @staticmethod
    def _build_raw(candles: List[Candle]) -> str:
        tokens = [f"O_{c.open:g} H_{c.high:g} L_{c.low:g} C_{c.close:g}" for c in candles]
        return "<chart> " + " ".join(tokens) + " </chart>"

    def slice(self, start: int, end: int) -> "CandleParser":
        return CandleParser.from_candles(self.candles[start:end], self.swing_window)

    def __len__(self)         : return len(self.candles)
    def __getitem__(self, idx): return self.candles[idx]

    # ── Swing ──────────────────────────────────────────────────────
    def is_swing_high(self, i: int, window: int = None) -> bool:
        w = window or self.swing_window
        if i - w < 0 or i + w >= len(self.candles): return False
        target = self.candles[i].high
        return target == max(self.candles[j].high for j in range(i - w, i + w + 1))

    def is_swing_low(self, i: int, window: int = None) -> bool:
        w = window or self.swing_window
        if i - w < 0 or i + w >= len(self.candles): return False
        target = self.candles[i].low
        return target == min(self.candles[j].low for j in range(i - w, i + w + 1))

    # ── FVG ────────────────────────────────────────────────────────
    def is_fvg(self, i: int) -> Optional[str]:
        if i < 2: return None
        c0, c2 = self.candles[i - 2], self.candles[i]
        if c2.low  > c0.high: return "BULL"
        if c2.high < c0.low : return "BEAR"
        return None

    # ── Pattern helpers ────────────────────────────────────────────
    @staticmethod
    def _body(c)       : return abs(c.close - c.open)
    @staticmethod
    def _upper_wick(c) : return c.high - max(c.open, c.close)
    @staticmethod
    def _lower_wick(c) : return min(c.open, c.close) - c.low
    @staticmethod
    def _range(c)      : r = c.high - c.low; return r if r > 0 else 1e-9

    def is_pin_bar(self, i: int, wick_ratio=0.6, body_ratio=0.3) -> Optional[str]:
        c = self.candles[i]
        rng, body = self._range(c), self._body(c)
        if body > body_ratio * rng: return None
        upper, lower = self._upper_wick(c), self._lower_wick(c)
        if lower >= wick_ratio * rng and upper < lower * 0.5: return "HAMMER"
        if upper >= wick_ratio * rng and lower < upper * 0.5: return "SHOOTING_STAR"
        return None

    def is_engulfing(self, i: int) -> Optional[str]:
        if i < 1: return None
        prev, curr = self.candles[i - 1], self.candles[i]
        pd_, cd_   = self.candles[i - 1].direction(), curr.direction()
        if pd_ == "BEAR" and cd_ == "BULL":
            if curr.open <= prev.close and curr.close >= prev.open:
                return "BULLISH_ENGULFING"
        if pd_ == "BULL" and cd_ == "BEAR":
            if curr.open >= prev.close and curr.close <= prev.open:
                return "BEARISH_ENGULFING"
        return None


# ══════════════════════════════════════════════════════════════════════
# CURRICULUM GENERATOR — 7 tầng
# ══════════════════════════════════════════════════════════════════════

class CurriculumGenerator:
    """
    Sinh text pretrain theo curriculum 7 tầng từ 1 CandleParser.

    Tầng 0 — Khái niệm nến cơ bản
    Tầng 1 — Phân loại từng nến
    Tầng 2 — Giá hiện tại
    Tầng 3 — Swing High / Swing Low
    Tầng 4 — Fair Value Gap
    Tầng 5 — Tổng hợp
    Tầng 6 — Mẫu hình nến (Pin Bar, Engulfing)
    """

    def __init__(self, parser: CandleParser):
        self.parser  = parser
        self.candles = parser.candles
        self.n       = len(parser)
        self._layers = [
            self._layer0_concept,
            self._layer1_classify,
            self._layer2_current_price,
            self._layer3_swing,
            self._layer4_fvg,
            self._layer5_synthesis,
            self._layer6_patterns,
        ]

    @property
    def num_layers(self) -> int: return len(self._layers)

    # ── Tầng 0 ────────────────────────────────────────────────────
    def _layer0_concept(self) -> str:
        return """\
=== KHÁI NIỆM CƠ BẢN: CÂY NẾN (CANDLE) ===

Một cây nến trong biểu đồ giá đại diện cho biến động giá trong một khoảng thời gian.
Mỗi cây nến gồm 4 yếu tố:
- Open  (giá mở cửa): giá tại thời điểm cây nến bắt đầu hình thành.
- High  (giá cao nhất): mức giá cao nhất đạt được trong suốt thời gian của cây nến.
- Low   (giá thấp nhất): mức giá thấp nhất đạt được trong suốt thời gian của cây nến.
- Close (giá đóng cửa): giá tại thời điểm cây nến kết thúc.

Trong 4 yếu tố trên, hai yếu tố quan trọng nhất để xác định HƯỚNG của cây nến là
Open và Close:
- Nếu Close lớn hơn Open: nến TĂNG (Bull/Bullish).
- Nếu Close nhỏ hơn Open: nến GIẢM (Bear/Bearish).
- Nếu Close gần như bằng Open: nến DOJI, thể hiện sự lưỡng lự.

High và Low quan trọng khi so sánh NHIỀU nến — cho biết mức giá xa nhất thị trường
đã chạm tới trong khoảng thời gian đó.
"""

    # ── Tầng 1 ────────────────────────────────────────────────────
    def _layer1_classify(self) -> str:
        lines = [
            "=== ÁP DỤNG: PHÂN LOẠI TỪNG CÂY NẾN TRONG CHART ===\n",
            f"Chart này có tổng cộng {self.n} cây nến:\n",
        ]
        for i, c in enumerate(self.candles):
            lines.append(c.description(i + 1))
        return "\n".join(lines).strip() + "\n"

    # ── Tầng 2 ────────────────────────────────────────────────────
    def _layer2_current_price(self) -> str:
        last = self.candles[-1]
        return (
            "=== KHÁI NIỆM: GIÁ HIỆN TẠI ===\n\n"
            "Giá hiện tại là giá Close của cây nến CUỐI CÙNG trong chart.\n\n"
            f"Trong chart này, cây nến cuối cùng là cây nến thứ {self.n} {last.tag()}. "
            f"Vậy giá hiện tại là {last.close:g}.\n"
        )

    # ── Tầng 3 ────────────────────────────────────────────────────
    def _layer3_swing(self) -> str:
        lines = [
            "=== KHÁI NIỆM NÂNG CAO: SWING HIGH / SWING LOW ===\n",
            "Swing High (đỉnh cục bộ): nến có High CAO HƠN tất cả nến lân cận.\n"
            "Swing Low  (đáy cục bộ): nến có Low THẤP HƠN tất cả nến lân cận.\n",
        ]
        found = False
        for i, c in enumerate(self.candles):
            if self.parser.is_swing_high(i):
                lines.append(
                    f"Cây nến thứ {i+1} {c.tag()} có High={c.high:g} "
                    f"cao hơn tất cả nến lân cận → SWING HIGH."
                )
                found = True
            if self.parser.is_swing_low(i):
                lines.append(
                    f"Cây nến thứ {i+1} {c.tag()} có Low={c.low:g} "
                    f"thấp hơn tất cả nến lân cận → SWING LOW."
                )
                found = True
        if not found:
            lines.append("Không tìm thấy Swing High hoặc Swing Low rõ ràng trong chart này.")
        return "\n".join(lines).strip() + "\n"

    # ── Tầng 4 ────────────────────────────────────────────────────
    def _layer4_fvg(self) -> str:
        lines = [
            "=== KHÁI NIỆM QUAN TRỌNG NHẤT: FAIR VALUE GAP (FVG) ===\n",
            "FVG xét 3 nến liên tiếp, so sánh nến ĐẦU với nến THỨ BA:\n"
            "- Bullish FVG: Low nến thứ 3 > High nến thứ 1 → khoảng trống tăng.\n"
            "- Bearish FVG: High nến thứ 3 < Low nến thứ 1 → khoảng trống giảm.\n",
        ]
        found = False
        for i in range(self.n):
            fvg = self.parser.is_fvg(i)
            if not fvg: continue
            c0, c2  = self.candles[i - 2], self.candles[i]
            fvg_vn  = "TĂNG (Bullish FVG)" if fvg == "BULL" else "GIẢM (Bearish FVG)"
            detail  = (
                f"Low nến {i+1}={c2.low:g} > High nến {i-1}={c0.high:g}"
                if fvg == "BULL" else
                f"High nến {i+1}={c2.high:g} < Low nến {i-1}={c0.low:g}"
            )
            lines.append(
                f"Nến {i-1}, {i}, {i+1}: FVG {fvg_vn} — {detail}."
            )
            found = True
        if not found:
            lines.append("Không tìm thấy Fair Value Gap trong chart này.")
        return "\n".join(lines).strip() + "\n"

    # ── Tầng 5 ────────────────────────────────────────────────────
    def _layer5_synthesis(self) -> str:
        lines = [
            "=== TỔNG HỢP: LIÊN KẾT CÁC KHÁI NIỆM TRÊN CÙNG MỘT CÂY NẾN ===\n",
            "Một cây nến có thể đồng thời mang nhiều vai trò: tăng/giảm/doji, "
            "Swing High/Low, và thuộc về một FVG.\n",
        ]
        found = False
        for i, c in enumerate(self.candles):
            roles = [{"BULL": "nến tăng", "BEAR": "nến giảm", "DOJI": "nến doji"}[c.direction()]]
            if self.parser.is_swing_high(i): roles.append("Swing High (đỉnh cục bộ)")
            if self.parser.is_swing_low(i) : roles.append("Swing Low (đáy cục bộ)")
            fvg = self.parser.is_fvg(i)
            if fvg:
                roles.append(f"nến hoàn thiện {'Bullish' if fvg=='BULL' else 'Bearish'} FVG")
            if len(roles) > 1:
                extra = ", ".join(roles[1:])
                lines.append(f"Cây nến thứ {i+1} {c.tag()} là {roles[0]}, đồng thời {extra}.")
                found = True
        if not found:
            lines.append("Không có nến nào mang đồng thời nhiều vai trò cấu trúc trong chart này.")
        return "\n".join(lines).strip() + "\n"

    # ── Tầng 6 ────────────────────────────────────────────────────
    def _layer6_patterns(self) -> str:
        lines = [
            "=== KHÁI NIỆM: MẪU HÌNH NẾN (PRICE ACTION PATTERNS) ===\n",
            "Các mẫu hình nến là gợi ý xác suất, KHÔNG phải quy luật chắc chắn.\n"
            "- Hammer: bóng dưới dài, thân nhỏ → gợi ý đảo chiều TĂNG.\n"
            "- Shooting Star: bóng trên dài, thân nhỏ → gợi ý đảo chiều GIẢM.\n"
            "- Bullish Engulfing: nến tăng nuốt trọn thân nến giảm trước → gợi ý TĂNG.\n"
            "- Bearish Engulfing: nến giảm nuốt trọn thân nến tăng trước → gợi ý GIẢM.\n",
        ]
        found = False
        for i, c in enumerate(self.candles):
            pin = self.parser.is_pin_bar(i)
            if pin == "HAMMER":
                lines.append(
                    f"Cây nến thứ {i+1} {c.tag()} có bóng dưới dài, thân nhỏ "
                    f"→ HAMMER, gợi ý đảo chiều TĂNG (xem xét BUY)."
                )
                found = True
            elif pin == "SHOOTING_STAR":
                lines.append(
                    f"Cây nến thứ {i+1} {c.tag()} có bóng trên dài, thân nhỏ "
                    f"→ SHOOTING STAR, gợi ý đảo chiều GIẢM (xem xét SELL)."
                )
                found = True

            eng = self.parser.is_engulfing(i)
            if eng == "BULLISH_ENGULFING":
                prev = self.candles[i - 1]
                lines.append(
                    f"Nến {i} {prev.tag()} và nến {i+1} {c.tag()} "
                    f"→ BULLISH ENGULFING, gợi ý lực mua áp đảo (xem xét BUY)."
                )
                found = True
            elif eng == "BEARISH_ENGULFING":
                prev = self.candles[i - 1]
                lines.append(
                    f"Nến {i} {prev.tag()} và nến {i+1} {c.tag()} "
                    f"→ BEARISH ENGULFING, gợi ý lực bán áp đảo (xem xét SELL)."
                )
                found = True

        if not found:
            lines.append("Không tìm thấy mẫu hình nến (Pin Bar/Engulfing) rõ ràng trong chart này.")
        return "\n".join(lines).strip() + "\n"

    # ── Render ─────────────────────────────────────────────────────
    def _render(self, indices: Sequence[int], include_header: bool = True) -> str:
        parts = []
        if include_header:
            parts.append(f"Dữ liệu chart đang phân tích: {self.parser.raw_text.strip()}\n")
        for i in indices:
            parts.append(self._layers[i]())
        return "\n\n".join(p.strip() for p in parts) + "\n"

    def full(self) -> str:
        """Sinh đủ 7 tầng."""
        return self._render(list(range(self.num_layers)))

    def random_subset(
        self,
        min_layers: int = 2,
        max_layers: int = None,
        rng: random.Random = None,
    ) -> str:
        """Sinh tổ hợp tầng ngẫu nhiên (subset rời rạc hoặc range liên tục)."""
        rng        = rng or random
        max_layers = min(max_layers or self.num_layers, self.num_layers)
        min_layers = max(1, min(min_layers, max_layers))

        if rng.choice(["subset", "range"]) == "range":
            length  = rng.randint(min_layers, max_layers)
            start   = rng.randint(0, max(0, self.num_layers - length))
            indices = list(range(start, start + length))
        else:
            k       = rng.randint(min_layers, max_layers)
            indices = sorted(rng.sample(range(self.num_layers), k))

        return self._render(indices)


# ══════════════════════════════════════════════════════════════════════
# CHART PRETRAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════

def _parquet_schema():
    return pa.schema([
        ("text",         pa.string()),
        ("source",       pa.string()),
        ("token_length", pa.int64()),
        ("meta",         pa.string()),
    ])


class ChartPretrainPipeline:
    """
    Pipeline chính: raw chart text → records → parquet.

    Mỗi chart dài (vd 200 nến) được cắt thành nhiều đoạn con ngẫu nhiên
    (20-30 nến), mỗi đoạn con sinh text curriculum theo mode chọn.
    """

    def __init__(
        self,
        tokenizer,
        source_name     : str  = "unknown",
        slices_per_chart: int  = 4,
        min_slice_len   : int  = 20,
        max_slice_len   : int  = 30,
        swing_window    : int  = 2,
        curriculum_mode : str  = "random",   # "full" | "random"
        min_layers      : int  = 2,
        max_layers      : int  = None,
        seed            : int  = None,
    ):
        self.tokenizer        = tokenizer
        self.source_name      = source_name
        self.slices_per_chart = slices_per_chart
        self.min_slice_len    = min_slice_len
        self.max_slice_len    = max_slice_len
        self.swing_window     = swing_window
        self.curriculum_mode  = curriculum_mode
        self.min_layers       = min_layers
        self.max_layers       = max_layers
        self.rng              = random.Random(seed) if seed is not None else random

    # ── Gen records từ 1 batch chart text ─────────────────────────
    def gen_records(
        self,
        raw_charts          : List[str],
        chart_index_offset  : int = 0,
    ) -> List[dict]:
        records = []

        for local_idx, raw in enumerate(raw_charts):
            chart_idx   = chart_index_offset + local_idx
            base_parser = CandleParser(raw, swing_window=self.swing_window)
            n           = len(base_parser)

            # Cắt đoạn con
            if n < self.min_slice_len:
                sub_ranges = [(0, n)]
            else:
                sub_ranges = []
                for _ in range(self.slices_per_chart):
                    sl  = self.rng.randint(self.min_slice_len, min(self.max_slice_len, n))
                    st  = self.rng.randint(0, n - sl)
                    sub_ranges.append((st, st + sl))

            for start, end in sub_ranges:
                sub = base_parser.slice(start, end)
                gen = CurriculumGenerator(sub)

                if self.curriculum_mode == "full":
                    text = gen.full()
                    n_layers = gen.num_layers
                else:
                    text = gen.random_subset(
                        min_layers=self.min_layers,
                        max_layers=self.max_layers,
                        rng=self.rng,
                    )
                    n_layers = text.count("===") // 2

                meta = {
                    "source_chart_index": chart_idx,
                    "slice_start"       : start,
                    "slice_end"         : end,
                    "num_candles"       : end - start,
                    "num_layers"        : n_layers,
                }
                
                token_length = len(self.tokenizer.encode(text, add_special_tokens=False))
                records.append({
                    "text"        : text,
                    "source"      : self.source_name,
                    "token_length": token_length,
                    "meta"        : json.dumps(meta, ensure_ascii=False),
                })

        return records

    # ── Build từ parquet (chart token đã encode từ nhánh 2) ───────
    def build_from_parquet(
        self,
        input_path : str,
        output_path: str,
        text_column: str = "text",
        batch_size : int = 2000,
    ) -> int:
        """
        Đọc parquet chart token theo batch → sinh text curriculum → ghi parquet.
        Input thường là output của ChartDatasetBuilder (nhánh 2).
        """
        if not _HAS_PYARROW:
            raise ImportError("Cần cài pyarrow: pip install pyarrow")

        schema = _parquet_schema()
        writer = None
        total  = 0

        try:
            pf = pq.ParquetFile(input_path)
            for batch_idx, batch in enumerate(
                pf.iter_batches(batch_size=batch_size, columns=[text_column])
            ):
                raw_charts = batch.column(text_column).to_pylist()
                records    = self.gen_records(raw_charts, chart_index_offset=batch_idx * batch_size)

                if not records:
                    continue

                table = pa.Table.from_pylist(records, schema=schema)
                if writer is None:
                    writer = pq.ParquetWriter(output_path, schema, compression="snappy")
                writer.write_table(table)

                total += len(records)
                print(f"[batch {batch_idx}] +{len(records)} samples | tổng: {total}")

        finally:
            if writer:
                writer.close()

        print(f"\n✅ Hoàn tất. {total} samples -> {output_path}")
        return total

    # ── Build từ list raw chart text ──────────────────────────────
    def build_from_list(
        self,
        raw_charts : List[str],
        output_path: str,
    ) -> int:
        """Build từ list chart text sẵn có, ghi ra parquet."""
        if not _HAS_PYARROW:
            raise ImportError("Cần cài pyarrow: pip install pyarrow")

        records = self.gen_records(raw_charts)
        if not records:
            print("Không có records nào.")
            return 0

        schema = _parquet_schema()
        table  = pa.Table.from_pylist(records, schema=schema)
        pq.write_table(table, output_path, compression="snappy")
        print(f"✅ {len(records)} samples -> {output_path}")
        return len(records)


# ══════════════════════════════════════════════════════════════════════
# DEMO
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    sample_chart = (
        "<chart> O_512 H_521 L_500 C_500 O_500 H_521 L_500 C_514 "
        "O_514 H_516 L_500 C_510 O_508 H_511 L_499 C_503 "
        "O_502 H_502 L_475 C_491 O_490 H_490 L_478 C_489 "
        "O_494 H_507 L_481 C_481 O_479 H_484 L_473 C_473 </chart>"
    )

    # ── Test CurriculumGenerator ──
    parser = CandleParser(sample_chart, swing_window=2)
    gen    = CurriculumGenerator(parser)

    print("=== FULL (7 tầng) ===\n")
    print(gen.full()[:500], "...\n")

    print("=== RANDOM SUBSET ===\n")
    rng = random.Random(42)
    print(gen.random_subset(min_layers=2, max_layers=4, rng=rng)[:500], "...\n")

    # ── Test pipeline ──
    pipeline = ChartPretrainPipeline(
        source_name     = "demo",
        slices_per_chart= 2,
        curriculum_mode = "random",
        seed            = 42,
    )
    records = pipeline.gen_records([sample_chart] * 5)
    print(f"Gen được {len(records)} records từ 5 chart.")
    print("\n── Record đầu tiên (rút gọn) ──")
    print(records[0]["text"][:300], "...")
    print("meta:", records[0]["meta"])