"""
data_pipeline.py
================
Tổng hợp toàn bộ pipeline xử lý data cho trading LLM:

    1. ChartCodec          — encode/decode OHLC <-> token text
    2. ChartDatasetBuilder — sinh dataset chart token từ CSV/DataFrame
    3. ActionDataGen       — sinh dataset (chart, action, sl, tp, result)
    4. PDFDataGen          — trích đoạn văn từ kho PDF trading
    5. ParquetUtils        — merge parquet files

Cách dùng nhanh:
    from data_pipeline import ChartCodec, ChartDatasetBuilder, ActionDataGen
    from data_pipeline import PDFDataGen, ParquetUtils
    from data_pipeline import M1_SCALE, M5_SCALE, H1_SCALE
"""

import re
import json
import os
import glob
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Iterator, List, Optional

import numpy as np
import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    _HAS_PYARROW = True
except ImportError:
    _HAS_PYARROW = False

try:
    import pdfplumber
    _HAS_PDFPLUMBER = True
except ImportError:
    _HAS_PDFPLUMBER = False


# ══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════

N_BINS    = 1024
M1_SCALE  = 24.0
M5_SCALE  = 27.38
M15_SCALE = 30.0
H1_SCALE  = 28.19
D1_SCALE  = 17.92

_TOKEN_RE = re.compile(r"([OHLC])_(\d+)")


# ══════════════════════════════════════════════════════════════════════
# 1. CHART CODEC
# ══════════════════════════════════════════════════════════════════════

def calculate_atr(df: pd.DataFrame, period: int = 100) -> np.ndarray:
    """Tính ATR chuẩn kỹ thuật (EMA-smoothed)."""
    high_low = df["High"] - df["Low"]
    high_cp  = np.abs(df["High"] - df["Close"].shift(1))
    low_cp   = np.abs(df["Low"]  - df["Close"].shift(1))
    tr  = np.max(np.vstack((high_low, high_cp, low_cp)), axis=0)
    atr = pd.Series(tr).ewm(span=period, adjust=False).mean().values
    return atr


class ChartCodec:
    """
    FSQ codec: OHLC số thực <-> token text (O_bin H_bin L_bin C_bin).

    Mỗi giá được normalize về [-1, 1] theo anchor_open + anchor_atr,
    rồi discretize vào N_BINS mức cố định.
    """

    def __init__(self, scale: float, n_bins: int = N_BINS):
        self.scale  = scale
        self.n_bins = n_bins

    def quantize_price(self, price, anchor_open, anchor_atr) -> int:
        if anchor_atr <= 0 or np.isnan(anchor_atr):
            raise ValueError("anchor_atr phải > 0")
        norm    = (price - anchor_open) / (self.scale * anchor_atr)
        norm    = np.clip(norm, -1.0, 1.0)
        bin_idx = int(round((norm + 1.0) / 2.0 * (self.n_bins - 1)))
        return bin_idx

    def dequantize_bin(self, bin_idx, anchor_open, anchor_atr) -> float:
        norm = (bin_idx / (self.n_bins - 1)) * 2.0 - 1.0
        return anchor_open + norm * self.scale * anchor_atr

    def encode_window(self, window_df: pd.DataFrame, anchor_open, anchor_atr) -> str:
        """N nến -> '<chart> O_.. H_.. L_.. C_.. ... </chart>'."""
        parts = ["<chart>"]
        for _, row in window_df.iterrows():
            o = self.quantize_price(row["Open"],  anchor_open, anchor_atr)
            h = self.quantize_price(row["High"],  anchor_open, anchor_atr)
            l = self.quantize_price(row["Low"],   anchor_open, anchor_atr)
            c = self.quantize_price(row["Close"], anchor_open, anchor_atr)
            parts.extend([f"O_{o}", f"H_{h}", f"L_{l}", f"C_{c}"])
        parts.append("</chart>")
        return " ".join(parts)

    def decode_window(self, text: str, anchor_open, anchor_atr) -> pd.DataFrame:
        """Token text -> DataFrame Open/High/Low/Close (giá xấp xỉ)."""
        buckets = {"O": [], "H": [], "L": [], "C": []}
        for letter, num in _TOKEN_RE.findall(text):
            buckets[letter].append(int(num))

        n = len(buckets["O"])
        if not all(len(buckets[k]) == n for k in "HLC"):
            raise ValueError("Số token O/H/L/C không khớp nhau.")

        rows = []
        for i in range(n):
            rows.append({
                "Open":  self.dequantize_bin(buckets["O"][i], anchor_open, anchor_atr),
                "High":  self.dequantize_bin(buckets["H"][i], anchor_open, anchor_atr),
                "Low":   self.dequantize_bin(buckets["L"][i], anchor_open, anchor_atr),
                "Close": self.dequantize_bin(buckets["C"][i], anchor_open, anchor_atr),
            })
        return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════
# 2. CHART DATASET BUILDER
# ══════════════════════════════════════════════════════════════════════

class ChartDatasetBuilder:
    """
    Quét toàn bộ lịch sử giá theo cửa sổ trượt, encode từng cửa sổ
    thành token text, lưu ra DataFrame hoặc parquet.
    """

    def __init__(
        self,
        codec       : ChartCodec,
        window_size : int = 200,
        stride      : int = 10,
        atr_period  : int = 100,
    ):
        self.codec       = codec
        self.window_size = window_size
        self.stride      = stride
        self.atr_period  = atr_period

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        df cần có cột Open/High/Low/Close, sort theo thời gian.

        Trả về DataFrame: start_idx, end_idx, anchor_open, anchor_atr, text.
        """
        df = df.reset_index(drop=True).copy()
        df["__atr__"] = calculate_atr(df, period=self.atr_period)

        records    = []
        last_start = len(df) - self.window_size

        for t in range(0, last_start + 1, self.stride):
            anchor_open = df.loc[t, "Open"]
            anchor_atr  = df.loc[t, "__atr__"]
            if anchor_atr <= 0 or np.isnan(anchor_atr):
                continue

            window = df.iloc[t : t + self.window_size]
            text   = self.codec.encode_window(window, anchor_open, anchor_atr)

            records.append({
                "start_idx"  : t,
                "end_idx"    : t + self.window_size - 1,
                "anchor_open": anchor_open,
                "anchor_atr" : anchor_atr,
                "text"       : text,
            })

        return pd.DataFrame(records)

    def build_from_file(
        self,
        input_path         : str,
        output_parquet_path: str,
        csv_kwargs         : dict = None,
    ) -> pd.DataFrame:
        """Đọc CSV -> build() -> lưu parquet."""
        csv_kwargs = csv_kwargs or {}
        df  = pd.read_csv(input_path, **csv_kwargs)
        out = self.build(df)
        out.to_parquet(output_parquet_path, index=False)
        print(f"Đã lưu {len(out)} mẫu -> {output_parquet_path}")
        return out

    def analyze_range(self, df: pd.DataFrame, forward_size: int = 60) -> dict:
        """
        Phân tích range thực tế của forward_size candle tiếp theo.
        Dùng để calibrate SL/TP bins phù hợp với asset.
        """
        df = df.reset_index(drop=True).copy()
        df["__atr__"] = calculate_atr(df, period=self.atr_period)

        moves = []
        for t in range(self.window_size, len(df) - forward_size, self.stride):
            anchor_open = df.loc[t, "Open"]
            anchor_atr  = df.loc[t, "__atr__"]
            if anchor_atr <= 0 or np.isnan(anchor_atr):
                continue
            forward  = df.iloc[t : t + forward_size]
            high_bin = self.codec.quantize_price(forward["High"].max(), anchor_open, anchor_atr)
            low_bin  = self.codec.quantize_price(forward["Low"].min(),  anchor_open, anchor_atr)
            moves.append(high_bin - low_bin)

        return {
            "mean"  : float(np.mean(moves)),
            "median": float(np.median(moves)),
            "p25"   : float(np.percentile(moves, 25)),
            "p75"   : float(np.percentile(moves, 75)),
        }


# ══════════════════════════════════════════════════════════════════════
# 3. ACTION DATA GEN
# ══════════════════════════════════════════════════════════════════════

class Action(Enum):
    BUY_25   = "buy_25"
    BUY_50   = "buy_50"
    BUY_100  = "buy_100"
    SELL_25  = "sell_25"
    SELL_50  = "sell_50"
    SELL_100 = "sell_100"

TRADE_ACTIONS = list(Action)
LONG_ACTIONS  = {Action.BUY_25, Action.BUY_50, Action.BUY_100}


class ActionDataGen:
    """
    Sinh dataset (chart_text, action, sl, tp) -> result + score.

    Mỗi window sinh N = len(actions) x len(sl_bins) x len(tp_bins) samples,
    đã lọc những combo RR < min_rr.
    """

    def __init__(
        self,
        codec       : ChartCodec,
        window_size : int  = 200,
        forward_size: int  = 60,
        spread_bins : int  = 1,
        sl_bins     : List[int] = None,
        tp_bins     : List[int] = None,
        min_rr      : float     = 1.0,
        atr_period  : int  = 100,
    ):
        self.codec        = codec
        self.window_size  = window_size
        self.forward_size = forward_size
        self.spread_bins  = spread_bins
        self.sl_bins      = sl_bins or [-20, -40, -60, -80]
        self.tp_bins      = tp_bins or [+40, +80, +120, +160]
        self.min_rr       = min_rr
        self.atr_period   = atr_period

    # ── Score ────────────────────────────────────────────────────
    def _compute_score(
        self,
        exit_type   : str,
        pnl_bins    : int,
        max_dd_bins : int,
        candle_exit : int,
    ) -> float:
        if exit_type == "sl_hit":
            return 0.0

        if exit_type == "timeout":
            return 3.0 if pnl_bins > 0 else 1.0

        if exit_type == "tp_hit":
            speed = 1.0 - (candle_exit / self.forward_size)
            clean = 1.0 - (abs(max_dd_bins) / max(abs(min(self.sl_bins)), 1))
            clean = max(clean, 0.0)
            score = 5.0 + speed * 2.5 + clean * 2.5
            return round(min(score, 10.0), 1)

        return 0.0

    # ── Simulate 1 trade (intracandle SL/TP check) ───────────────
    def _simulate(
        self,
        forward_df  : pd.DataFrame,
        entry_bin   : int,
        action      : Action,
        sl_delta    : int,
        tp_delta    : int,
        anchor_open : float,
        anchor_atr  : float,
    ) -> dict:
        is_long = action in LONG_ACTIONS

        if is_long:
            effective_entry = entry_bin + self.spread_bins
            sl_bin = effective_entry + sl_delta   # sl_delta âm
            tp_bin = effective_entry + tp_delta   # tp_delta dương
        else:
            effective_entry = entry_bin - self.spread_bins
            sl_bin = effective_entry - sl_delta
            tp_bin = effective_entry - tp_delta

        max_dd_bins = 0
        exit_type   = "timeout"
        exit_bin    = self.codec.quantize_price(
            forward_df.iloc[-1]["Close"], anchor_open, anchor_atr
        )
        candle_exit = self.forward_size

        for i, row in forward_df.iterrows():
            high_bin = self.codec.quantize_price(row["High"], anchor_open, anchor_atr)
            low_bin  = self.codec.quantize_price(row["Low"],  anchor_open, anchor_atr)

            if is_long:
                dd = low_bin - effective_entry
                if dd < max_dd_bins:
                    max_dd_bins = dd
                if low_bin <= sl_bin:
                    exit_type, exit_bin, candle_exit = "sl_hit", sl_bin, i + 1
                    break
                if high_bin >= tp_bin:
                    exit_type, exit_bin, candle_exit = "tp_hit", tp_bin, i + 1
                    break
            else:
                dd = effective_entry - high_bin
                if dd < max_dd_bins:
                    max_dd_bins = dd
                if high_bin >= sl_bin:
                    exit_type, exit_bin, candle_exit = "sl_hit", sl_bin, i + 1
                    break
                if low_bin <= tp_bin:
                    exit_type, exit_bin, candle_exit = "tp_hit", tp_bin, i + 1
                    break

        pnl_bins = (exit_bin - effective_entry) if is_long else (effective_entry - exit_bin)
        score    = self._compute_score(exit_type, pnl_bins, max_dd_bins, candle_exit)

        return {
            "exit_type"  : exit_type,
            "candle_exit": candle_exit,
            "pnl_bins"   : pnl_bins,
            "max_dd_bins": max_dd_bins,
            "score"      : score,
        }

    # ── Format 1 sample ──────────────────────────────────────────
    @staticmethod
    def _format(chart_text, action, sl_delta, tp_delta, result) -> str:
        return (
            f"{chart_text}\n"
            f"<action>{action.value}</action>\n"
            f"<sl>{sl_delta}</sl>\n"
            f"<tp>{tp_delta}</tp>\n"
            f"<result>{result['exit_type']} | "
            f"candle:{result['candle_exit']} | "
            f"pnl_bins:{result['pnl_bins']:+d} | "
            f"max_dd_bins:{result['max_dd_bins']:+d} | "
            f"score:{result['score']}</result>"
        )

    # ── Gen dataset ───────────────────────────────────────────────
    def gen(self, df: pd.DataFrame, stride: int = 10) -> List[str]:
        """
        Quét df, mỗi window sinh samples cho mọi combo action × sl × tp.
        Entry tại open candle tiếp theo (sau window) — không lookahead.
        """
        df = df.reset_index(drop=True).copy()
        if "__atr__" not in df.columns:
            df["__atr__"] = calculate_atr(df, period=self.atr_period)

        samples    = []
        last_start = len(df) - self.window_size - self.forward_size

        for t in range(0, last_start + 1, stride):
            anchor_open = df.loc[t, "Open"]
            anchor_atr  = df.loc[t, "__atr__"]
            if anchor_atr <= 0 or np.isnan(anchor_atr):
                continue

            window_df  = df.iloc[t : t + self.window_size]
            chart_text = self.codec.encode_window(window_df, anchor_open, anchor_atr)

            entry_open = df.loc[t + self.window_size, "Open"]
            entry_bin  = self.codec.quantize_price(entry_open, anchor_open, anchor_atr)

            forward_df = df.iloc[
                t + self.window_size : t + self.window_size + self.forward_size
            ].reset_index(drop=True)

            for action in TRADE_ACTIONS:
                for sl in self.sl_bins:
                    for tp in self.tp_bins:
                        if tp < abs(sl) * self.min_rr:
                            continue
                        result = self._simulate(
                            forward_df, entry_bin, action,
                            sl, tp, anchor_open, anchor_atr,
                        )
                        samples.append(self._format(chart_text, action, sl, tp, result))

        return samples

    # ── Balance 1:1:1 ────────────────────────────────────────────
    @staticmethod
    def balance(samples: List[str], seed: int = None) -> List[str]:
        """Lọc cân bằng 1:1:1 theo tp_hit / sl_hit / timeout."""
        if seed is not None:
            random.seed(seed)

        buckets: dict = defaultdict(list)
        for s in samples:
            if "tp_hit"  in s: buckets["tp_hit"].append(s)
            elif "sl_hit" in s: buckets["sl_hit"].append(s)
            else:               buckets["timeout"].append(s)

        min_count = min(len(v) for v in buckets.values())
        balanced  = []
        for k in buckets:
            balanced.extend(random.sample(buckets[k], min_count))
        random.shuffle(balanced)
        return balanced

    # ── Distribution check ────────────────────────────────────────
    @staticmethod
    def distribution(samples: List[str]) -> dict:
        """Đếm tỉ lệ exit_type trong danh sách samples."""
        from collections import Counter
        exits = [s.split("result>")[1].split(" |")[0] for s in samples]
        total = len(samples)
        return {k: {"count": v, "pct": round(v / total * 100, 1)}
                for k, v in Counter(exits).items()}


# ══════════════════════════════════════════════════════════════════════
# 4. PDF DATA GEN
# ══════════════════════════════════════════════════════════════════════

def _split_page_into_paragraphs(
    page,
    gap_threshold_ratio: float = 1.8,
) -> List[str]:
    """Tách 1 trang PDF thành đoạn văn dựa trên khoảng cách dọc."""
    if not _HAS_PDFPLUMBER:
        raise ImportError("Cần cài pdfplumber: pip install pdfplumber")

    lines = page.extract_text_lines()
    if not lines:
        return []
    if len(lines) == 1:
        text = lines[0]["text"].strip()
        return [text] if text else []

    gaps         = [lines[i]["top"] - lines[i - 1]["bottom"] for i in range(1, len(lines))]
    positive_gap = [g for g in gaps if g > 0]
    median_gap   = statistics.median(positive_gap) if positive_gap else 1.0
    threshold    = max(median_gap * gap_threshold_ratio, median_gap + 2.0)

    paragraphs: List[str] = []
    current: List[str] = [lines[0]["text"]]

    for i in range(1, len(lines)):
        if lines[i]["top"] - lines[i - 1]["bottom"] > threshold:
            joined = re.sub(r"\s+", " ", " ".join(current)).strip()
            if joined:
                paragraphs.append(joined)
            current = [lines[i]["text"]]
        else:
            current.append(lines[i]["text"])

    joined = re.sub(r"\s+", " ", " ".join(current)).strip()
    if joined:
        paragraphs.append(joined)
    return paragraphs


def _is_likely_noise(paragraph: str, min_len: int = 80) -> bool:
    if len(paragraph) < min_len:
        return True
    alpha = sum(1 for ch in paragraph if ch.isalpha())
    return alpha / max(len(paragraph), 1) < 0.5


class PDFDataGen:
    """Trích đoạn văn từ kho PDF trading (text-based), ghi ra Parquet."""

    def __init__(self, min_paragraph_len: int = 80):
        if not _HAS_PDFPLUMBER:
            raise ImportError("Cần cài pdfplumber: pip install pdfplumber")
        self.min_paragraph_len = min_paragraph_len

    def extract_from_file(self, pdf_path: str, source_name: str = None) -> List[dict]:
        """Trích đoạn văn từ 1 file PDF, trả về list record."""
        source_name = source_name or os.path.basename(pdf_path)
        records: List[dict] = []
        para_idx = 0

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                for para in _split_page_into_paragraphs(page):
                    if _is_likely_noise(para, self.min_paragraph_len):
                        continue
                    meta = {
                        "page_number"    : page_num,
                        "paragraph_index": para_idx,
                        "char_length"    : len(para),
                    }
                    records.append({
                        "text"        : para,
                        "source"      : source_name,
                        "token_length": 0,
                        "meta"        : json.dumps(meta, ensure_ascii=False),
                    })
                    para_idx += 1
        return records

    def build_to_parquet(
        self,
        input_dir  : str,
        output_path: str,
        pattern    : str = "*.pdf",
    ) -> int:
        """Xử lý kho PDF, ghi append ra parquet. Trả về tổng số đoạn văn."""
        if not _HAS_PYARROW:
            raise ImportError("Cần cài pyarrow: pip install pyarrow")

        schema = pa.schema([
            ("text",         pa.string()),
            ("source",       pa.string()),
            ("token_length", pa.int64()),
            ("meta",         pa.string()),
        ])
        writer: Optional[pq.ParquetWriter] = None
        total = 0

        try:
            for path in sorted(glob.glob(os.path.join(input_dir, "**", pattern), recursive=True)):
                try:
                    records = self.extract_from_file(path)
                except Exception as e:
                    print(f"⚠️  Bỏ qua {path}: {e}")
                    continue
                if not records:
                    print(f"⚠️  {path}: không có đoạn văn (có thể là PDF scan ảnh).")
                    continue
                table = pa.Table.from_pylist(records, schema=schema)
                if writer is None:
                    writer = pq.ParquetWriter(output_path, schema, compression="snappy")
                writer.write_table(table)
                total += len(records)
                print(f"[{os.path.basename(path)}] +{len(records)} đoạn | tổng: {total}")
        finally:
            if writer:
                writer.close()

        print(f"\n✅ Hoàn tất. {total} đoạn văn -> {output_path}")
        return total


# ══════════════════════════════════════════════════════════════════════
# 5. PARQUET UTILS
# ══════════════════════════════════════════════════════════════════════

class ParquetUtils:
    """Tiện ích đọc/ghi/merge file Parquet."""

    @staticmethod
    def merge(
        input_dir        : str,
        output_dir       : str,
        target_size_mb   : int = 500,
        pattern          : str = "*.parquet",
        compression      : str = "ZSTD",
    ) -> None:
        """Gom nhiều file parquet nhỏ thành file lớn ~target_size_mb."""
        if not _HAS_PYARROW:
            raise ImportError("Cần cài pyarrow: pip install pyarrow")

        os.makedirs(output_dir, exist_ok=True)
        file_list = sorted(glob.glob(os.path.join(input_dir, pattern)))
        if not file_list:
            print(f"Không tìm thấy file Parquet trong: {input_dir}")
            return

        print(f"Tìm thấy {len(file_list)} file.")
        target_bytes = target_size_mb * 1024 * 1024
        schema       = pq.ParquetFile(file_list[0]).schema.to_arrow_schema()
        writer       = None
        current_size = 0
        file_idx     = 1

        def _new_path():
            nonlocal file_idx
            p = os.path.join(output_dir, f"part_{file_idx:04d}.parquet")
            file_idx += 1
            return p

        try:
            for path in file_list:
                size = os.path.getsize(path)
                if writer and current_size + size > target_bytes:
                    writer.close()
                    writer       = None
                    current_size = 0

                if writer is None:
                    out = _new_path()
                    print(f"Tạo file mới: {out}")
                    writer = pq.ParquetWriter(out, schema, compression=compression)

                pf = pq.ParquetFile(path)
                for rg in range(pf.num_row_groups):
                    writer.write_table(pf.read_row_group(rg))
                current_size += size
        finally:
            if writer:
                writer.close()
        print("✅ Merge hoàn tất.")

    @staticmethod
    def iter_batches(
        input_path : str,
        text_column: str = "text",
        batch_size : int = 2000,
    ) -> Iterator[List[str]]:
        """Đọc parquet theo batch, yield list string mỗi lần."""
        if not _HAS_PYARROW:
            raise ImportError("Cần cài pyarrow: pip install pyarrow")
        pf = pq.ParquetFile(input_path)
        for batch in pf.iter_batches(batch_size=batch_size, columns=[text_column]):
            yield batch.column(text_column).to_pylist()


# ══════════════════════════════════════════════════════════════════════
# DEMO
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("Demo data_pipeline.py")
    print("=" * 60)

    # ── Chart codec roundtrip ──
    np.random.seed(42)
    n     = 500
    close = 2000 + np.cumsum(np.random.normal(0, 1, n))
    df    = pd.DataFrame({
        "Open" : close - np.random.uniform(0, 0.5, n),
        "High" : close + np.random.uniform(0, 1.0, n),
        "Low"  : close - np.random.uniform(0, 1.0, n),
        "Close": close,
    })

    codec   = ChartCodec(scale=M1_SCALE)
    builder = ChartDatasetBuilder(codec, window_size=50, stride=10)
    ds      = builder.build(df)
    print(f"\n[ChartDatasetBuilder] {len(ds)} windows")

    # ── Analyze range ──
    stats = builder.analyze_range(df, forward_size=60)
    print(f"[Range stats] {stats}")

    # ── Action data gen ──
    gen = ActionDataGen(
        codec,
        window_size=50,
        forward_size=60,
        sl_bins=[-20, -40],
        tp_bins=[+40, +80],
    )
    samples = gen.gen(df, stride=10)
    print(f"\n[ActionDataGen] {len(samples)} samples (raw)")
    print(f"[Distribution] {gen.distribution(samples)}")

    balanced = gen.balance(samples, seed=42)
    print(f"[Balanced]     {len(balanced)} samples")
    print(f"[Distribution] {gen.distribution(balanced)}")

    print("\n── Sample đầu tiên ──")
    print(balanced[0] if balanced else "(trống)")