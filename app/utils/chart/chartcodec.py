"""
chart_codec.py — Encode / decode chuỗi OHLC <-> token text cho LLM
=====================================================================

Tổ chức thành 2 class:
    ChartCodec          : encode/decode 1 cửa sổ (window) OHLC <-> text
    ChartDatasetBuilder : sinh dataset (DataFrame/parquet) từ giá thô,
                          dùng ChartCodec bên trong cho từng cửa sổ

Ý tưởng pipeline:
    1. Với mỗi thời điểm t, lấy cửa sổ N nến (window_size, ví dụ 100)
       FORWARD từ t -> t+window_size-1 (100 nến kể từ "hiện tại" t).
       Hệ quy chiếu lấy tại nến ĐẦU cửa sổ (chính là t, thời điểm đang
       đứng, không lookahead):
           anchor_open = Open tại t
           anchor_atr  = ATR_period tại t
    2. Mọi giá trong cửa sổ (Open/High/Low/Close của từng nến) được
       chuẩn hoá về [-1, 1]:
           norm = (price - anchor_open) / (SCALE * anchor_atr)
       rồi rời rạc hoá theo FSQ (Finite Scalar Quantization): mỗi
       channel O/H/L/C lượng tử hoá ĐỘC LẬP vào lưới cố định 1024 mức
       (không dùng codebook học như VQ-VAE truyền thống) -> bin nguyên
       trong [0, n_bins-1]. bin=0 <-> norm=-1 (thấp nhất), bin=n_bins-1
       <-> norm=+1 (cao nhất), chia đối xứng 50/50 quanh anchor.
    3. Mỗi nến sinh ra 4 token: O_<bin> H_<bin> L_<bin> C_<bin>, cả cửa
       sổ được bọc trong <chart> ... </chart>.
    4. Token chỉ mang thông tin TƯƠNG ĐỐI theo (anchor_open, anchor_atr)
       -> PHẢI lưu lại 2 giá trị này kèm mỗi mẫu, không thể tự suy ra
       chỉ từ text khi decode.
"""
import re
import numpy as np
import pandas as pd

N_BINS = 1024
_TOKEN_RE = re.compile(r"([OHLC])_(\d+)")

# Các loại scale factor có thể dùng để chuẩn hoá giá trong cửa sổ OHLC
M1_SCALE = 24.0      # Window Range = M1_SCALE * ATR
M5_SCALE = 27.38      # Window Range = M5_SCALE * ATR
M15_SCALE = 30.0     # Window Range = M15_SCALE * ATR
H1_SCALE = 28.19     # Window Range = H1_SCALE * ATR
D1_SCALE = 17.92     # Window Range = D1_SCALE * ATR

# ──────────────────────────────────────────────────────────────────────
# Hàm tiện ích độc lập: ATR
# ──────────────────────────────────────────────────────────────────────
def calculate_atr(df: pd.DataFrame, period: int = 100) -> np.ndarray:
    """Tính chỉ báo ATR chuẩn kỹ thuật (EMA-smoothed)."""
    high_low = df["High"] - df["Low"]
    high_cp = np.abs(df["High"] - df["Close"].shift(1))
    low_cp = np.abs(df["Low"] - df["Close"].shift(1))

    tr = np.max(np.vstack((high_low, high_cp, low_cp)), axis=0)
    atr = pd.Series(tr).ewm(span=period, adjust=False).mean().values
    return atr


# ──────────────────────────────────────────────────────────────────────
# Class 1: ChartCodec — encode/decode 1 cửa sổ OHLC <-> text token
# ──────────────────────────────────────────────────────────────────────
class ChartCodec:
    """
    FSQ codec cho 1 cửa sổ OHLC: số thực <-> token text.

    scale, n_bins cố định cho cả đời codec (set 1 lần ở __init__), nên
    gọi encode_window/decode_window không cần truyền lại mỗi lần.
    """

    def __init__(self, scale: float, n_bins: int = N_BINS):
        self.scale = scale
        self.n_bins = n_bins

    # ---- quantize / dequantize 1 giá trị ----
    def quantize_price(self, price, anchor_open, anchor_atr) -> int:
        """FSQ: price thật -> bin nguyên [0, n_bins-1] trên lưới cố định.

        bin=0 <-> norm=-1 (thấp nhất), bin=n_bins-1 <-> norm=+1 (cao nhất).
        """
        if anchor_atr <= 0 or np.isnan(anchor_atr):
            raise ValueError("anchor_atr phải > 0")
        norm = (price - anchor_open) / (self.scale * anchor_atr)
        norm = np.clip(norm, -1.0, 1.0)
        bin_idx = int(round((norm + 1.0) / 2.0 * (self.n_bins - 1)))
        return bin_idx

    def dequantize_bin(self, bin_idx, anchor_open, anchor_atr) -> float:
        """bin nguyên -> giá thật (xấp xỉ, sai số tối đa = nửa bin)."""
        norm = (bin_idx / (self.n_bins - 1)) * 2.0 - 1.0
        return anchor_open + norm * self.scale * anchor_atr

    # ---- encode / decode 1 cửa sổ ----
    def encode_window(self, window_df: pd.DataFrame, anchor_open, anchor_atr) -> str:
        """N nến (Open/High/Low/Close) -> '<chart> O_.. H_.. L_.. C_.. ... </chart>'."""
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
        """
        Chuỗi token -> DataFrame Open/High/Low/Close (giá xấp xỉ).

        Gom token theo NHÃN CHỮ CÁI (O/H/L/C), không phụ thuộc khoảng
        trắng/định dạng chính xác -> chịu được nhiễu nhẹ khi parse text
        do model generate ra (miễn giữ đúng thứ tự O,H,L,C lặp mỗi nến).
        """
        buckets = {"O": [], "H": [], "L": [], "C": []}
        for letter, num in _TOKEN_RE.findall(text):
            buckets[letter].append(int(num))

        n_candles = len(buckets["O"])
        if not all(len(buckets[k]) == n_candles for k in "HLC"):
            raise ValueError(
                f"Số token O/H/L/C không khớp nhau: "
                f"O={len(buckets['O'])} H={len(buckets['H'])} "
                f"L={len(buckets['L'])} C={len(buckets['C'])} "
                f"— text có thể bị model sinh lỗi/thiếu token."
            )

        rows = []
        for i in range(n_candles):
            rows.append({
                "Open":  self.dequantize_bin(buckets["O"][i], anchor_open, anchor_atr),
                "High":  self.dequantize_bin(buckets["H"][i], anchor_open, anchor_atr),
                "Low":   self.dequantize_bin(buckets["L"][i], anchor_open, anchor_atr),
                "Close": self.dequantize_bin(buckets["C"][i], anchor_open, anchor_atr),
            })
        return pd.DataFrame(rows)