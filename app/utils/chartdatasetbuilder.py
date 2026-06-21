from chartcodec import ChartCodec, calculate_atr, N_BINS, M1_SCALE, M5_SCALE, M15_SCALE, H1_SCALE, D1_SCALE
import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────
# Class 2: ChartDatasetBuilder — sinh dataset text từ DataFrame giá thô
# ──────────────────────────────────────────────────────────────────────
class ChartDatasetBuilder:
    """
    Quét toàn bộ lịch sử giá theo cửa sổ trượt, dùng ChartCodec để encode
    từng cửa sổ thành text, trả về DataFrame (hoặc lưu thẳng ra parquet).
    """

    def __init__(self, codec: ChartCodec, window_size: int = 100,
                 stride: int = 10, atr_period: int = 100):
        self.codec = codec
        self.window_size = window_size
        self.stride = stride
        self.atr_period = atr_period

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        df cần có cột Open/High/Low/Close, 1 dòng = 1 nến, đã sort theo
        thời gian.

        Convention: tại mỗi thời điểm t, lấy cửa sổ FORWARD gồm
        window_size nến từ t đến t+window_size-1. Anchor lấy tại CHÍNH
        t (nến đầu cửa sổ) -> không lookahead bias.

        Trả về DataFrame:
            start_idx    : index nến neo (= t) trong df gốc
            end_idx      : index nến cuối cửa sổ (t + window_size - 1)
            anchor_open  : giá Open tại t — bắt buộc lưu để decode
            anchor_atr   : ATR tại t — bắt buộc lưu để decode
            text         : chuỗi token — dùng trực tiếp cho
                            tokenizer.encode_batch(df['text'].tolist())
        """
        df = df.reset_index(drop=True).copy()
        df["__atr__"] = calculate_atr(df, period=self.atr_period)
        # KHÔNG dropna+reset_index ở đây — sẽ làm lệch start_idx/end_idx
        # khỏi vị trí thật trong df gốc. Skip trực tiếp trong loop bên dưới.

        records = []
        last_start = len(df) - self.window_size
        for t in range(0, last_start + 1, self.stride):
            anchor_open = df.loc[t, "Open"]
            anchor_atr = df.loc[t, "__atr__"]
            if anchor_atr <= 0 or np.isnan(anchor_atr):
                continue

            window = df.iloc[t: t + self.window_size]
            text = self.codec.encode_window(window, anchor_open, anchor_atr)

            records.append({
                "start_idx": t,
                "end_idx": t + self.window_size - 1,
                "anchor_open": anchor_open,
                "anchor_atr": anchor_atr,
                "text": text,
            })

        return pd.DataFrame(records)

    def build_from_file(self, input_path: str, output_parquet_path: str,
                         csv_kwargs: dict = None) -> pd.DataFrame:
        """Đọc CSV giá thô -> build() -> lưu parquet."""
        csv_kwargs = csv_kwargs or {}
        df = pd.read_csv(input_path, **csv_kwargs)
        out = self.build(df)
        out.to_parquet(output_parquet_path, index=False)
        print(f"Đã lưu {len(out)} mẫu -> {output_parquet_path}")
        return out
    

# ──────────────────────────────────────────────────────────────────────
# Demo + sanity-check roundtrip (chạy: python chart_codec.py)
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)
    mock_length = 5000
    close_prices = 60000 + np.cumsum(np.random.normal(0, 150, mock_length))
    df_demo = pd.DataFrame({
        "Open":  close_prices - np.random.randint(50, 200, mock_length),
        "High":  close_prices + np.random.randint(100, 400, mock_length),
        "Low":   close_prices - np.random.randint(100, 400, mock_length),
        "Close": close_prices,
    })

    SCALE = M1_SCALE
    WINDOW = 100

    codec = ChartCodec(scale=SCALE)
    builder = ChartDatasetBuilder(codec, window_size=WINDOW, stride=50, atr_period=100)

    dataset = builder.build(df_demo)
    print(f"Sinh được {len(dataset)} mẫu text.")
    print("Mẫu đầu tiên (rút gọn):", dataset.iloc[0]["text"][:120], "...")
    print()

    # Roundtrip check trên mẫu đầu tiên
    sample = dataset.iloc[0]
    decoded = codec.decode_window(sample["text"], sample["anchor_open"], sample["anchor_atr"])

    df_demo["__atr__"] = calculate_atr(df_demo, period=100)
    t0 = int(sample["start_idx"])
    original = (
        df_demo.iloc[t0: t0 + WINDOW][["Open", "High", "Low", "Close"]]
        .reset_index(drop=True)
    )

    err = (decoded - original).abs()
    print("Sai số tuyệt đối lớn nhất sau roundtrip (đơn vị giá):")
    print(err.max())
    print(f"Sai số tương đối lớn nhất theo ATR: {err.max().max() / sample['anchor_atr']:.4f}")
    print(f"(Resolution lý thuyết 1 bin ~ {2 * SCALE / (N_BINS - 1):.4f} x ATR)")