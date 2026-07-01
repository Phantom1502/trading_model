import pandas as pd
import numpy as np
from app.ict.parser import CandleParser
from app.ict.basic import classify_direction, is_pin_bar, DOJI_THRESHOLD_BINS
from app.ict.structure import is_fvg

# Load parquet đã encode (output của ChartDatasetBuilder)
df = pd.read_parquet(r"data\chart_XAUUSD_dataset_1Min.parquet")

gap_sizes = []
for raw_text in df["text"]:
    parser = CandleParser(raw_text)
    for i in range(2, len(parser)):
        fvg_type = is_fvg(parser, i)
        if fvg_type:
            c0, c2 = parser[i-2], parser[i]
            gap = (c2.low - c0.high) if fvg_type == "BULL" else (c0.low - c2.high)
            gap_sizes.append(gap)

gap_arr = np.array(gap_sizes)
print(f"Tổng FVG phát hiện: {len(gap_arr):,}")
for p in [1, 5, 10, 25, 50, 75, 90]:
    print(f"  p{p}: {np.percentile(gap_arr, p):.1f} bin")