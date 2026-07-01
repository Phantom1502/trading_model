"""
stats_swept.py — Thống kê Swept (Liquidity Sweep) trên data thật
========================================================================
Chạy 1 lần trên data CSV/parquet thật, in ra:
    1. Tần suất sweep (bao nhiêu % nến thực hiện 1 sweep) theo TỪNG
       lookback khác nhau — để chọn lookback mặc định phù hợp (hiện
       đang hardcode 10 trong is_swept()/scan_all_swept()).
    2. Phân phối `depth` (độ sâu sweep = vượt qua swing level bao
       nhiêu bin) — đây là đại lượng chính dùng làm nền cho Q-score
       graded của Swept (sweep càng sâu, càng đáng tin).
    3. Tỷ lệ SWEEP_HIGH vs SWEEP_LOW — kiểm tra có lệch bất thường
       không (lệch nhiều có thể là dấu hiệu bug hoặc đặc thù data).

Cách dùng:
    python stats_swept.py --parquet data/chart_XAUUSD_dataset_1Min.parquet
    python stats_swept.py --parquet ... --sample 5000   # chạy nhanh thử trước
"""

import argparse
import numpy as np
import pandas as pd

from app.ict.parser import CandleParser, SWING_WINDOW_DEFAULT
from app.ict.ict import scan_all_swept


def compute_swept_stats(df: pd.DataFrame, lookbacks: list[int], text_col: str = "text") -> dict:
    """
    Trả về dict thống kê cho từng lookback trong `lookbacks`:
        {
            lookback: {
                "total_candles": int,
                "n_sweep_high": int,
                "n_sweep_low" : int,
                "depth_high"  : list[int],
                "depth_low"   : list[int],
            }
        }
    """
    results = {lb: {
        "total_candles": 0,
        "n_sweep_high": 0,
        "n_sweep_low": 0,
        "depth_high": [],
        "depth_low": [],
    } for lb in lookbacks}

    for raw_text in df[text_col]:
        parser = CandleParser(raw_text, swing_window=SWING_WINDOW_DEFAULT)
        n = len(parser)

        for lb in lookbacks:
            r = results[lb]
            r["total_candles"] += n

            sweeps = scan_all_swept(parser, lookback=lb)
            for s in sweeps:
                if s["type"] == "SWEEP_HIGH":
                    r["n_sweep_high"] += 1
                    r["depth_high"].append(s["depth"])
                else:
                    r["n_sweep_low"] += 1
                    r["depth_low"].append(s["depth"])

    return results


def print_stats(results: dict):
    for lb, r in results.items():
        total = r["total_candles"]
        n_high, n_low = r["n_sweep_high"], r["n_sweep_low"]
        n_total_sweep = n_high + n_low

        print(f"\n{'='*60}")
        print(f"  lookback = {lb}")
        print(f"{'='*60}")
        print(f"  Tổng nến phân tích : {total:,}")
        print(f"  Sweep High         : {n_high:,} ({n_high/total*100:.2f}%)")
        print(f"  Sweep Low          : {n_low:,} ({n_low/total*100:.2f}%)")
        print(f"  Tổng Sweep         : {n_total_sweep:,} ({n_total_sweep/total*100:.2f}%)")
        if n_total_sweep > 0:
            print(f"  Tỷ lệ High/Low     : {n_high/n_total_sweep*100:.1f}% / {n_low/n_total_sweep*100:.1f}%")

        if r["depth_high"]:
            d_high = np.array(r["depth_high"])
            print(f"\n  Depth Sweep High (bin):")
            for p in [10, 25, 50, 75, 90, 95, 99]:
                print(f"    p{p}: {np.percentile(d_high, p):.1f}")

        if r["depth_low"]:
            d_low = np.array(r["depth_low"])
            print(f"\n  Depth Sweep Low (bin):")
            for p in [10, 25, 50, 75, 90, 95, 99]:
                print(f"    p{p}: {np.percentile(d_low, p):.1f}")


def main():
    parser = argparse.ArgumentParser(description="Thống kê Swept trên data thật")
    parser.add_argument("--parquet", type=str, required=True, help="Đường dẫn file parquet chart dataset")
    parser.add_argument("--text-col", type=str, default="text")
    parser.add_argument(
        "--lookbacks", type=int, nargs="+", default=[5, 10, 15, 20],
        help="Danh sách lookback cần thống kê (default: 5 10 15 20)",
    )
    parser.add_argument("--sample", type=int, default=None, help="Chỉ lấy N dòng đầu để chạy nhanh thử")
    args = parser.parse_args()

    df = pd.read_parquet(args.parquet)
    if args.sample:
        df = df.head(args.sample)

    print(f"Đang chạy trên {len(df):,} chart, lookbacks={args.lookbacks}...")
    results = compute_swept_stats(df, args.lookbacks, text_col=args.text_col)
    print_stats(results)


if __name__ == "__main__":
    main()