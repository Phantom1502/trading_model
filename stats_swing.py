"""
stats_swing.py — Thống kê Swing High/Low trên data thật
================================================================
Chạy 1 lần trên data CSV/parquet thật, in ra:
    1. Tần suất swing (bao nhiêu % nến là swing) theo TỪNG swing_window
       khác nhau — để chọn swing_window mặc định phù hợp.
    2. Khoảng cách (số nến) giữa 2 swing liên tiếp cùng loại — input để
       sau này tính lookback hợp lý cho is_swept().
    3. "Prominence" — độ nổi bật của swing so với window xung quanh
       (High[swing] - max(High các nến còn lại trong window) cho swing
       high, đối xứng cho swing low) — đây chính là đại lượng dùng làm
       nền cho Q-score graded của Swing sau này (swing càng nổi bật,
       càng đáng tin).

Cách dùng:
    python stats_swing.py --parquet data/chart_XAUUSD_dataset_1Min.parquet
"""

import argparse
import numpy as np
import pandas as pd

from app.ict.parser import CandleParser
from app.ict.structure import is_swing_high, is_swing_low


def compute_swing_stats(df: pd.DataFrame, swing_windows: list[int], text_col: str = "text") -> dict:
    """
    Trả về dict thống kê cho từng swing_window trong `swing_windows`:
        {
            window: {
                "total_candles": int,
                "n_swing_high": int,
                "n_swing_low" : int,
                "pct_swing_high": float,
                "pct_swing_low" : float,
                "gap_high_bins" : list[int],   # khoảng cách (số nến) giữa 2 swing high liên tiếp
                "gap_low_bins"  : list[int],
                "prominence_high": list[int],  # High[swing] - max(High lân cận trong window)
                "prominence_low" : list[int],  # min(Low lân cận trong window) - Low[swing]
            }
        }
    """
    results = {w: {
        "total_candles": 0,
        "n_swing_high": 0,
        "n_swing_low": 0,
        "gap_high": [],
        "gap_low": [],
        "prominence_high": [],
        "prominence_low": [],
    } for w in swing_windows}

    for raw_text in df[text_col]:
        parser_base = CandleParser(raw_text)
        n = len(parser_base)

        for w in swing_windows:
            parser = CandleParser(raw_text, swing_window=w)
            r = results[w]
            r["total_candles"] += n

            last_high_idx = None
            last_low_idx  = None

            for i in range(n):
                if is_swing_high(parser, i):
                    r["n_swing_high"] += 1
                    if last_high_idx is not None:
                        r["gap_high"].append(i - last_high_idx)
                    last_high_idx = i

                    # prominence: High[i] - max High của các nến khác trong window
                    neighbors = [parser[j].high for j in range(i - w, i + w + 1) if j != i]
                    if neighbors:
                        r["prominence_high"].append(parser[i].high - max(neighbors))

                if is_swing_low(parser, i):
                    r["n_swing_low"] += 1
                    if last_low_idx is not None:
                        r["gap_low"].append(i - last_low_idx)
                    last_low_idx = i

                    neighbors = [parser[j].low for j in range(i - w, i + w + 1) if j != i]
                    if neighbors:
                        r["prominence_low"].append(min(neighbors) - parser[i].low)

    return results


def print_stats(results: dict):
    for w, r in results.items():
        total = r["total_candles"]
        print(f"\n{'='*60}")
        print(f"  swing_window = {w}")
        print(f"{'='*60}")
        print(f"  Tổng nến phân tích : {total:,}")
        print(f"  Swing High         : {r['n_swing_high']:,} ({r['n_swing_high']/total*100:.2f}%)")
        print(f"  Swing Low          : {r['n_swing_low']:,} ({r['n_swing_low']/total*100:.2f}%)")

        if r["gap_high"]:
            gap_h = np.array(r["gap_high"])
            print(f"\n  Khoảng cách giữa 2 Swing High liên tiếp (số nến):")
            for p in [10, 25, 50, 75, 90]:
                print(f"    p{p}: {np.percentile(gap_h, p):.1f}")

        if r["gap_low"]:
            gap_l = np.array(r["gap_low"])
            print(f"\n  Khoảng cách giữa 2 Swing Low liên tiếp (số nến):")
            for p in [10, 25, 50, 75, 90]:
                print(f"    p{p}: {np.percentile(gap_l, p):.1f}")

        if r["prominence_high"]:
            prom_h = np.array(r["prominence_high"])
            print(f"\n  Prominence Swing High (bin, càng lớn càng nổi bật):")
            for p in [10, 25, 50, 75, 90, 95, 99]:
                print(f"    p{p}: {np.percentile(prom_h, p):.1f}")

        if r["prominence_low"]:
            prom_l = np.array(r["prominence_low"])
            print(f"\n  Prominence Swing Low (bin, càng lớn càng nổi bật):")
            for p in [10, 25, 50, 75, 90, 95, 99]:
                print(f"    p{p}: {np.percentile(prom_l, p):.1f}")


def main():
    parser = argparse.ArgumentParser(description="Thống kê Swing High/Low trên data thật")
    parser.add_argument("--parquet", type=str, required=True, help="Đường dẫn file parquet chart dataset")
    parser.add_argument("--text-col", type=str, default="text")
    parser.add_argument(
        "--swing-windows", type=int, nargs="+", default=[1, 2, 3, 5],
        help="Danh sách swing_window cần thống kê (default: 1 2 3 5)",
    )
    parser.add_argument("--sample", type=int, default=None, help="Chỉ lấy N dòng đầu để chạy nhanh thử")
    args = parser.parse_args()

    df = pd.read_parquet(args.parquet)
    if args.sample:
        df = df.head(args.sample)

    print(f"Đang chạy trên {len(df):,} chart, swing_windows={args.swing_windows}...")
    results = compute_swing_stats(df, args.swing_windows, text_col=args.text_col)
    print_stats(results)


if __name__ == "__main__":
    main()