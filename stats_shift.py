"""
stats_shift.py — Thống kê Shift/MSS (BOS/CHoCH) trên data thật
========================================================================
Chạy 1 lần trên data CSV/parquet thật, in ra:
    1. Tần suất BOS vs CHoCH (bao nhiêu % nến shift, tỷ lệ BOS/CHoCH).
    2. Phân phối `break_distance` (Close vượt qua swing_level bao nhiêu
       bin) — đại lượng chính dùng làm nền cho Q-score graded của Shift,
       tương tự depth/prominence/gap_size đã dùng cho Swept/Swing/FVG.
    3. Phân phối độ dài "chuỗi BOS liên tiếp" (số shift liên tiếp cùng
       loại BOS trước khi gặp CHoCH) — cho biết độ bền trend trong data
       thật, hữu ích khi thiết kế curriculum (Giai đoạn 5).

VẤN ĐỀ CẦN LƯU Ý: scan_all_shift() cần `initial_trend` truyền vào tường
minh (không tự suy được) — với mục đích THỐNG KÊ THUẦN (không phải sinh
data chính thức), script này chạy CẢ 2 giả định initial_trend="BULL" và
"BEAR" trên cùng bộ chart, báo cáo song song để kiểm tra độ nhạy của kết
quả với giả định ban đầu (nếu 2 kết quả gần giống nhau, giả định initial
trend ít quan trọng; nếu khác biệt lớn, cần cẩn trọng hơn khi chọn
initial_trend thật cho pipeline gen data ở Giai đoạn 5).

Cách dùng:
    python stats_shift.py --parquet data/chart_XAUUSD_dataset_1Min.parquet
    python stats_shift.py --parquet ... --sample 5000   # chạy nhanh thử trước
"""

import argparse
import numpy as np
import pandas as pd

from app.ict.parser import CandleParser, SWING_WINDOW_DEFAULT
from app.ict.ict import scan_all_shift, SWEPT_LOOKBACK_DEFAULT


def _run_length_stats(events: list[dict]) -> list[int]:
    """
    Từ list event (đã theo đúng thứ tự thời gian trong 1 chart), tính độ
    dài từng "chuỗi BOS liên tiếp" — số lượng BOS liên tiếp trước khi gặp
    1 CHoCH (hoặc trước khi hết chart). Trả về list các độ dài chuỗi.
    """
    runs = []
    current_run = 0
    for e in events:
        if e["type"] == "BOS":
            current_run += 1
        else:   # CHoCH — kết thúc 1 chuỗi (có thể current_run == 0 nếu CHoCH liên tiếp)
            runs.append(current_run)
            current_run = 0
    if current_run > 0:
        runs.append(current_run)   # chuỗi BOS còn dang dở ở cuối chart
    return runs


def compute_shift_stats(df: pd.DataFrame, initial_trend: str, lookback: int, text_col: str = "text") -> dict:
    """Trả về dict thống kê cho 1 giả định initial_trend cố định."""
    result = {
        "total_candles": 0,
        "n_bos": 0,
        "n_choch": 0,
        "break_distance_bos": [],
        "break_distance_choch": [],
        "run_lengths": [],
    }

    for raw_text in df[text_col]:
        parser = CandleParser(raw_text, swing_window=SWING_WINDOW_DEFAULT)
        n = len(parser)
        result["total_candles"] += n

        events = scan_all_shift(parser, initial_trend=initial_trend, lookback=lookback)

        for e in events:
            candle = parser[e["shift_candle_idx"]]
            break_distance = abs(candle.close - e["swing_level"])
            if e["type"] == "BOS":
                result["n_bos"] += 1
                result["break_distance_bos"].append(break_distance)
            else:
                result["n_choch"] += 1
                result["break_distance_choch"].append(break_distance)

        result["run_lengths"].extend(_run_length_stats(events))

    return result


def print_stats(label: str, r: dict):
    total = r["total_candles"]
    n_bos, n_choch = r["n_bos"], r["n_choch"]
    n_total = n_bos + n_choch

    print(f"\n{'='*60}")
    print(f"  initial_trend = {label}")
    print(f"{'='*60}")
    print(f"  Tổng nến phân tích : {total:,}")
    print(f"  BOS                : {n_bos:,} ({n_bos/total*100:.2f}%)")
    print(f"  CHoCH              : {n_choch:,} ({n_choch/total*100:.2f}%)")
    if n_total > 0:
        print(f"  Tỷ lệ BOS/CHoCH    : {n_bos/n_total*100:.1f}% / {n_choch/n_total*100:.1f}%")

    if r["break_distance_bos"]:
        d = np.array(r["break_distance_bos"])
        print(f"\n  break_distance BOS (bin):")
        for p in [10, 25, 50, 75, 90, 95, 99]:
            print(f"    p{p}: {np.percentile(d, p):.1f}")

    if r["break_distance_choch"]:
        d = np.array(r["break_distance_choch"])
        print(f"\n  break_distance CHoCH (bin):")
        for p in [10, 25, 50, 75, 90, 95, 99]:
            print(f"    p{p}: {np.percentile(d, p):.1f}")

    if r["run_lengths"]:
        runs = np.array(r["run_lengths"])
        print(f"\n  Độ dài chuỗi BOS liên tiếp (trước khi gặp CHoCH):")
        for p in [10, 25, 50, 75, 90, 95, 99]:
            print(f"    p{p}: {np.percentile(runs, p):.1f}")
        print(f"    mean: {runs.mean():.2f}   max: {runs.max()}")


def main():
    parser = argparse.ArgumentParser(description="Thống kê Shift/MSS (BOS/CHoCH) trên data thật")
    parser.add_argument("--parquet", type=str, required=True, help="Đường dẫn file parquet chart dataset")
    parser.add_argument("--text-col", type=str, default="text")
    parser.add_argument("--lookback", type=int, default=SWEPT_LOOKBACK_DEFAULT)
    parser.add_argument("--sample", type=int, default=None, help="Chỉ lấy N dòng đầu để chạy nhanh thử")
    args = parser.parse_args()

    df = pd.read_parquet(args.parquet)
    if args.sample:
        df = df.head(args.sample)

    print(f"Đang chạy trên {len(df):,} chart, lookback={args.lookback}...")
    print(f"(Chạy CẢ 2 giả định initial_trend để kiểm tra độ nhạy — xem docstring script)")

    for trend in ("BULL", "BEAR"):
        r = compute_shift_stats(df, initial_trend=trend, lookback=args.lookback, text_col=args.text_col)
        print_stats(trend, r)


if __name__ == "__main__":
    main()