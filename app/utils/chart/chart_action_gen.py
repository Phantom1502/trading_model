import numpy as np
import pandas as pd
from enum import Enum
from chartcodec import ChartCodec, calculate_atr, M1_SCALE
from collections import defaultdict
import random

# ── Constants ─────────────────────────────────────────────────────────
WINDOW_SIZE  = 20
FORWARD_SIZE = 60
SPREAD_BINS  = 1

SL_BINS = [-20, -40, -60, -80]
TP_BINS = [+40, +80, +120, +160]  # RR 1:2 minimum

class Action(Enum):
    BUY_25    = "buy_25"
    BUY_50    = "buy_50"
    BUY_100   = "buy_100"
    SELL_25   = "sell_25"
    SELL_50   = "sell_50"
    SELL_100  = "sell_100"

TRADE_ACTIONS = list(Action)

# ── Score ─────────────────────────────────────────────────────────────
def compute_score(
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
        speed = 1.0 - (candle_exit / FORWARD_SIZE)
        clean = 1.0 - (abs(max_dd_bins) / max(abs(min(SL_BINS)), 1))
        clean = max(clean, 0.0)
        score = 5.0 + speed * 2.5 + clean * 2.5
        return round(min(score, 10.0), 1)

    return 0.0

# ── Simulate ──────────────────────────────────────────────────────────
def simulate_trade(
    forward_df  : pd.DataFrame,
    entry_bin   : int,
    action      : Action,
    sl_delta    : int,
    tp_delta    : int,
    codec       : ChartCodec,
    anchor_open : float,
    anchor_atr  : float,
) -> dict:
    is_long = action in (Action.BUY_25, Action.BUY_50, Action.BUY_100)

    if is_long:
        effective_entry = entry_bin + SPREAD_BINS
        sl_bin = effective_entry + sl_delta   # âm
        tp_bin = effective_entry + tp_delta   # dương
    else:
        effective_entry = entry_bin - SPREAD_BINS
        sl_bin = effective_entry - sl_delta   # sl_delta âm → cộng
        tp_bin = effective_entry - tp_delta   # tp_delta dương → trừ

    max_dd_bins = 0
    exit_type   = "timeout"
    exit_bin    = codec.quantize_price(
        forward_df.iloc[-1]["Close"], anchor_open, anchor_atr
    )
    candle_exit = FORWARD_SIZE

    for i, row in forward_df.iterrows():
        high_bin  = codec.quantize_price(row["High"],  anchor_open, anchor_atr)
        low_bin   = codec.quantize_price(row["Low"],   anchor_open, anchor_atr)

        if is_long:
            dd = low_bin - effective_entry
            if dd < max_dd_bins:
                max_dd_bins = dd

            if low_bin <= sl_bin:
                exit_type   = "sl_hit"
                exit_bin    = sl_bin
                candle_exit = i + 1
                break

            if high_bin >= tp_bin:
                exit_type   = "tp_hit"
                exit_bin    = tp_bin
                candle_exit = i + 1
                break
        else:
            dd = effective_entry - high_bin
            if dd < max_dd_bins:
                max_dd_bins = dd

            if high_bin >= sl_bin:
                exit_type   = "sl_hit"
                exit_bin    = sl_bin
                candle_exit = i + 1
                break

            if low_bin <= tp_bin:
                exit_type   = "tp_hit"
                exit_bin    = tp_bin
                candle_exit = i + 1
                break

    if is_long:
        pnl_bins = exit_bin - effective_entry
    else:
        pnl_bins = effective_entry - exit_bin

    score = compute_score(exit_type, pnl_bins, max_dd_bins, candle_exit)

    return {
        "exit_type"   : exit_type,
        "candle_exit" : candle_exit,
        "pnl_bins"    : pnl_bins,
        "max_dd_bins" : max_dd_bins,
        "score"       : score,
    }

# ── Format ────────────────────────────────────────────────────────────
def format_sample(
    chart_text  : str,
    action      : Action,
    sl_delta    : int,
    tp_delta    : int,
    result      : dict,
) -> str:
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

# ── Main gen ──────────────────────────────────────────────────────────
def gen_dataset(
    df         : pd.DataFrame,
    codec      : ChartCodec,
    stride     : int = 10,
    atr_period : int = 100,
) -> list[str]:
    df = df.reset_index(drop=True).copy()
    df["__atr__"] = calculate_atr(df, period=atr_period)

    samples   = []
    last_start = len(df) - WINDOW_SIZE - FORWARD_SIZE

    for t in range(0, last_start + 1, stride):
        anchor_open = df.loc[t, "Open"]
        anchor_atr  = df.loc[t, "__atr__"]
        if anchor_atr <= 0 or np.isnan(anchor_atr):
            continue

        # Chart context
        window_df  = df.iloc[t : t + WINDOW_SIZE]
        chart_text = codec.encode_window(window_df, anchor_open, anchor_atr)

        # Entry tại open candle tiếp theo
        entry_open = df.loc[t + WINDOW_SIZE, "Open"]
        entry_bin  = codec.quantize_price(entry_open, anchor_open, anchor_atr)

        # Forward OHLC
        forward_df = df.iloc[
            t + WINDOW_SIZE : t + WINDOW_SIZE + FORWARD_SIZE
        ].reset_index(drop=True)

        # Gen combinations
        for action in TRADE_ACTIONS:
            for sl in SL_BINS:
                for tp in TP_BINS:
                    if tp <= abs(sl):   # skip RR < 1:2
                        continue

                    result = simulate_trade(
                        forward_df, entry_bin, action,
                        sl, tp, codec, anchor_open, anchor_atr
                    )
                    text = format_sample(chart_text, action, sl, tp, result)
                    samples.append(text)

    return samples

def balance_dataset(samples: list[str]) -> list[str]:
    buckets = defaultdict(list)
    for s in samples:
        if 'tp_hit' in s:
            buckets['tp_hit'].append(s)
        elif 'sl_hit' in s:
            buckets['sl_hit'].append(s)
        else:
            buckets['timeout'].append(s)

    # Lấy theo min count → 1:1:1
    min_count = min(len(v) for v in buckets.values())
    
    balanced = []
    for k in buckets:
        balanced.extend(random.sample(buckets[k], min_count))
    
    random.shuffle(balanced)
    return balanced

# ── Demo ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = pd.read_csv(r"data\XAUUSD_1Min.csv")
    df = df.iloc[:1000]

    codec   = ChartCodec(scale=M1_SCALE)
        
    # Xem average move trong 60 candle XAUUSD M1
    df = df.reset_index(drop=True).copy()
    df["__atr__"] = calculate_atr(df, period=100)

    moves = []
    for t in range(200, len(df) - 60, 10):
        anchor_open = df.loc[t, "Open"]
        anchor_atr  = df.loc[t, "__atr__"]
        if anchor_atr <= 0 or np.isnan(anchor_atr):
            continue
        forward = df.iloc[t:t+60]
        high_bin = codec.quantize_price(forward["High"].max(), anchor_open, anchor_atr)
        low_bin  = codec.quantize_price(forward["Low"].min(),  anchor_open, anchor_atr)
        moves.append(high_bin - low_bin)

    print(f"Average range 60 candle: {np.mean(moves):.1f} bins")
    print(f"Median: {np.median(moves):.1f} bins")
    print(f"P25: {np.percentile(moves, 25):.1f} bins")
    print(f"P75: {np.percentile(moves, 75):.1f} bins")


    samples = gen_dataset(df, codec, stride=10)

    print(f"Tổng samples: {len(samples)}")
    print("\n── Sample đầu tiên ──")
    print(samples[0])

    # Distribution check
    from collections import Counter
    exits = [s.split("result>")[1].split(" |")[0] for s in samples]
    print("\n── Exit distribution ──")
    for k, v in Counter(exits).items():
        print(f"  {k}: {v} ({v/len(samples)*100:.1f}%)")
        
    balanced = balance_dataset(samples)  # assign kết quả
    print("\n── Balanced dataset ──")
    print(balanced)