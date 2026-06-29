from app.utils.chart.data_pipeline import ChartCodec, ChartDatasetBuilder, ActionDataGen, M1_SCALE, D1_SCALE
from app.utils.chart.chart_pretrain_pipeline import ChartPretrainPipeline
from app.memlm.tokenizer import VietnameseTokenizer
import pandas as pd

def gen_action_data(
    df_path     : str,
    output_path : str,
    scale       : float     = M1_SCALE,
    window_size : int       = 20,
    forward_size: int       = 60,
    stride      : int       = 10,
    sl_bins     : list      = None,
    tp_bins     : list      = None,
    seed        : int       = 42,
) -> list[str]:
    df    = pd.read_csv(df_path)
    codec = ChartCodec(scale=scale)
    gen   = ActionDataGen(
        codec,
        window_size  = window_size,
        forward_size = forward_size,
        sl_bins      = sl_bins or [-20, -40, -60, -80],
        tp_bins      = tp_bins or [+40, +80, +120, +160],
    )

    samples  = gen.gen(df, stride=stride)
    balanced = gen.balance(samples, seed=seed)

    print(f"Raw: {len(samples)} | Balanced: {len(balanced)}")
    print(f"Distribution: {gen.distribution(balanced)}")

    # Lưu ra file text
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(balanced))
    print(f"✅ Saved → {output_path}")

    return balanced

def gen_base_trading_data(
    tokenizer   : VietnameseTokenizer,
    source_name : str,
    input_path  : str,
    output_path : str,
):
    pipeline = ChartPretrainPipeline(
        tokenizer, 
        source_name=source_name
    )
    pipeline.build_from_parquet(
        input_path  = input_path,
        output_path = output_path,
    )
    
def build_raw_dataset(
    input_ohlc: str,
    output_path: str,
    scale: float = M1_SCALE,
    window_size: int = 100,
    atr_period: int = 100,
    stride: int = 10,
) -> None:
    codec = ChartCodec(scale=scale)
    
    builder = ChartDatasetBuilder(
        codec, 
        window_size=window_size, 
        stride=stride, 
        atr_period=atr_period
    )
    builder.build_from_file(input_ohlc, output_path)

if __name__ == "__main__":
    import os

    base_dir = os.path.dirname(os.path.abspath(__file__))
    tok_path = os.path.join(base_dir, "app", "memlm", "custom_tokenizer")
    print(f"Testing tokenizer: {tok_path}\n")

    tokenizer = VietnameseTokenizer(pretrained_name=tok_path)
    
    csv_path = r"data\XAUUSD_Daily.csv"
    output_path = r"data\chart_XAUUSD_dataset_Daily.parquet"
    build_raw_dataset(csv_path, output_path, D1_SCALE)
    
    # print output
    df = pd.read_parquet(output_path)
    print(df.head())
    
    # gen base data
    base_output_path = r"data\XAUUSD_Daily.parquet"
    gen_base_trading_data(
        tokenizer   = tokenizer,
        source_name = "XAUUSD_Daily",
        input_path  = output_path,
        output_path = base_output_path,
    )
    
    # print output
    df = pd.read_parquet(base_output_path)
    print(df.head())
    
    # gen action data
    action_output_path = r"data\XAUUSD_Daily_action_data.txt"
    gen_action_data(
        df_path     = base_output_path,
        output_path = action_output_path,
    )