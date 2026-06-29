from app.utils.chart.data_pipeline import ChartCodec, ChartDatasetBuilder, ActionDataGen, M1_SCALE, D1_SCALE
from app.utils.chart.chart_pretrain_pipeline import ChartPretrainPipeline
from app.memlm.tokenizer import VietnameseTokenizer
from app.utils.chart.book_and_output_pipeline import BookPipeline, OutputPipeline

import pandas as pd

def gen_action_data(
    df_path     : str,
    output_path : str,
    tokenizer   : VietnameseTokenizer,
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

    gen.build_to_parquet(
        df          = df,
        tokenizer   = tokenizer,
        output_path = output_path,
        source_name = "action_data",
        stride      = stride,
    )

def gen_base_trading_data(
    tokenizer   : VietnameseTokenizer,
    source_name : str,
    input_parquet_path  : str,
    output_path : str,
):
    pipeline = ChartPretrainPipeline(
        tokenizer, 
        source_name=source_name
    )
    pipeline.build_from_parquet(
        input_path  = input_parquet_path,
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
    
    csv_path = r"data\XAUUSD_1Min.csv"
    raw_parquet_path = r"data\chart_XAUUSD_dataset_1Min.parquet"
    '''
    build_raw_dataset(
        input_ohlc  = csv_path,
        output_path = raw_parquet_path,
        scale       = M1_SCALE,
        window_size = 100,
        atr_period  = 100,
        stride      = 10
    )
    
    # gen base data
    base_ds_output_path = r"data\XAUUSD_1Min_BASE_DS.parquet"
    gen_base_trading_data(
        tokenizer   = tokenizer,
        source_name = "XAUUSD_1Min",
        input_parquet_path  = raw_parquet_path,
        output_path = base_ds_output_path,
    )
    print(f"Đã tạo thành công {base_ds_output_path}")
    '''
    # gen action data
    action_output_path = r"data\XAUUSD_1Min_ACTION_DS.parquet"
    gen_action_data(
        tokenizer   = tokenizer,
        df_path     = csv_path,
        output_path = action_output_path,
    )
    print(f"Đã tạo thành công {action_output_path}")
    
    # Nhánh 3
    book = BookPipeline(tokenizer, min_paragraph_len=500)
    book.build(input_dir="data/books", output_path="data/books.parquet")
    print(f"Đã tạo thành công data/books.parquet")