import random
import pandas as pd
from app.ict.candle import Candle
from app.ict.parser import CandleParser

if __name__ == "__main__":
    from app.trading.gen.rawchart_gen import RawChartGenerator
    from app.trading.utils.parquet_writer import ParquetWriterUtil
    
    input_files = [
        "data/encoded/AUDUSD_1Min_preprocessed_encoded.csv",
        "data/encoded/AUDUSD_5Min_preprocessed_encoded.csv",
        "data/encoded/AUDUSD_15Min_preprocessed_encoded.csv",
        "data/encoded/AUDUSD_H1_preprocessed_encoded.csv",
        "data/encoded/EURUSD_1Min_preprocessed_encoded.csv",
        "data/encoded/EURUSD_5Min_preprocessed_encoded.csv",
        "data/encoded/EURUSD_15Min_preprocessed_encoded.csv",
        "data/encoded/EURUSD_H1_preprocessed_encoded.csv",
        "data/encoded/GBPUSD_1Min_preprocessed_encoded.csv",
        "data/encoded/GBPUSD_5Min_preprocessed_encoded.csv",
        "data/encoded/GBPUSD_15Min_preprocessed_encoded.csv",
        "data/encoded/GBPUSD_H1_preprocessed_encoded.csv",
        "data/encoded/US500_1Min_preprocessed_encoded.csv",
        "data/encoded/US500_5Min_preprocessed_encoded.csv",
        "data/encoded/US500_15Min_preprocessed_encoded.csv",
        "data/encoded/US500_H1_preprocessed_encoded.csv",
        "data/encoded/XAUUSD_1Min_preprocessed_encoded.csv",
        "data/encoded/XAUUSD_5Min_preprocessed_encoded.csv",
        "data/encoded/XAUUSD_15Min_preprocessed_encoded.csv",
        "data/encoded/XAUUSD_H1_preprocessed_encoded.csv",
        "data/encoded/XAUUSD_Daily_preprocessed_encoded.csv",
    ]
    
    inputs_val_files = [
        "data/encoded/EURUSD_M1_Val_preprocessed_encoded.csv",
        "data/encoded/GBPUSD_M1_Val_preprocessed_encoded.csv",
        "data/encoded/XAUUSD_M1_Val_preprocessed_encoded.csv",
    ]
    
    pawriter = ParquetWriterUtil("data/dataset/val_raw_100candles.parquet")
    for f in inputs_val_files:
        print(f"Processing {f}")
        pawriter.run(RawChartGenerator(f))
        
    pawriter.close()
        
    # Read parquet file
    df = pd.read_parquet(r'data\dataset\val_raw_100candles.parquet')

    # Display head
    print(len(df))

    print("Dataset shape:", df.shape)

    print(df.head(5))