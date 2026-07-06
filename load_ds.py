import random
import pandas as pd
from app.ict.candle import Candle
from app.ict.parser import CandleParser
# Read parquet file
df = pd.read_parquet('data\chart_XAUUSD_dataset_Daily.parquet')

# Display head
print(df.head())

print("Dataset shape:", df.shape)

for _ in range(20):
    parser = CandleParser(raw_text=df.sample(n=1)['text'].values[0])
    start_idx = random.randint(0, len(parser)-20)
    end_idx = start_idx + 20
    candles = parser.slice(start_idx, end_idx)
    slice_parser = CandleParser.from_candles(candles)
    print(slice_parser.raw_text)