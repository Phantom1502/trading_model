file_path = r"data\chart_XAUUSD_dataset_Daily.parquet"

import pyarrow.parquet as pq

table = pq.read_table(file_path)
df = table.to_pandas()

print(df['text'][100])