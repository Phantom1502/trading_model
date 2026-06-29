file_path = "data\pdf_pretrain_dataset.parquet"

import pyarrow.parquet as pq

table = pq.read_table(file_path)
df = table.to_pandas()

print(df['text'][150:200])