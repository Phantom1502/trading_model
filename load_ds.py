import pandas as pd

# Read parquet file
df = pd.read_parquet('E:\LLM Dataset\Mix/mix_part_0001.parquet')

# Display head
print(df.head())

print("Dataset shape:", df.shape)

print(df.tail(100))