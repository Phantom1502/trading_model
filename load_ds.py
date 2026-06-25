import pandas as pd

# Read parquet file
df = pd.read_parquet('data/chart_XAUUSD_dataset_1Min.parquet')

# Display head
print(df.head())

print("Dataset shape:", df.shape)

print(df.iloc[0]['text'])