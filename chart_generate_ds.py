
from app.utils.chart.chartcodec import ChartCodec, calculate_atr, N_BINS, M1_SCALE, M5_SCALE, M15_SCALE, H1_SCALE, D1_SCALE
from app.utils.chart.chartdatasetbuilder import ChartDatasetBuilder

import pandas as pd

codec = ChartCodec(scale=M5_SCALE)        # set 1 lần
builder = ChartDatasetBuilder(codec, window_size=100, stride=10, atr_period=100)

# Sinh dataset               # hoặc
dataset = builder.build_from_file('data\\XAUUSD_5Min.csv', "data\\chart_XAUUSD_dataset_5Min.parquet")