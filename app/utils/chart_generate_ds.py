"""
chart_generate_ds.py — Script demo sinh chart dataset từ CSV OHLC.

Giả định chạy từ THƯ MỤC GỐC project:
    python app/utils/chart_generate_ds.py

Trước đây file này chạy code ngay ở module scope (side-effect khi import),
nay bọc trong `if __name__ == "__main__":` để an toàn khi (vô tình) bị
import từ nơi khác thay vì chạy trực tiếp.
"""

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = _THIS_DIR
while not os.path.isdir(os.path.join(_REPO_ROOT, "app")):
    _parent = os.path.dirname(_REPO_ROOT)
    if _parent == _REPO_ROOT:
        raise RuntimeError(
            "Không tìm thấy thư mục gốc project (thư mục chứa 'app/')."
        )
    _REPO_ROOT = _parent
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from app.utils.chart.chartcodec import (
    ChartCodec, calculate_atr, N_BINS,
    M1_SCALE, M5_SCALE, M15_SCALE, H1_SCALE, D1_SCALE,
)
from app.utils.chart.chartdatasetbuilder import ChartDatasetBuilder

import pandas as pd


def main():
    codec = ChartCodec(scale=M5_SCALE)
    builder = ChartDatasetBuilder(codec, window_size=100, stride=10, atr_period=100)

    input_csv  = os.path.join(_REPO_ROOT, "data", "XAUUSD_5Min.csv")
    output_pq  = os.path.join(_REPO_ROOT, "data", "chart_XAUUSD_dataset_5Min.parquet")

    dataset = builder.build_from_file(input_csv, output_pq)
    return dataset


if __name__ == "__main__":
    main()