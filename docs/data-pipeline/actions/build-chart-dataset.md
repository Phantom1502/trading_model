# Action: Build Chart Dataset (OHLC → token text)

Files: `app/utils/chart/chartcodec.py` (`ChartCodec`),
`app/utils/chart/chartdatasetbuilder.py` (`ChartDatasetBuilder`).
Bản hợp nhất mới hơn: `app/utils/chart/data_pipeline.py` (cùng logic, gộp
chung với `ActionDataGen`/`PDFDataGen`/`ParquetUtils` — dùng file này cho
code mới, 2 file `chartcodec.py`/`chartdatasetbuilder.py` giữ để tương
thích ngược).

## Khi nào dùng

Khi cần chuyển 1 file CSV giá thô (Open/High/Low/Close theo thời gian)
thành text token hoá để đưa vào tokenizer/model.

## Quy ước bắt buộc — đọc trước khi sửa

1. **Không lookahead bias**: cửa sổ lấy **forward** từ thời điểm neo `t`
   (từ `t` đến `t + window_size - 1`), *không* lấy ngược về quá khứ.
2. **Anchor lấy tại chính `t`** (`anchor_open = Open[t]`,
   `anchor_atr = ATR[t]`) — không phải trung bình cả window.
3. **Phải lưu `anchor_open` + `anchor_atr` cùng mẫu.** Token chỉ mang
   thông tin *tương đối* theo 2 giá trị này — không thể tự suy ra giá trị
   thật khi decode nếu thiếu.

## Cách dùng

```python
from app.utils.chart.chartcodec import ChartCodec, M1_SCALE
from app.utils.chart.chartdatasetbuilder import ChartDatasetBuilder

codec = ChartCodec(scale=M1_SCALE)
builder = ChartDatasetBuilder(codec, window_size=100, stride=10, atr_period=100)
dataset = builder.build_from_file("data/XAUUSD_1Min.csv", "data/chart_dataset.parquet")
```

Output parquet có cột: `start_idx`, `end_idx`, `anchor_open`, `anchor_atr`,
`text` — `text` sẵn sàng cho `tokenizer.encode_batch(df["text"].tolist())`.

## FSQ (Finite Scalar Quantization)

- Mỗi giá (Open/High/Low/Close) normalize: `(price - anchor_open) / (scale * anchor_atr)`,
  clip về `[-1, 1]`, rồi discretize độc lập từng kênh vào lưới cố định
  `N_BINS=1024` (bin=0 ↔ norm=-1, bin=1023 ↔ norm=+1).
- **Không dùng codebook học** như VQ-VAE — lưới cố định, đối xứng, đơn giản
  và luôn round-trip được (`quantize_price` / `dequantize_bin`).
- Mỗi nến sinh 4 token: `O_<bin> H_<bin> L_<bin> C_<bin>`, bọc trong
  `<chart> ... </chart>`.

## Chọn `scale` theo khung thời gian

Hằng số `*_SCALE` (M1/M5/M15/H1/D1 trong `chartcodec.py`) được suy ra bằng
phân tích thống kê thực tế, **không phải số áng chừng**:

```python
# app/utils/chart/chart_data_scale_factor.py — chạy độc lập để tính scale mới
FINAL_SCALE_FACTOR = np.percentile(all_max_ratios, 99.9)
```

Ý tưởng: quét toàn bộ lịch sử, với mỗi window tính tỉ lệ "dạt xa ATR lớn
nhất" (`max|price - anchor_open| / anchor_atr`), lấy percentile 99.9% làm
scale bảo hiểm — đủ để hầu hết biến động thực tế nằm trong `[-1, 1]` mà
không lãng phí resolution cho các outlier hiếm gặp.

**Khi thêm 1 khung thời gian mới** (vd M30, W1) hoặc 1 asset có biến động
khác biệt lớn (crypto vs forex): chạy lại `chart_data_scale_factor.py` trên
dữ liệu thật của khung/asset đó trước khi hard-code 1 scale constant mới —
đừng tái sử dụng scale của khung khác.

## Resolution

```
1 bin ≈ 2 * scale / (N_BINS - 1) × ATR
```

Roundtrip demo (`chartdatasetbuilder.py __main__`) in ra sai số tuyệt đối
lớn nhất sau decode — dùng để verify khi đổi `scale` hoặc `n_bins`.
