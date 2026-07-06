# Action: Build Action/Trade-Simulation Dataset

Files: `app/utils/chart/chart_action_gen.py` (bản gốc/script),
`app/utils/chart/data_pipeline.py` → class `ActionDataGen` (bản hợp nhất,
khuyến nghị dùng cho code mới — có `build_to_parquet` streaming).

## Mục đích

Sinh dataset dạy model liên kết **(chart, action, SL, TP) → kết quả thực
tế** (dùng giá lịch sử thật để mô phỏng, không phải model tự đoán) — dữ
liệu kiểu reward/outcome cho reasoning về giao dịch.

## Luồng sinh 1 sample

1. Lấy `window_size` nến làm context (`chart_text` qua `ChartCodec`).
2. Entry tại giá `Open` của nến **ngay sau** window (không lookahead).
3. Lấy `forward_size` nến tiếp theo, mô phỏng intracandle High/Low để xác
   định SL/TP nào bị chạm trước (`_simulate`).
4. Với **mọi tổ hợp** `action × sl_bins × tp_bins` (đã lọc `tp < |sl| * min_rr`
   để loại bỏ RR quá thấp), sinh 1 sample text.

```python
from app.utils.chart.data_pipeline import ChartCodec, ActionDataGen, M1_SCALE

codec = ChartCodec(scale=M1_SCALE)
gen = ActionDataGen(
    codec, window_size=20, forward_size=60,
    sl_bins=[-20, -40, -60, -80], tp_bins=[+40, +80, +120, +160], min_rr=2.0,
)
gen.build_to_parquet(df, output_path="data/action.parquet", tokenizer=tok, stride=10)
```

## Action space

```
BUY_25, BUY_50, BUY_100, SELL_25, SELL_50, SELL_100
```
(số hậu tố = % vị thế — logic simulate không phân biệt size, chỉ ảnh hưởng
tới cách model học liên hệ action-size với outcome).

## Scoring (`_compute_score`)

| exit_type | Score |
|---|---|
| `sl_hit` | 0.0 |
| `timeout`, PnL > 0 | 3.0 |
| `timeout`, PnL ≤ 0 | 1.0 |
| `tp_hit` | `5.0 + speed*2.5 + clean*2.5` (tối đa 10.0) — `speed` = thoát càng nhanh càng cao điểm, `clean` = drawdown tối đa càng thấp so với SL càng cao điểm |

## Balance dataset

`ActionDataGen.balance(samples, seed)`: lọc về tỉ lệ **1:1:1** giữa
`tp_hit` / `sl_hit` / `timeout` bằng random sampling xuống mức count nhỏ
nhất. **Luôn kiểm tra distribution trước khi balance** — nếu 1 bucket quá
nhỏ, balance sẽ vứt bỏ phần lớn 2 bucket còn lại; cân nhắc mở rộng dữ liệu
nguồn thay vì downsample mạnh.

```python
dist = ActionDataGen.distribution(samples)   # {"tp_hit": {"count":..,"pct":..}, ...}
```

## Build streaming ra parquet (file lớn, tránh hết RAM)

`build_to_parquet(df, output_path, stride, source_name, tokenizer, batch_size=10000)`
— dùng generator nội bộ `_gen_iter()`, ghi ra parquet theo batch, **không**
gom toàn bộ list samples vào RAM trước khi ghi (khác với `gen()` — trả về
`list[str]` đầy đủ, chỉ nên dùng cho dữ liệu nhỏ/demo).

## Việc cần lưu ý khi đổi tham số

- Đổi `sl_bins`/`tp_bins`/`min_rr` → phân phối `exit_type` thay đổi hoàn
  toàn → chạy lại `distribution()` để kiểm tra cân bằng trước khi balance.
- `forward_size` quá nhỏ so với biến động thực tế của asset → quá nhiều
  `timeout` → nên chạy `ChartDatasetBuilder.analyze_range()` (trong
  `data_pipeline.py`) trước để calibrate `sl_bins`/`tp_bins`/`forward_size`
  phù hợp asset, thay vì đoán.
