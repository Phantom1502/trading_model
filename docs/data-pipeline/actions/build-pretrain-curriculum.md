# Action: Build Pretrain Curriculum Dataset (dạy model đọc chart)

Files: `app/utils/chart/candle_parser.py` (`CandleParser`, `Candle`),
`app/utils/chart/curriculum_generator.py` / `chart_pretrain_pipeline.py`
(`CurriculumGenerator`, `ChartPretrainPipeline`).

> Lưu ý: `curriculum_generator.py` và phần `CurriculumGenerator` trong
> `chart_pretrain_pipeline.py` là **2 bản triển khai song song cùng logic**
> (7 tầng curriculum giống hệt nhau). `chart_pretrain_pipeline.py` là bản
> gộp chung với pipeline build-to-parquet, dùng cho code mới; giữ đồng bộ
> nếu sửa nội dung 1 tầng ở 1 trong 2 file.

## Mục đích

Sinh text pretrain **THUẦN** (continued pretraining, không Q&A/JSON) dạy
model đọc hiểu chart theo curriculum 7 tầng từ dễ đến khó:

| Tầng | Tên | Nội dung |
|---|---|---|
| 0 | concept | Nến là gì, Open/High/Low/Close, yếu tố nào quan trọng nhất |
| 1 | classify_candle | Áp dụng khái niệm — gọi tên Bull/Bear/Doji từng nến |
| 2 | current_price | Giá hiện tại = Close của nến CUỐI CÙNG |
| 3 | swing_structure | Swing High/Low — cần so sánh NHIỀU nến |
| 4 | fvg | Fair Value Gap — 3 nến liên tiếp, so nến đầu với nến 3 (khó nhất) |
| 5 | synthesis | Tổng hợp — 1 nến có thể mang nhiều vai trò cùng lúc |
| 6 | candle_patterns | Pin Bar (Hammer/Shooting Star), Engulfing + gợi ý xác suất (không khẳng định chắc) |

## Random hoá — không phải sample nào cũng cần đủ 7 tầng

`generate_random_subset()` / `random_subset()` chọn 1 trong 2 kiểu:
- **`"subset"`**: chọn k tầng bất kỳ (không liên tục), nhưng **luôn sort lại
  theo thứ tự gốc 0→6** khi render — đảo ngược thứ tự sẽ phá vỡ mạch nhận
  thức dễ→khó.
- **`"range"`**: chọn 1 dải liên tục (vd tầng 1→3).

## Pipeline build từ parquet lớn (không load hết vào RAM)

```python
from app.utils.chart.chart_pretrain_pipeline import ChartPretrainPipeline

pipeline = ChartPretrainPipeline(
    tokenizer, source_name="XAUUSD_1Min",
    slices_per_chart=4, min_slice_len=20, max_slice_len=30,
    curriculum_mode="random",  # "full" | "random"
)
pipeline.build_from_parquet(
    input_path  = "data/chart_XAUUSD_dataset_1Min.parquet",  # output của build-chart-dataset
    output_path = "data/pretrain_XAUUSD_1Min.parquet",
)
```

- Đọc input theo **row-group** (`pf.iter_batches`) — RAM chỉ tỉ lệ với
  `batch_size`, không tỉ lệ tổng số dòng file.
- Mỗi chart dài (100 nến, output của builder chart dataset) được **cắt
  thành nhiều đoạn con ngẫu nhiên** (`slices_per_chart`, `min/max_slice_len`)
  — cắt **sau khi đã parse** thành `List[Candle]` (không cắt trên text thô)
  để tránh cắt giữa token `O_/H_/L_/C_` và để mỗi đoạn con tự sinh lại
  `raw_text` khớp đúng nội dung của nó (`CandleParser.slice()`).
- Schema output giống hệt các nhánh khác — xem
  `docs/conventions/parquet-schema.md`.

## CandleParser — các hàm phát hiện pattern

| Method | Ý nghĩa |
|---|---|
| `is_swing_high(i, window)` / `is_swing_low(i, window)` | High/Low cực trị trong cửa sổ `±window` (mặc định `swing_window=2`, đã validate trên dữ liệu thật) |
| `is_fvg(i)` | So `candles[i-2]` với `candles[i]`, bỏ qua nến giữa → `"BULL"` / `"BEAR"` / `None` |
| `is_pin_bar(i, wick_ratio=0.6, body_ratio=0.3)` | `"HAMMER"` / `"SHOOTING_STAR"` / `None` dựa trên tỉ lệ wick/body |
| `is_engulfing(i)` | So nến `i-1` và `i` → `"BULLISH_ENGULFING"` / `"BEARISH_ENGULFING"` / `None` |

Toàn bộ hàm detect này **không phụ thuộc lẫn nhau** (vd pattern không cần
biết có đang ở swing hay không) — tầng 5 (synthesis) mới là nơi liên kết
kết quả của các detector lại với nhau trên cùng 1 nến.
