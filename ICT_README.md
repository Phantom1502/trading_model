# app/ict — ICT Detector Package

Package độc lập phát hiện và đánh giá các yếu tố phân tích kỹ thuật ICT
(Inner Circle Trader) từ chuỗi nến OHLC đã được mã hóa thành bin token.
Đây là nền tảng cho Judge / Reward Model trong hệ thống trading.

> **Quan trọng:** Mọi giá trị `open/high/low/close` trong package này đều
> là **bin index** (số nguyên parse từ token `O_xxx`/`H_xxx`/`L_xxx`/`C_xxx`),
> **không phải giá thật**. Label và detector hoàn toàn nhất quán với những
> gì model thấy — không có leak thông tin giá thật.

---

## Cấu trúc

```
app/ict/
├── candle.py       — Candle dataclass + parse_candles(), build_raw_text()
├── parser.py        — CandleParser: slice(), window helper
├── basic.py          — Lớp 1-2: classify_direction, is_pin_bar, is_engulfing
├── structure.py       — Lớp 3: is_swing_high, is_swing_low, is_fvg (binary)
├── ict.py              — Lớp 4: is_swept, grade_fvg, is_shift, scan_all_*
├── relations.py         — Lớp 5: build_relations (Tầng 2)
├── facts.py              — build_facts(): gom toàn bộ detector → 1 fact JSON
├── render.py              — STUB: fact JSON → 4 dạng mẫu tin (Giai đoạn 5)
├── validate.py             — STUB: validate câu GPT khớp fact JSON (Giai đoạn 5)
├── scripts/
│   ├── stats_swing.py        — thống kê Swing High/Low trên data thật
│   └── stats_swept.py         — thống kê Swept (depth, lookback) trên data thật
└── tests/
    ├── conftest.py
    ├── test_candle_parser_basic.py
    ├── test_slice.py
    ├── test_bull_bear.py
    ├── test_pin_bar.py
    ├── test_engulfing.py
    ├── test_swing.py
    ├── test_fvg_binary.py
    ├── test_swept.py
    ├── test_fvg_graded.py
    ├── test_shift.py
    ├── test_relations.py
    └── test_facts.py
```

---

## Cài đặt & chạy test

```bash
pip install pytest

# Từ root project
python -m pytest app/ict/tests/ -v
# Expected: 75/75 passed
```

---

## Các lớp kỹ năng

Package tổ chức theo lớp phụ thuộc — lớp sau build trên lớp trước:

| Lớp | Module | Nội dung |
|-----|--------|----------|
| 0 | `candle.py`, `parser.py` | Parse token → Candle, slice, window |
| 1 | `basic.py` | Bull/Bear/Doji, Pin Bar (Hammer/Shooting Star) |
| 2 | `basic.py` | Engulfing (2 nến liên tiếp) |
| 3 | `structure.py` | Swing High/Low, FVG binary |
| 4 | `ict.py` | Swept, FVG graded, Shift/MSS (BOS/CHoCH) — cả 3 đã xong |
| 5 | `relations.py` | Quan hệ thứ tự/hướng/overlap giữa các event |

---

## Cách dùng nhanh

### Parse chuỗi chart token

```python
from app.ict.parser import CandleParser

raw = "<chart> O_500 H_510 L_490 C_505 O_505 H_520 L_500 C_515 ... </chart>"
parser = CandleParser(raw, swing_window=2)

print(len(parser))          # số nến
print(parser[0])            # Candle(O=500, H=510, L=490, C=505)
print(parser[0].tag())      # "<chart> O_500 H_510 L_490 C_505 </chart>"

sub = parser.slice(0, 20)   # cắt 20 nến đầu
```

### Khi viết test (không có raw_text)

```python
from app.ict.candle import Candle
from app.ict.parser import CandleParser

candles = [
    Candle(open=500, high=510, low=490, close=505),
    Candle(open=505, high=520, low=500, close=515),
]
parser = CandleParser.from_candles(candles, swing_window=2)
```

### Phân loại từng nến

```python
from app.ict.basic import classify_direction, is_pin_bar, is_engulfing

direction = classify_direction(parser[0])   # "BULL" | "BEAR" | "DOJI"
pin       = is_pin_bar(parser[0])           # "HAMMER" | "SHOOTING_STAR" | None
engulf    = is_engulfing(parser[0], parser[1])  # "BULLISH_ENGULFING" | ... | None
```

### Swing High/Low và FVG

```python
from app.ict.structure import is_swing_high, is_swing_low, is_fvg

# Kiểm tra từng nến theo index trong parser
for i in range(len(parser)):
    if is_swing_high(parser, i):
        print(f"Swing High tại nến {i}: H={parser[i].high}")
    if is_swing_low(parser, i):
        print(f"Swing Low tại nến {i}: L={parser[i].low}")
    fvg = is_fvg(parser, i)
    if fvg:
        print(f"FVG {fvg} hoàn thiện tại nến {i}")
```

### Swept (Liquidity Sweep)

```python
from app.ict.ict import is_swept, scan_all_swept

# Kiểm tra 1 nến cụ thể
result = is_swept(parser, index=15, lookback=10)
if result:
    print(result)
    # {
    #   "type": "SWEEP_HIGH",
    #   "swept_candle_idx": 15,
    #   "swing_idx": 10,
    #   "swing_level": 530,
    #   "depth": 5
    # }

# Quét toàn chart (đúng cách — xử lý broken swing)
all_sweeps = scan_all_swept(parser, lookback=10)
```

> **Lưu ý:** Luôn dùng `scan_all_swept()` khi quét toàn chart, không gọi
> `is_swept()` lặp thủ công. Lý do: `scan_all_swept()` cộng dồn tập
> `broken` qua từng nến để đảm bảo swing đã bị phá không được tính sweep
> lại, điều mà gọi đơn lẻ không tự xử lý được.

### FVG graded

```python
from app.ict.ict import grade_fvg

result = grade_fvg(parser, index=10, upto_index=15)
if result:
    print(result)
    # {
    #   "type": "BULL",
    #   "fvg_candle_idx": 10,
    #   "gap_low": 510,
    #   "gap_high": 525,
    #   "gap_size_bins": 15,
    #   "fill_pct": 33.3
    # }
```

> **Lưu ý:** `fill_pct` hiện phản ánh mức lấp **lớn nhất từng đạt được**
> (max overlap qua các nến từ `index+1` đến `upto_index`), không phải
> trạng thái tức thời của nến cuối. Hành vi này còn 1 KNOWN ISSUE cần
> chốt (xem `test_fvg_graded.py::test_near_miss_fill_then_extend`).

### Shift / MSS (BOS / CHoCH)

```python
from app.ict.ict import is_shift, scan_all_shift

# Kiểm tra 1 nến cụ thể — CẦN truyền trend hiện tại (không tự suy ra)
result = is_shift(parser, index=15, trend="BULL", lookback=10)
if result:
    print(result)
    # {
    #   "type": "BOS",              # hoặc "CHoCH"
    #   "direction": "BULL",        # hướng SAU khi shift
    #   "shift_candle_idx": 15,
    #   "swing_idx": 10,
    #   "swing_level": 530,
    #   "broken_type": "HIGH"       # hoặc "LOW"
    # }

# Quét toàn chart — trend TỰ ĐỘNG evolve sau mỗi CHoCH
all_shifts = scan_all_shift(parser, initial_trend="BULL", lookback=10)
```

> **Lưu ý quan trọng — khác biệt với `is_swept`:** `is_shift` dùng
> **Close** (giá đóng cửa) để xác nhận phá cấu trúc, KHÔNG dùng High/Low
> như `is_swept`. Wick xuyên qua swing nhưng Close không vượt → KHÔNG
> tính là shift (dù cùng nến đó vẫn có thể là 1 sweep hợp lệ — 2 khái
> niệm độc lập, xem `test_shift.py::test_near_miss_wick_only_no_close_break`).
>
> **Luôn dùng `scan_all_shift()` khi quét toàn chart**, không gọi
> `is_shift()` lặp thủ công với `trend` cố định — lý do: sau 1 CHoCH,
> trend đảo chiều, các lần shift tiếp theo PHẢI đánh giá theo trend mới.
> `scan_all_shift()` tự động evolve trend và cộng dồn `broken` set (swing
> đã shift không được dùng làm mốc tham chiếu lại), giống nguyên tắc của
> `scan_all_swept()`.
>
> `initial_trend` là input nghiệp vụ nằm NGOÀI phạm vi 20 nến của chart
> (vd suy từ HTF bias) — hàm không tự đoán, PHẢI truyền vào tường minh.

### Gom toàn bộ thành fact JSON

```python
from app.ict.facts import build_facts

# initial_trend BẮT BUỘC — build_facts() không có giá trị default,
# gọi thiếu sẽ raise TypeError (xem test_facts.py::test_clear_facts_requires_initial_trend)
facts = build_facts(parser, initial_trend="BULL", lookback=10)
# {
#   "n_candles": 20,
#   "swept"    : [...],
#   "fvg"      : [...],
#   "shift"    : [...],
#   "relations": [...],
# }
```

---

## Ngưỡng đã xác nhận bằng thống kê (Giai đoạn 2 — hoàn tất phần Bull/Bear/Pin Bar)

Chạy trên **19,403,600 nến XAUUSD M1** (`chart_XAUUSD_dataset_1Min.parquet`).

### Bull / Bear / Doji

```
DOJI (threshold=2)  : 3,357,804 (17.3%)
BULL                 : 8,050,295 (41.5%)
BEAR                 : 7,995,501 (41.2%)

Phân phối |body| (bin):
  p50: 9.0   p75: 16.0   p90: 27.0   p95: 36.0   p99: 61.0
```

**Quyết định:** `DOJI_THRESHOLD_BINS = 2`. Tỷ lệ Doji 17.3% hợp lý cho
khung M1 (nhiều giai đoạn đi ngang ngoài giờ giao dịch chính), BULL/BEAR
cân đối gần 50/50.

### Pin Bar (Hammer / Shooting Star)

```
body/range (chỉ tính nến range>0, N=19,403,600):
  p50: 0.526   p75: 0.760   p90: 0.913   p95: 1.000   p99: 1.000

Hammer       : 1,534,573 (7.91%)
Shooting Star: 1,426,373 (7.35%)
Nến range=0 (sau quantize): 0 (0.0%)
```

**Quyết định:** giữ nguyên `PIN_BAR_WICK_RATIO = 0.6`, `PIN_BAR_BODY_RATIO = 0.3`.
Tổng tỷ lệ Pin Bar ~15.3% cao hơn mức "hiếm, nổi bật" điển hình (3-8%),
nhưng **giữ nguyên có chủ đích**: Pin Bar chỉ là 1 component binary trong
entry condition tổng hợp (swept + shift + FVG + price action tại FVG),
không phải bộ lọc cuối. Mức độ "đẹp" của từng Pin Bar cụ thể sẽ được chấm
Q-score graded riêng (công thức đo lường liên tục, không phải ngưỡng
binary) — đúng nguyên tắc Binary vs Graded (spec mục 3). Ngưỡng binary
rộng ở bước detect là hợp lý; phân biệt "đẹp" vs "tầm thường" thuộc bước
graded, chưa triển khai.

### Fair Value Gap — phân phối `gap_size_bins`

Chạy trên cùng bộ 19,403,600 nến:

```
Tổng FVG phát hiện: 4,908,005   (≈ 25.3% tổng số nến hoàn thiện 1 FVG)

Phân phối gap_size_bins:
  p1: 1.0    p5: 1.0    p10: 2.0   p25: 3.0
  p50: 7.0   p75: 14.0  p90: 24.0
```

**Nhận xét:** tần suất FVG khá cao (~1/4 nến), phân phối lệch phải rõ rệt
(p50=7 → p90=24, tăng ~3.4 lần). Đáng chú ý: **p1 = p5 = 1.0 bin**, nghĩa
là ít nhất 5% tổng số FVG (≈245,000 case trên tập này) có gap chỉ đúng 1
bin — case `boundary_gap_1_bin` trong `test_fvg_binary.py` không phải
case lý thuyết hiếm gặp mà là 1 phân khúc dữ liệu thực sự đáng kể.

**Quyết định:** giữ nguyên `is_fvg()` (Lớp 3, binary) KHÔNG thêm ngưỡng
minimum size — theo đúng nguyên tắc đã áp dụng cho Pin Bar: binary chỉ
là 1 component trong entry condition tổng hợp, không phải bộ lọc cuối.
`gap_size_bins` đã có sẵn trong `grade_fvg()` (Lớp 4) chính là cơ chế để
downstream tự phân biệt FVG yếu (1-bin) và FVG mạnh (24+ bin) mà không
cần cắt bỏ ở tầng detect.

**Lưu ý cho Giai đoạn 5 (curriculum/render):** nên lấy mẫu trải đều theo
percentile khi sinh data training, không để tự nhiên — vì tự nhiên sẽ bị
áp đảo bởi case nhỏ (1-7 bin, chiếm hơn nửa phân phối), model cần thấy đủ
cả case biên lẫn case rõ ràng.

### Swing High / Swing Low (theo `swing_window`)

Chạy trên 194,036 chart (19,403,600 nến), so sánh 4 giá trị `swing_window`:

```
                    w=1        w=2        w=3        w=5
Swing High freq    25.35%     15.02%     10.62%      6.59%
Swing Low  freq    25.34%     15.00%     10.59%      6.58%

Gap giữa 2 swing liên tiếp (p50, số nến):
                     3          6          8          12

Prominence (p50 / p90 / p99, bin) — GẦN NHƯ BẤT BIẾN theo window:
  Swing High         2 / 10 / 25   2 / 10 / 26   2 / 10 / 26   2 / 10 / 27
  Swing Low          2 / 11 / 27   2 / 11 / 28   2 / 11 / 30   2 / 11 / 31
```

**Phát hiện quan trọng:** `prominence` (độ nổi bật = High[swing] - max(High lân cận
trong window), đối xứng cho Low) **gần như không đổi** dù `swing_window` thay đổi
1→5. Window chỉ lọc theo **mật độ** (bao nhiêu swing được phát hiện), không lọc
theo **chất lượng** (swing sống sót mạnh/yếu như nhau bất kể window). Cấu trúc
giá có tính tự tương tự (self-similar) ở các quy mô này — đây là lý do
`prominence` là ứng viên tốt và **ổn định** cho Q-score graded của Swing sau
này, độc lập với lựa chọn `swing_window` ở tầng binary detect.

**Quyết định `swing_window`:** ràng buộc thực tế quan trọng hơn thống kê thuần —
chart chỉ có **20 nến** (giới hạn token). Mỗi `swing_window=w` khiến `2×w` nến
ở 2 đầu chuỗi không thể đánh giá (thiếu context 2 bên):

| `swing_window` | Nến "chết" ở biên | % chart mất | Kỳ vọng số swing / chart 20 nến |
|---|---|---|---|
| 1 | 2 | 10% | ~5 |
| **2 (mặc định)** | 4 | 20% | ~3 |
| 3 | 6 | 30% | ~2 |
| 5 | 10 | **50%** | ~1.3, nửa chart không đánh giá được |

`swing_window=5` loại bỏ nửa chart khỏi khả năng đánh giá — quá đắt cho chart
20 nến, dù về thống kê thuần window lớn "sạch" hơn (ít tie-break). Giữ
`swing_window=2` làm mặc định: cân bằng giữa đủ context và không mất quá nhiều
biên chart; vì prominence bất biến theo window, không đánh đổi "chất lượng"
swing khi chọn window nhỏ hơn.

### Swept — phân phối `depth` theo `lookback`

Chạy trên cùng 194,036 chart (19,403,600 nến), `swing_window=2`:

```
                    lb=5       lb=10      lb=15      lb=20
Sweep freq         8.59%      11.72%     12.17%     12.23%
High/Low ratio    50.5/49.5  50.4/49.6  50.5/49.5  50.5/49.5

Depth (p50 / p90 / p99, bin) — GẦN NHƯ BẤT BIẾN theo lookback:
  Sweep High       7 / 24 / 62-64        (không đổi đáng kể qua các lookback)
  Sweep Low        7 / 25 / 66-69        (không đổi đáng kể qua các lookback)
```

**Đường cong bão hòa rõ rệt:** lb=5→10 tăng mạnh (+3.13pp), lb=10→15 chậm lại
(+0.45pp), lb=15→20 gần như không đổi (+0.06pp, xấp xỉ nhiễu thống kê). Điểm
"khuỷu tay" nằm ở lookback=10-15.

**Phát hiện lặp lại quy luật đã thấy ở Swing (`prominence`) và FVG:** `depth`
gần như bất biến theo `lookback` — tham số tìm kiếm quyết định **số lượng**
sweep phát hiện được, không quyết định **độ sâu** của sweep sống sót. Đây là
quy luật nhất quán xuyên suốt 3 detector đã thống kê (Swing/FVG/Swept), đáng
ghi nhận khi thiết kế detector Shift/MSS sau này: nhiều khả năng `depth`/
`prominence`-kiểu đại lượng sẽ tiếp tục là ứng viên ổn định cho Q-score,
không phụ thuộc nhiều vào tham số cấu hình tìm kiếm.

**Quyết định `lookback`:** giữ `SWEPT_LOOKBACK_DEFAULT = 10` — nằm ngay sau
điểm bắt đầu bão hòa, capture ~96% giá trị khả dụng tối đa (11.72% so với
12.23% ở lb=20) với chi phí tìm kiếm thấp hơn 33% so với lb=15. Khác với
`swing_window`, `lookback` không loại bỏ nến biên (code tự co lại bằng
`max(0, upto_index - lookback)`), nên ràng buộc chart 20 nến ít nghiêm trọng
hơn ở tham số này.

### Còn thiếu — cần chạy thống kê tiếp

5 detector chính (Bull/Bear, Pin Bar, FVG, Swing, Swept) đã có thống kê xác
nhận. `is_shift()`/`scan_all_shift()` đã triển khai xong (Giai đoạn 3) —
thống kê cho nó giờ có thể chạy được, chưa làm:

| Đại lượng | Trạng thái | Ảnh hưởng tới |
|---|---|---|
| Tần suất BOS vs CHoCH, phân phối khoảng cách Close vượt swing_level | ❌ Chưa chạy | Q-score graded cho Shift, phân biệt BOS/CHoCH "mạnh/yếu" |
| Phân phối độ dài "chuỗi BOS liên tiếp" trước khi có CHoCH | ❌ Chưa chạy | Hiểu độ bền trend trong data thật, hữu ích cho việc chọn `initial_trend` khi render curriculum |

---

## Trạng thái triển khai

| Module | Trạng thái | Ghi chú |
|--------|-----------|---------|
| `candle.py` | ✅ Xong | |
| `parser.py` | ✅ Xong | |
| `basic.py` | ✅ Xong | Ngưỡng ĐÃ XÁC NHẬN bằng thống kê thật (N=19.4M nến XAUUSD M1) |
| `structure.py` | ✅ Xong | |
| `ict.py::is_swept` / `scan_all_swept` | ✅ Xong | 10/10 golden test pass (8 gốc + 2 bổ sung) |
| `ict.py::grade_fvg` | ⚠️ Nháp | Logic có, chưa chốt behavior `fill_pct` (KNOWN ISSUE) |
| `ict.py::is_shift` / `scan_all_shift` | ✅ Xong | 11/11 golden test pass (test-first: viết test trước khi có logic) |
| `relations.py` | ✅ Xong | Tie-breaking `SAME_CANDLE` chưa có rule ưu tiên (by design) |
| `facts.py` | ✅ Xong | `initial_trend` giờ BẮT BUỘC (không default) — gọi thiếu raise `TypeError` |
| `render.py` | ❌ Stub | Giai đoạn 5 |
| `validate.py` | ❌ Stub | Giai đoạn 5 |

**Tổng 75/75 golden test pass** trên toàn bộ `app/ict/tests/` (13 file, xem
mục Cấu trúc ở đầu README để biết breakdown theo từng module).

---

## Lộ trình tiếp theo

**Giai đoạn 2 — ✅ Hoàn tất.** Toàn bộ 5 detector chính (Bull/Bear, Pin Bar,
FVG, Swing, Swept) đã có thống kê xác nhận trên data XAUUSD M1 thật, xem
mục "Ngưỡng đã xác nhận" ở trên.

**Giai đoạn 3 — ✅ Hoàn tất.** `is_shift()`/`scan_all_shift()` đã triển khai
theo test-first, 11/11 golden test pass, đã wire vào `facts.py` (yêu cầu
`initial_trend` tường minh).

**Việc còn lại — chưa làm, KHÔNG overlap nhau:**

| # | Việc | Thuộc giai đoạn | Phụ thuộc |
|---|---|---|---|
| 1 | Thống kê BOS/CHoCH (tần suất, phân phối) trên data thật | Giai đoạn 2 (bổ sung muộn) | Không — có thể chạy ngay, script tương tự `stats_swept.py` |
| 2 | Chốt behavior `fill_pct` trong `grade_fvg()` (KNOWN ISSUE — hiện giữ max lịch sử, cần quyết định có đổi sang "trạng thái sau cùng" không) | Giai đoạn 3 (còn sót lại) | Không — độc lập, chỉ cần quyết định + sửa 1 hàm |
| 3 | Chạy `build_relations()` trên output thật của `build_facts()` (không chỉ unit test tay) — kiểm tra rule tie-breaking `SAME_CANDLE` có cần chốt logic ưu tiên không khi gặp data thật | Giai đoạn 4 | Cần việc 1 xong trước (để có đủ cả 3 loại event: swept + fvg + shift trong data thật) |
| 4 | Triển khai `render.py` (4 dạng mẫu tin) + `validate.py` | Giai đoạn 5 | Cần việc 2, 3 xong trước — fact JSON phải đáng tin trước khi render thành data training |

**Đề xuất thứ tự làm:** 1 → 2 → 3 → 4 (việc 1 và 2 độc lập nhau, có thể làm
song song nếu muốn; việc 3 và 4 bắt buộc tuần tự sau).