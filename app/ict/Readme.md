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
├── render.py              — Template engine: fact JSON → 4 dạng mẫu tin (KHÔNG dùng GPT)
├── validate.py             — validate_cross_consistency + validate_no_leakage
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
    ├── test_relations_integration.py
    ├── test_facts.py
    ├── test_render.py
    └── test_validate.py
```

---

## Cài đặt & chạy test

```bash
pip install pytest

# Từ root project
python -m pytest app/ict/tests/ -v
# Expected: 107/107 passed
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

> **Lưu ý:** `fill_pct` phản ánh **vị trí hiện tại** — chỉ tính overlap của
> nến CUỐI CÙNG (tại `upto_index`) với vùng gap, KHÔNG tích lũy lịch sử.
> Nếu giá từng lấp sâu rồi rời khỏi gap, `fill_pct` sẽ GIẢM theo (không giữ
> lại mức lấp sâu nhất từng đạt). Quyết định đã chốt, xem
> `test_fvg_graded.py::test_near_miss_fill_then_extend`.

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

### Quan hệ giữa các event (`relations`)

`facts["relations"]` (đã tự động gọi bên trong `build_facts`) là list quan
hệ giữa MỌI CẶP event, mỗi phần tử:

```python
{
    "event_a_idx": 1, "event_b_idx": 6,
    "order"         : "A_BEFORE_B",   # | "B_BEFORE_A" | "SAME_CANDLE"
    "same_direction": False,           # | True | None
    "overlap"       : False,           # | True | None
}
```

> **Rule tie-breaking `SAME_CANDLE` (đã quyết định 1 phần):** khi Swept và
> Shift trùng đúng 1 nến (kịch bản kinh điển "nến breakout mạnh" — wick quét
> thanh khoản, đóng cửa xác nhận phá cấu trúc), `order` tự động resolve
> thành `A_BEFORE_B`/`B_BEFORE_A` theo rule **Swept-trước-Shift** (wick hình
> thành trong lúc nến chạy, Close xác nhận sau). Case có **FVG** tham gia
> vẫn giữ nguyên nhãn `SAME_CANDLE` — **chưa quyết định**, không suy đoán
> thêm.
>
> Đây **không phải case hiếm** — trên chart thật, sweep+shift trùng nến xảy
> ra khá tự nhiên khi 1 nến vừa xuyên wick qua swing vừa đóng cửa xác nhận,
> xem `test_relations_integration.py::test_clear_swept_shift_same_candle_resolves_naturally`.

> **Lưu ý:** `render.py` (mục dưới) **không còn tiêu thụ** `facts["relations"]`
> nữa — kể từ khi bỏ field `SEQUENCE` (xem "Field SEQUENCE — ĐÃ BỎ"), việc
> render 4 dạng mẫu tin không cần tới nó. `relations` vẫn được tính trong
> `facts.py` (vẫn là fact/ground truth hợp lệ của Tầng 2) — giữ lại cho khả
> năng dùng sau này (vd tính Q-score tổng hợp ở Tầng 3), không phải phần
> thừa cần xoá.

### Render 4 dạng mẫu tin (KHÔNG dùng GPT)

> **Quyết định quan trọng:** ban đầu dự kiến dùng GPT để diễn đạt fact
> thành văn xuôi. Sau khi pipeline detector đã vững chắc và test kỹ, quyết
> định **bỏ GPT** — vì đó là bước duy nhất có thể hallucinate số liệu,
> trong khi mọi bước khác đều deterministic và verify được 100%. Thay bằng
> **template engine nội bộ**: nhiều mẫu câu cố định/loại event, chọn ngẫu
> nhiên (`random.Random`), số liệu chèn trực tiếp từ fact dict qua string
> interpolation — không có khả năng sai số liệu, xem `test_render.py`.

```python
from app.ict.render import render_all_samples
import random

samples = render_all_samples(facts, parser.raw_text, rng=random.Random(42))
# list tối đa 4 dict (Swept/FVG/Shift/Tổng hợp), mỗi dict:
# {
#   "chart": "<chart> ... </chart>",       # KHÔNG đổi
#   "request": "Phân tích Liquidity Sweep trong chart này.",
#   "explanation": "Nến thứ 6 quét qua đỉnh cũ ...",
#   "eval": "<eval>T=SWEEP_HIGH C=6 SC=3 SL=530 D=5</eval>",
#   "text": "<chart>...\n{request}\n{explanation}\n{eval}",  # 4 phần ghép sẵn
#   "event_count": 2,        # số event ĐÃ HIỂN THỊ (sau khi lọc top-K nếu có)
#   "total_event_count": 2,  # số event THẬT SỰ phát hiện (trước khi lọc)
# }
```

> **Giới hạn phạm vi (v1, CHƯA quyết định):** chart có 0 event (không phát
> hiện Swept/FVG/Shift nào) hiện **SKIP hoàn toàn** — không sinh mẫu "không
> tìm thấy pattern". Đây có thể là data âm (negative example) hữu ích cho
> training, nhưng là quyết định domain cần bàn riêng, KHÔNG tự ý thêm.
>
> Đánh số nến trong text/eval là **1-based** (khớp convention
> `candle_parser.py`), trong khi mọi index trong fact dict là 0-based —
> `render.py` tự động +1 khi hiển thị.

### Bảng viết tắt key trong `<eval>` (đã chốt sau khi có dữ liệu thật)

| Đầy đủ (cũ) | Viết tắt (mới) | Đầy đủ (cũ) | Viết tắt (mới) |
|---|---|---|---|
| `EVENT` (tiền tố) | `E` | `GAP_LOW` | `GL` |
| `TYPE` | `T` | `GAP_HIGH` | `GH` |
| `CANDLE` | `C` | `GAP_SIZE` | `GS` |
| `SWING_CANDLE` | `SC` | `FILL_PCT` | `FP` |
| `SWING_LEVEL` | `SL` | `DIRECTION` | `DIR` |
| `DEPTH` | `D` | `BROKEN` | `BR` |

Lý do: tokenizer BPE nhỏ (~16k vocab, chủ yếu train tiếng Việt/code) không
có sẵn token nguyên khối cho chuỗi ALLCAPS_GẠCH_DƯỚI hiếm gặp, phải tách
nhỏ nhiều mảnh — cộng dồn đáng kể qua nhiều field/event. Xem "Case study 3"
bên dưới để biết số liệu thật đã dẫn tới quyết định này.

### Top-K cho FVG — giới hạn dựa trên dữ liệu thật

> **Phát hiện quan trọng từ validate quy mô lớn (5.7M dòng, cửa sổ 20 nến):**
> giới hạn cửa sổ 20 nến MỘT MÌNH KHÔNG ĐỦ tránh vượt `max_seq=512`. Mật độ
> FVG (~25%/nến) đủ cao để mẫu FVG đơn lẻ đã vượt ngân sách ngay ở **p50**
> (536 token, 51.13% mẫu vượt 512), mẫu Tổng hợp còn nặng hơn (**p50=730,
> 87.99% vượt 512**). Swept/Shift KHÔNG cần giới hạn — 0% vượt ngân sách ở
> cùng thống kê.

`render_fvg_sample` và phần FVG trong `render_synthesis_sample` áp dụng
`FVG_TOP_K` (mặc định = 4): nếu số FVG thật sự phát hiện vượt ngưỡng này,
chỉ giữ lại K event **đáng chú ý nhất**, chọn theo 2 tiêu chí kết hợp:

1. **Gần vùng giá hiện tại** — khoảng cách `|trung điểm gap - Close nến cuối|`
2. **Gần về thời gian** — `candle_idx` gần cuối cửa sổ hơn

Kết hợp bằng **rank** (không phải giá trị thô, vì 2 đại lượng khác đơn vị):
mỗi tiêu chí xếp hạng riêng, cộng 2 rank lại, giữ K event có tổng rank nhỏ
nhất — sau đó **sắp xếp lại theo thứ tự thời gian** khi hiển thị (rank chỉ
quyết định giữ/bỏ, không quyết định thứ tự đọc).

```python
from app.ict.render import FVG_TOP_K   # = 4, xem docstring render.py

sample = render_fvg_sample(facts, raw_chart_text)
sample["event_count"]        # <= FVG_TOP_K
sample["total_event_count"]  # số FVG thật sự phát hiện (có thể > FVG_TOP_K)
```

> **`FVG_TOP_K=4` là giá trị mặc định BAN ĐẦU** — đã xác nhận qua validate
> thật: FVG đơn lẻ giờ 0% vượt `max_seq` (p50=450). Mẫu Tổng hợp vẫn còn
> vượt nhiều dù đã lọc (81.98% > 512 trước khi bỏ SEQUENCE + rút gọn key,
> xem "Case study 3" bên dưới) — nên regenerate + chạy lại
> `stats_validate.py` sau khi áp dụng cả 2 thay đổi để xem tỷ lệ mới.

### Field SEQUENCE — ĐÃ BỎ khỏi mẫu Tổng hợp

Bản thiết kế đầu có field `SEQUENCE` riêng (dạng `"1<2,2~3"`) mã hoá quan
hệ thứ tự giữa các event. **Đã bỏ hoàn toàn** vì dư thừa: field `C`
(candle) đã có sẵn trong TỪNG event — model tự so sánh 2 giá trị `C` để
suy thứ tự thời gian mà không cần field riêng liệt kê lại. Cái duy nhất
`SEQUENCE` từng cung cấp thêm là rule tie-break "Swept trước Shift" khi
trùng candle — giờ truyền đạt đơn giản hơn: **mẫu Tổng hợp sắp xếp event
theo đúng thứ tự thời gian (candle_idx) khi hiển thị** (sort ổn định giữ
đúng thứ tự Swept-trước-Shift khi trùng candle).

Hệ quả: `render.py` **không còn cần `facts["relations"]`** cho bất kỳ việc
gì — toàn bộ logic remap index (từng cần để giữ `SEQUENCE` đúng sau khi
lọc top-K) biến mất theo, code đơn giản hẳn.

### Validate sample đã render

```python
from app.ict.validate import validate_cross_consistency, validate_no_leakage

validate_cross_consistency(samples)   # True/False — field trùng TYPE+CANDLE
                                       # giữa các mẫu phải khớp giá trị
validate_no_leakage(samples[0])       # True/False — mọi số trong Lý giải
                                       # phải truy được nguồn từ Eval
```

> **Giản lược so với thiết kế ban đầu:** vì không còn GPT, bỏ hẳn phần
> "đối chiếu số liệu với GPT output" — số liệu đúng theo construction. Giữ
> lại 2 validate này vì vẫn có thể xảy ra do bug trong chính template/logic
> render (không phải do GPT) — xem case study bug thật đã tìm thấy trong
> `relations.py` ở mục trên, đây là lý do không nên bỏ hẳn lớp validate dù
> tưởng chừng "chắc chắn đúng theo construction".

---

## Ngưỡng đã xác nhận bằng thống kê (Giai đoạn 2 — hoàn tất toàn bộ 5 detector chính)

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

### Shift / MSS — phân phối BOS/CHoCH

Chạy trên cùng 194,036 chart (19,403,600 nến), `swing_window=2`, `lookback=10`,
kiểm tra CẢ 2 giả định `initial_trend` để đo độ nhạy:

```
                        initial=BULL       initial=BEAR
BOS                     3.90%              3.88%
CHoCH                   4.08%              4.09%
Tỷ lệ BOS/CHoCH         48.8% / 51.2%      48.6% / 51.4%

break_distance (p50 / p90 / p99, bin) — GẦN NHƯ BẤT BIẾN:
  BOS                   7 / 26 / 67        7 / 26 / 67
  CHoCH                 7 / 25 / 63        7 / 24 / 63

Độ dài chuỗi BOS liên tiếp trước khi gặp CHoCH:
  p50: 0   p75: 1   p90: 2   p95: 3   p99: 5   mean: 0.85   max: 11-12
```

**Phát hiện 1 — `initial_trend` gần như không ảnh hưởng kết quả tổng thể:**
chênh lệch BULL vs BEAR chỉ ~0.02pp trên 19.4M nến. `scan_all_shift()` tự
evolve trend theo shift thật đầu tiên gặp được, nên giả định ban đầu "rửa
trôi" rất nhanh — không cần lo lắng khi chọn `initial_trend` cho Giai đoạn 5.

**Phát hiện 2 — CHoCH nhiều hơn BOS, run length cực ngắn (mean=0.85, p50=0):**
hơn nửa số lần shift, ngay sau đó là 1 CHoCH khác, hầu như không có BOS chen
giữa. Ở `swing_window=2`, cấu trúc M1 XAUUSD gần như là **zigzag cục bộ liên
tục**, không phải "trend bền vững nhiều bước" — hợp lý với mục đích entry
micro-structure (scalping-style), không phải theo dõi xu hướng dài hạn.

**Ý nghĩa cho Q-score:** vì phần lớn shift là chuỗi ngắn/đảo chiều liên tục,
nhãn binary BOS/CHoCH không tự nói lên "mạnh/yếu" (giống Pin Bar, FVG đã bàn
trước) — `break_distance` sẽ là tín hiệu quan trọng hơn khi tính Q-score
tổng hợp: CHoCH với break_distance 2-3 bin (gần noise) khác hẳn ý nghĩa so
với CHoCH 25+ bin.

---

### Quy luật chung xuyên suốt cả 4 detector (Swing / FVG / Swept / Shift)

Thống kê ở tất cả 4 nhóm trên đều cho ra **cùng 1 phát hiện lặp lại**:

| Detector | Đại lượng "độ mạnh" | Bất biến theo tham số nào |
|---|---|---|
| Swing | `prominence` | `swing_window` (1→5) |
| FVG | `gap_size_bins` | (không có tham số tìm kiếm, nhưng phân phối ổn định) |
| Swept | `depth` | `lookback` (5→20) |
| Shift | `break_distance` | `initial_trend` (BULL/BEAR) |

**Tham số cấu hình tìm kiếm (window/lookback/initial state) quyết định SỐ
LƯỢNG sự kiện phát hiện được, KHÔNG quyết định ĐỘ MẠNH của sự kiện sống
sót.** Đây là quy luật đủ nhất quán (4/4 detector) để dùng làm nguyên tắc
thiết kế chung: khi cần Q-score graded cho bất kỳ detector nào trong package
này, ưu tiên tìm đại lượng "độ mạnh cục bộ" tương tự (chênh lệch/khoảng
cách/kích thước tại điểm phát hiện) thay vì cố tinh chỉnh tham số tìm kiếm
để lọc chất lượng — tham số tìm kiếm không làm việc đó tốt.

---

## Case study 1: bug chỉ integration test mới bắt được

`tests/test_relations.py` (unit test, dùng dict event **tự tạo tay**) từng
pass 100% trong khi `relations.py` có 1 bug thật: `_event_direction()` chỉ
đọc field `"type"` để suy hướng BULL/BEAR, khớp đúng với Swept
(`"SWEEP_HIGH"`/`"SWEEP_LOW"`) và FVG (`"BULL"`/`"BEAR"` trực tiếp) — nhưng
Shift event dùng `"type"="BOS"`/`"CHoCH"` (không khớp pattern nào), khiến
`same_direction` **luôn trả về `None` sai** cho mọi relation có Shift tham
gia, dù event đó có sẵn field `"direction"` tường minh.

Bug này **không bị unit test cũ bắt được** vì dict event trong
`test_relations.py` được viết tay, không có case nào dùng đúng shape thật
của Shift event. Chỉ khi `tests/test_relations_integration.py` chạy
`build_relations()` trên **output thật** của `build_facts()` (dict event do
chính `scan_all_shift()` sinh ra) mới lộ ra sai lệch.

**Bài học áp dụng cho phần còn lại của package:** unit test với dict/data tự
tạo tay dễ vô tình "khớp đúng" giả định của code đang test — không thay thế
được ít nhất 1 lượt integration test chạy trên pipeline thật trước khi coi
1 module là "đã xong". Nguyên tắc này nên áp dụng lại khi triển khai
`render.py`/`validate.py` ở Giai đoạn 5.

---

## Case study 2: validate quy mô lớn tìm ra vấn đề mà sandbox không thấy được

Chạy `stats_validate.py` trên **5.7 triệu dòng data thật** (đã render, cửa
sổ 20 nến) lộ ra 2 phát hiện mà toàn bộ test trong sandbox (dù đã 108 test)
không hề bắt được — vì sandbox chỉ test trên vài chart tổng hợp nhỏ, không
đủ đa dạng để chạm vào các trường hợp này:

1. **Bug template thật (0.6% dòng fail `validate_no_leakage`):** 2 template
   FVG có chữ số **"1"** trong câu tiếng Việt tự nhiên ("hình thành **1**
   khoảng trống") — không phải số liệu từ fact dict, nhưng `validate_no_leakage`
   (cố tình thiết kế đơn giản) hiểu nhầm là leak. Sandbox test trước đó
   dùng seed ngẫu nhiên nhỏ, không may "trúng" đúng 2 template này — chỉ
   khi chạy đủ lớn (hàng triệu dòng, mọi template variant đều được dùng
   nhiều lần) mới chắc chắn lộ ra. Đã fix (bỏ chữ "1" thừa) + thêm
   `test_clear_no_leakage_every_template_variant` duyệt **toàn bộ** template
   một cách tất định (không phụ thuộc random seed) để chặn tái diễn.

2. **Vấn đề thiết kế nghiêm trọng hơn nhiều (không phải bug code, mà là data
   không như kỳ vọng):** dù đã giới hạn cửa sổ 20 nến, mẫu FVG đơn lẻ vẫn
   vượt `max_seq=512` ngay ở **p50** (536 token, 51% mẫu vượt), mẫu Tổng hợp
   còn nặng hơn (**p50=730, 88% vượt**). Nguyên nhân: mật độ FVG (~25%/nến)
   đủ cao để riêng việc giới hạn cửa sổ KHÔNG đủ — quyết định "giảm cửa sổ
   20 nến sẽ giải quyết được" (chốt trước khi có số liệu) hoá ra **không
   đủ**, phải quay lại áp dụng `FVG_TOP_K` đã cân nhắc nhưng gác lại trước
   đó.

**Bài học:** cả 2 phát hiện đều chỉ lộ ra khi chạy **đủ lớn trên data thật**
— không phải vì sandbox test sai, mà vì một số vấn đề (tần suất bug hiếm,
phân phối đuôi dài của token length) về bản chất cần cỡ mẫu lớn mới bộc lộ
rõ. Không nên coi 100% test pass trong sandbox là tín hiệu "sẵn sàng train" —
luôn cần ít nhất 1 lượt validate + thống kê trên toàn bộ (hoặc phần lớn)
dataset thật trước khi đưa vào train, đúng theo gate đã đặt ra ở Giai đoạn 5
("tỷ lệ pass validate trên batch nhỏ đủ cao trước khi gen hàng loạt" — cần
áp dụng lại tương tự sau khi gen hàng loạt, không chỉ trước).

---

## Case study 3: FVG_TOP_K áp dụng xong vẫn chưa đủ — 2 tối ưu bổ sung

Sau khi thêm `FVG_TOP_K=4` (Case study 2), regenerate + chạy lại
`stats_validate.py` trên 5.75M dòng cho kết quả:

```
                    TRƯỚC top-K          SAU top-K
fvg (đơn lẻ)        p50=536, 51.1% vượt  p50=450, 0% vượt   ✅ Giải quyết xong
synthesis           p50=730, 88.0% vượt  p50=635, 82.0% vượt ⚠️ Vẫn còn nặng
```

FVG đơn lẻ đã giải quyết triệt để, nhưng **Tổng hợp vẫn còn 82% vượt
ngân sách** — vì nó cộng thêm Swept (~2.4 event trung bình) + Shift (~1.6)
không bị lọc, cộng với chi phí field verbosity của từng event. Dẫn tới 2
tối ưu bổ sung, cả 2 đều xuất phát từ câu hỏi trực tiếp của người dùng khi
soát lại thiết kế, không phải tự phát hiện qua thống kê:

1. **Bỏ field `SEQUENCE`** — câu hỏi đặt ra: "field `C` (candle) đã đủ để
   model suy thứ tự chưa, có thật sự cần liệt kê lại quan hệ không?". Rà
   lại đúng: `SEQUENCE` (dạng `"1<2,2~3"`) chỉ thêm token mà không thêm
   thông tin mới ngoài rule tie-break "Swept trước Shift" — và rule đó
   giờ truyền đạt đơn giản hơn bằng cách **sắp xếp event theo thời gian**
   khi hiển thị (sort ổn định giữ đúng thứ tự khi trùng candle). Hệ quả
   phụ: loại bỏ hoàn toàn nhu cầu dùng `facts["relations"]` trong
   `render.py`, code đơn giản hẳn (không cần remap index sau top-K nữa).

2. **Rút gọn tên field** (`EVENT`→`E`, `TYPE`→`T`, `SWING_CANDLE`→`SC`...)
   — câu hỏi đặt ra: "các key này lặp lại khá nhiều, viết tắt có ảnh hưởng
   gì không?". Đây nhiều khả năng là đòn bẩy token LỚN HƠN cả việc bỏ
   SEQUENCE: tokenizer BPE nhỏ (~16k vocab, chủ yếu train tiếng Việt/code)
   không có sẵn token nguyên khối cho chuỗi ALLCAPS_GẠCH_DƯỚI hiếm gặp như
   `SWING_CANDLE`, phải tách nhỏ nhiều mảnh — cộng dồn qua nhiều field ×
   nhiều event trong 1 mẫu Tổng hợp.

Cả 2 thay đổi đụng vào **cả `render.py` lẫn `validate.py`** (vì
`_event_identity()`/`_parse_eval_events()` hardcode tên field để nhận
diện event) — đã cập nhật đồng bộ, 107/107 test pass sau khi sửa toàn bộ
test liên quan.

**Kết quả đo lại sau khi regenerate (5.75M dòng) — ✅ Đã chốt:**

```
                    Trước (top-K only)   Sau (bỏ SEQUENCE + rút gọn key)
Tổng thể vượt 512   21.96%               7.08%       ↓ giảm ~3.1 lần
synthesis vượt 512  81.98%               26.32%      ↓ giảm ~3.1 lần
synthesis p50       730                  453         ↓ giảm 277 token
synthesis max        1328                 913         ↓ giảm 415 token
fvg/swept/shift      0% vượt              0% vượt, p50 giảm thêm (vd swept 305→259)
```

Rút gọn key có lợi cho **mọi** loại mẫu (không chỉ Tổng hợp) — xác nhận
qua số liệu thật, đúng dự đoán ban đầu.

**Quyết định:** chấp nhận 7.08% tổng thể vượt ngân sách, KHÔNG tối ưu
thêm. Lý do: `TokenChunkDataset` (dataset.py) không drop document dài hơn
`seg_len` mà CẮT thành nhiều đoạn nối tiếp — 7.08% mẫu bị cắt tạo ra 1
lượng nhỏ "mẫu gãy" (bắt đầu đột ngột giữa/gần cuối block `<eval>`), chấp
nhận được ở quy mô pretrain trộn nhiều domain, không đáng đánh đổi thêm
information loss bằng cách giảm `FVG_TOP_K` hoặc lọc thêm Swept/Shift.

**Bài học:** không phải mọi tối ưu quan trọng đều lộ ra qua thống kê —
2 thay đổi ở đây tới từ việc **rà soát lại thiết kế bằng câu hỏi trực
tiếp** ("có thật sự cần field này không", "tên field dài có ảnh hưởng
gì") chứ không phải từ con số. Chạy số liệu tốt vẫn cần đi kèm việc định
kỳ hỏi lại "phần này có thật sự cần thiết không" — nhất là với phần mới
thêm gần đây (SEQUENCE mới thêm cách đây không lâu để giải quyết vấn đề
khác, chưa kịp bị đặt câu hỏi lại).

---

## Trạng thái triển khai

| Module | Trạng thái | Ghi chú |
|--------|-----------|---------|
| `candle.py` | ✅ Xong | |
| `parser.py` | ✅ Xong | |
| `basic.py` | ✅ Xong | Ngưỡng ĐÃ XÁC NHẬN bằng thống kê thật (N=19.4M nến XAUUSD M1) |
| `structure.py` | ✅ Xong | |
| `ict.py::is_swept` / `scan_all_swept` | ✅ Xong | 10/10 golden test pass (8 gốc + 2 bổ sung) |
| `ict.py::grade_fvg` | ✅ Xong | `fill_pct` = vị trí hiện tại (không tích lũy), đã chốt qua golden test |
| `ict.py::is_shift` / `scan_all_shift` | ✅ Xong | 11/11 golden test pass (test-first: viết test trước khi có logic) |
| `relations.py` | ✅ Xong | Tie-breaking Swept-trước-Shift ĐÃ quyết định; case có FVG vẫn `SAME_CANDLE` (chưa quyết định) |
| `facts.py` | ✅ Xong | `initial_trend` giờ BẮT BUỘC (không default) — gọi thiếu raise `TypeError` |
| `render.py` | ✅ Xong (v1 + top-K + key rút gọn) | Template engine, KHÔNG dùng GPT. `FVG_TOP_K=4`. Đã bỏ SEQUENCE, key rút gọn (xem README). Chưa xử lý case "0 event" |
| `validate.py` | ✅ Xong | `validate_cross_consistency` + `validate_no_leakage`, parse theo key rút gọn mới |

**Tổng 107/107 golden test pass** trên toàn bộ `app/ict/tests/` (16 file, xem
mục Cấu trúc ở đầu README để biết breakdown theo từng module).

---

## Lộ trình tiếp theo

**Giai đoạn 2 — ✅ Hoàn tất.** Toàn bộ 5 detector chính (Bull/Bear, Pin Bar,
FVG, Swing, Swept) đã có thống kê xác nhận trên data XAUUSD M1 thật, xem
mục "Ngưỡng đã xác nhận" ở trên. Thống kê BOS/CHoCH của Shift (việc #1 bên
dưới) cũng đã hoàn tất, dù `is_shift()` thuộc Giai đoạn 3 — xếp gộp vào đây
vì cùng bản chất "thống kê xác nhận tham số".

**Giai đoạn 3 — ✅ Hoàn tất.** `is_shift()`/`scan_all_shift()` đã triển khai
theo test-first, 11/11 golden test pass, đã wire vào `facts.py` (yêu cầu
`initial_trend` tường minh).

**Việc còn lại — chưa làm, KHÔNG overlap nhau:**

| # | Việc | Trạng thái | Thuộc giai đoạn | Phụ thuộc |
|---|---|---|---|---|
| 1 | ~~Thống kê BOS/CHoCH trên data thật~~ | ✅ **Đã xong** — xem mục "Shift / MSS" ở trên | Giai đoạn 2 (bổ sung muộn) | — |
| 2 | ~~Chốt behavior `fill_pct` trong `grade_fvg()`~~ | ✅ **Đã xong** — `fill_pct` = vị trí hiện tại (không tích lũy), 2 test mới xác nhận | Giai đoạn 3 (còn sót lại) | — |
| 3 | ~~Chạy `build_relations()` trên output thật của `build_facts()`~~ | ✅ **Đã xong** — phát hiện + fix 1 bug thật (`_event_direction` bỏ sót field `direction` của Shift), chốt rule Swept-trước-Shift khi trùng nến. Xem mục "Case study" ở trên | Giai đoạn 4 | — |
| 4 | ~~Triển khai `render.py` + `validate.py`~~ | ✅ **Đã xong (v1)** — quyết định bỏ GPT, dùng template engine deterministic. Phát hiện + fix 2 bug thật trong chính `validate.py` (regex số nuốt nhầm dấu chấm cuối câu, logic cross-consistency là tautology không bao giờ fail) | Giai đoạn 5 | — |

**Toàn bộ 4 việc theo lộ trình ban đầu đã hoàn tất — package ở trạng thái
functional end-to-end** (CSV → detector → fact JSON → 4 dạng mẫu tin →
validate), xem demo chạy thật trong lịch sử session.

**Giai đoạn 5.5 — ✅ Hoàn tất (bổ sung sau khi có data thật).** Chạy
`stats_validate.py` trên 5.7M dòng đã gen, phát hiện + fix:
- Bug template thật (0.6% fail `validate_no_leakage`) — 2 template FVG có
  chữ "1" tự nhiên gây false-positive. Đã fix + thêm test duyệt toàn bộ
  template (không phụ thuộc random seed).
- Vấn đề thiết kế nghiêm trọng hơn: cửa sổ 20 nến một mình KHÔNG đủ tránh
  vượt `max_seq=512` (FVG đơn lẻ p50=536, Tổng hợp p50=730). Đã thêm
  `FVG_TOP_K=4` — chọn theo kết hợp gần giá hiện tại + gần thời gian.

**Giai đoạn 5.6 — ✅ Hoàn tất, ĐÃ ĐO LẠI VÀ CHỐT (2 tối ưu bổ sung sau khi
regenerate).** Regenerate lại với `FVG_TOP_K` cho thấy FVG đơn lẻ đã hết
vượt ngân sách (0%), nhưng Tổng hợp vẫn còn 82% vượt. Xử lý bằng 2 thay
đổi (xem "Case study 3" ở trên):
- **Bỏ field `SEQUENCE`** — dư thừa vì field `C` đã đủ để suy thứ tự thời
  gian; thay bằng sắp xếp event theo candle_idx khi hiển thị. Hệ quả phụ:
  `render.py` không còn cần `facts["relations"]`.
- **Rút gọn tên field trong `<eval>`** (bảng viết tắt ở trên) — giảm chi
  phí token do tokenizer BPE nhỏ phải tách nhỏ chuỗi ALLCAPS_GẠCH_DƯỚI dài.

Đo lại trên 5.75M dòng sau 2 thay đổi: tổng thể vượt ngân sách giảm từ
21.96% → **7.08%**, Tổng hợp giảm từ 81.98% → **26.32%**. Đã quyết định
**chấp nhận mức 7.08% này, không tối ưu thêm** — `TokenChunkDataset` cắt
document dài thay vì drop, nên phần vượt ngân sách chỉ tạo 1 lượng nhỏ
mẫu bị cắt (không mất hẳn), chấp nhận được ở quy mô pretrain trộn nhiều
domain.

**Open item còn lại (domain decision, chưa tự ý làm):**
- Chart 0 event (không phát hiện Swept/FVG/Shift nào) hiện bị `render.py`
  SKIP hoàn toàn — nếu cần data âm (negative example) cho training, cần
  quyết định format riêng trước khi thêm.