# ICT Judge/Reward Model — Detectors

> **Lưu ý nguồn thông tin**: source code của `app/ict/` không nằm trong tập
> tài liệu đã rà soát khi viết trang này — nội dung dưới đây tổng hợp từ
> ghi chú thiết kế/kết quả đã biết về package này. Khi có source thật, đối
> chiếu lại và cập nhật file này cho khớp code (đặc biệt tên hàm/tham số
> chính xác).

## Mục đích package

Đánh giá khả năng hiểu pattern chart (theo trường phái ICT — Inner Circle
Trader: liquidity sweep, FVG, market structure shift) của model, đối chiếu
với **outcome giá thực tế** — không dựa vào label do con người gán tay.

## Input format

Chuỗi 20-candle OHLC, tokenize hoá thành **1024-bin discrete price token**
mỗi kênh O/H/L/C — cùng cơ chế FSQ với `ChartCodec`
(`docs/data-pipeline/actions/build-chart-dataset.md`), nhưng window ngắn
hơn nhiều (20 nến thay vì 100+) — phù hợp cho task đánh giá pattern cục bộ
thay vì pretrain hiểu chart tổng quát.

## Pipeline

```
CSV OHLC
  → Detectors (Candle, Swing, FVG, Swept, Shift/MSS, Relations)
  → 4 loại training sample (Swept / FVG / Shift / Synthesis)
```

## Detector chính

| Detector | Phát hiện gì |
|---|---|
| Candle | phân loại nến (tương tự `Candle.direction()` trong pipeline curriculum) |
| Swing | swing high/low cục bộ |
| FVG | Fair Value Gap 3 nến |
| **Swept** | liquidity sweep — giá quét qua 1 vùng thanh khoản (vd 1 swing trước đó) rồi đảo chiều |
| Shift/MSS (Market Structure Shift / Change of Character) | đảo chiều cấu trúc thị trường — khác Swing đơn thuần vì cần xác nhận phá vỡ cấu trúc theo hướng ngược |
| Relations | liên kết nhiều detector trên cùng 1 nến/vùng (tương tự tầng "synthesis" ở curriculum pretrain) |

## Tham số đã validate trên dữ liệu thật (không phải giá trị mặc định chưa kiểm chứng)

| Tham số | Giá trị | Ghi chú |
|---|---|---|
| `swing_window` | 2 | khớp với `swing_window` mặc định ở `CandleParser` phần curriculum |
| `SWEPT_LOOKBACK_DEFAULT` | 10 | số nến nhìn lại để tìm vùng thanh khoản có thể bị sweep |
| `FVG_TOP_K` | 4 | chọn theo **proximity + recency rank**, không phải lấy tất cả FVG tìm được |
| Abbreviated eval keys | — | giảm overflow token của synthesis từ ~82% xuống ~26% (xem `docs/ict/sample-types.md`) |

## Nguyên tắc bắt buộc: Ground truth luôn từ detector, không hand-craft

`build_facts()` (detector output) là **nguồn sự thật duy nhất** cho mọi
sample training và benchmark. Test unit hand-craft từng phát hiện được bug
tích hợp thực tế mà unit test tay không bắt được — ví dụ
`_event_direction()` từng âm thầm trả `None` cho **mọi** Shift event, chỉ
lộ ra khi chạy integration test với `build_facts()` thật trên dữ liệu thật.

→ Khi thêm detector mới hoặc sửa detector cũ: **luôn** viết integration
test chạy qua `build_facts()` đầu-cuối, không chỉ test hàm detector cô lập.

## Test-first

Package được xây **test-first**: 107 golden test viết trước khi tích hợp
pipeline — xem thêm nguyên tắc chung ở `docs/conventions/testing.md`.
