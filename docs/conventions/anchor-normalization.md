# Convention: Anchor Normalization & No-Lookahead

Quy tắc nền tảng, áp dụng cho **mọi** nơi encode giá thành token (chart
pretrain, action dataset, ICT sample) — vi phạm quy tắc này gây
**lookahead bias**, một lỗi rất khó phát hiện bằng mắt thường (model vẫn
train "được", loss vẫn giảm) nhưng khiến mọi số liệu benchmark/production
sau này vô nghĩa vì model đã "nhìn thấy tương lai" lúc train.

## Quy tắc

1. **Cửa sổ luôn lấy FORWARD từ mốc neo `t`**: dữ liệu dùng để encode 1
   window là `[t, t + window_size - 1]` — không bao giờ lấy dữ liệu *trước*
   `t` để tính anchor.
2. **Anchor lấy đúng tại `t`** (nến đầu tiên của window):
   - `anchor_open = Open[t]`
   - `anchor_atr = ATR_period[t]` (ATR tính bằng EMA-smoothed, xem
     `calculate_atr()` — dùng dữ liệu **tính đến** `t`, không dùng dữ liệu
     sau `t`).
   - **Không** dùng trung bình/max/min của cả window làm anchor — anchor
     phải là giá trị mà một hệ thống thực tế **đã biết được tại thời điểm
     `t`**, trước khi thấy các nến sau đó.
3. **Entry cho action dataset**: lấy tại giá `Open` của nến **ngay sau**
   window (`t + window_size`), không phải nến cuối cùng trong window.

## Vì sao phải lưu `anchor_open` + `anchor_atr` cùng mỗi sample

Token (`O_bin H_bin L_bin C_bin`) chỉ mã hoá vị trí **tương đối** so với
anchor:

```
norm = (price - anchor_open) / (scale * anchor_atr)
bin  = round((norm + 1) / 2 * (n_bins - 1))
```

Không có cách nào suy ngược ra `price` thật từ riêng chuỗi token — bắt
buộc phải lưu `anchor_open`/`anchor_atr` kèm sample nếu sau này cần
decode lại giá thật (`ChartCodec.decode_window`) hoặc muốn map lại
prediction của model về giá trị tiền tệ thực.

## Áp dụng vào action/trade-simulation

`ActionDataGen._simulate()` mô phỏng SL/TP dựa trên bin đã quantize theo
cùng `anchor_open`/`anchor_atr` của window — **không** re-quantize theo
anchor mới ở mỗi nến forward. Điều này đảm bảo toàn bộ simulation (entry,
SL, TP, exit) nằm trên cùng 1 hệ quy chiếu nhất quán.

## Checklist khi review code encode giá mới

- [ ] Anchor được lấy tại đúng mốc `t`, không phải trung bình/tương lai?
- [ ] ATR dùng EMA-smoothed tính đến `t` (không leak dữ liệu sau `t` vào
      công thức EMA)?
- [ ] Sample output có lưu kèm `anchor_open`/`anchor_atr` (hoặc đủ thông
      tin để suy ra) nếu cần round-trip sau này?
- [ ] Nếu window bao gồm cả điểm entry cho action, entry có nằm **sau**
      window context (không lookahead vào chính window đang phân tích)?
