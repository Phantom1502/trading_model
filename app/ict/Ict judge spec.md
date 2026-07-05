# Spec: ICT Reward Model / Judge — Lộ trình & Dataset Design

> Tài liệu tổng hợp toàn bộ quyết định kiến trúc đã thống nhất, dùng làm tài liệu tham chiếu khi triển khai. Không phải code — chỉ là spec. Thay thế bản trước đó, bổ sung lộ trình đầy đủ từ CSV thô và bảng golden test case chi tiết.

---

## 1. Bối cảnh & mục tiêu

- Dự án: xây "trọng tài" (Judge / Reward Model) để chấm điểm setup giao dịch theo ICT, lấy cảm hứng từ kiến trúc Red Queen Gödel Machine (RQGM) — judge và agent cùng tiến hóa, judge được verify qua ground-truth anchor, đóng băng theo epoch.
- **Trọng tâm đầu tư: Judge (reward model), không phải agent.** Agent sinh setup là việc "rẻ", judge phân biệt tốt/tệ mới là việc "đắt" và quyết định chất lượng toàn hệ thống.
- Model nền hiện tại nhỏ, seq_len giới hạn 512 token, dùng tokenizer riêng (xem mục 6).
- Mục tiêu cuối: model không chỉ "detect có/không", mà phải **suy luận và liên kết** các yếu tố ICT (việc giải thuật không tự làm được) — nhưng "suy luận" này dừng ở Tầng 2 (liên kết fact); đánh giá giá trị thật (Tầng 3) là phán đoán xác suất theo outcome, không phải reasoning logic thuần túy.

---

## 2. Kiến trúc 3 tầng học

| Tầng | Nội dung | Verify bằng | Trạng thái |
|---|---|---|---|
| **1. Yếu tố đơn lẻ** | Swept, FVG, Shift/MSS — binary (có/không) + Q-score graded (0-10) | Golden test tay (unit test, biết trước đáp án) | Swept: xong (`is_swept`, 8/8 golden test pass). FVG, Shift: chưa làm |
| **2. Liên kết thứ tự (mô tả)** | Quan hệ giữa các event đã detect: thứ tự thời gian, cùng hướng hay không, có chồng lấp vùng giá không | Vẫn là fact tính được 100% từ dữ liệu đã detect — verify như tầng 1 | Chưa làm (`build_relations`) |
| **3. Liên kết sâu / đánh giá tổng thể** | Setup này có đáng tin không — dựa trên tổ hợp tầng 1+2 | KHÔNG verify bằng giải thuật. Chỉ verify được qua outcome thật, qua cơ chế Epoch/Meta | Đợi đủ data outcome (Giai đoạn 4-5 trong mục 8) |

**Nguyên tắc khóa cứng:** Tầng 1 và 2 là fact — GPT (nếu dùng để sinh văn bản) chỉ được diễn đạt lại, không được tự suy luận thêm. Tầng 3 là phán đoán giá trị — không có đường tắt nào để dạy nó mà không neo vào outcome thật; cố tạo "hướng dẫn" riêng cho tầng 3 mà bỏ qua outcome sẽ tạo ra hallucination núp dưới vỏ "phân tích chuyên sâu".

---

## 3. Nguyên tắc Binary vs Graded vs Trọng số (3 tầng không được gộp)

| Cấp | Là gì | Ai/cái gì quyết định | Đổi theo thời gian? |
|---|---|---|---|
| **Binary** | Component có xảy ra hay không (objective, deterministic) | Giải thuật (rule cứng từ OHLC) | Không |
| **Graded (Q-score)** | Chất lượng của 1 instance cụ thể (vd: sweep sâu/nhanh = Q cao) | Công thức đo được (độ sâu, tốc độ reject...), KHÔNG phải cảm tính gán tay | Không (công thức cố định) |
| **Trọng số tổng hợp** | Component nào quan trọng bao nhiêu % trong điểm tổng | Học/fit lại dựa trên tương quan với outcome thật | Có — đổi mỗi epoch |

**Hệ quả thiết kế:** Trọng số KHÔNG được bake cứng vào model qua pretrain/SFT, vì mỗi lần epoch đổi trọng số sẽ phải regenerate dataset + train lại — mất hết lợi ích của cơ chế epoch. Model chỉ học dự đoán Q-score của từng component riêng; việc nhân trọng số + cộng tổng làm ở ngoài model (phép tính đơn giản, đổi tức thì).

---

## 4. Nguyên tắc ngưỡng/threshold — không cảm tính, phải có thống kê

Mọi ngưỡng quyết định nhị phân (vd: ngưỡng phân Doji, `wick_ratio`/`body_ratio` của Pin Bar, ngưỡng "near-miss" của Engulfing...) **không được chốt bằng trực giác**. Quy trình bắt buộc trước khi đưa ngưỡng vào golden test chính thức:

1. Chạy detector với ngưỡng tạm (cảm tính ban đầu) trên 1 sample lớn từ data CSV thật.
2. Thống kê phân phối đại lượng liên quan (vd: `|Close - Open|` tính theo bin, size FVG tính theo bin).
3. Xem ngưỡng hiện tại rơi vào đâu trong phân phối — % case bị ảnh hưởng bởi ngưỡng đó.
4. Quyết định giữ/đổi ngưỡng dựa trên con số, không dựa trên "cảm thấy hợp lý".

Việc này áp dụng cho **mọi detector mới** trước khi golden test case ở mục 9 được coi là "chốt cuối cùng" — golden test case liệt kê dưới đây dùng giá trị placeholder, cần điền số thật sau khi có thống kê.

---

## 5. Format dữ liệu — 4 phần chuẩn cho mọi mẫu tin

```
[1. CHART]      input, giữ nguyên token gốc <chart>...</chart>, KHÔNG đổi
[2. YÊU CẦU]    input, câu hỏi (có thể đa dạng cách diễn đạt mỗi mẫu)
[3. LÝ GIẢI]    output, văn xuôi tự nhiên, dạy CÁCH suy luận
[4. CHẤM ĐIỂM]  output, block <eval>...</eval> structured, máy đọc lại được
```

### Nguyên tắc thiết kế đã chốt

- Không inline tag vào giữa `<chart>`. Lý do: phá chu kỳ 4 token/nến (O H L C) mà model pretrain đã quen, làm hỏng transfer learning. Eval luôn là block riêng, sau khi đóng `</chart>`.
- Trích dẫn lại nến bằng số thứ tự ("nến 3", "nến 6"), không lặp lại OHLC đầy đủ trong câu văn — model đã thấy OHLC ở phần chart phía trên, lặp lại tốn token vô ích. (Đánh đổi: model phải tự "đếm" đúng vị trí nến — đây là lý do mục 9 có riêng nhóm test đếm/trích dẫn nến của model nền.)
- `<eval>` dùng `KEY=VALUE` cách nhau bằng space, đánh số theo `E1_, E2_...` nếu nhiều sự kiện (N==1 không đánh số) — không xuống dòng nhiều field, tiết kiệm token, vẫn đủ đơn giản để regex parse.
- Nguyên tắc kiểm tra "rò rỉ": phần [3. Lý giải] phải LUÔN suy ra được trực tiếp từ phần [4. Chấm điểm]. Nếu xóa hết phần 3, người đọc vẫn tái dựng đúng câu chuyện từ phần 4. Nếu phần 3 có chi tiết mà phần 4 không có field tương ứng → dấu hiệu template thêm thắt khi diễn đạt → có script `validate_no_leakage()` bắt tự động trước khi đưa vào training set.
- Độ dài co giãn theo số sự kiện trong chart: <=2 sự kiện → câu đầy đủ, dạy lập luận kỹ. >=3 sự kiện → câu ngắn gọn hơn để không vượt budget token. Không cố định 1 độ dài cứng.

### Bảng viết tắt key trong `<eval>` (đã chốt sau khi có dữ liệu thật)

Validate quy mô lớn (5.7M dòng, cửa sổ 20 nến) cho thấy tên field đầy đủ
(`EVENT1_SWING_CANDLE`...) tốn token đáng kể — tokenizer BPE nhỏ (~16k
vocab, chủ yếu train tiếng Việt/code) không có sẵn token nguyên khối cho
chuỗi ALLCAPS_GẠCH_DƯỚI hiếm gặp, phải tách nhỏ nhiều mảnh. Đã rút gọn:

| Đầy đủ (cũ) | Viết tắt (mới) | Đầy đủ (cũ) | Viết tắt (mới) |
|---|---|---|---|
| `EVENT` (tiền tố) | `E` | `GAP_LOW` | `GL` |
| `TYPE` | `T` | `GAP_HIGH` | `GH` |
| `CANDLE` | `C` | `GAP_SIZE` | `GS` |
| `SWING_CANDLE` | `SC` | `FILL_PCT` | `FP` |
| `SWING_LEVEL` | `SL` | `DIRECTION` | `DIR` |
| `DEPTH` | `D` | `BROKEN` | `BR` |

Ví dụ mẫu Tổng hợp sau khi rút gọn:
```
<eval>E1_T=BULL E1_C=4 E1_GL=508 E1_GH=510 E1_GS=2 E1_FP=0.0 E2_T=SWEEP_HIGH E2_C=6 E2_SC=3 E2_SL=530 E2_D=5</eval>
```

### Field SEQUENCE — ĐÃ BỎ (quyết định thay đổi so với thiết kế ban đầu)

Bản thiết kế đầu tiên có field `SEQUENCE` riêng mã hoá quan hệ thứ tự giữa
các event (dạng `"1<2,2~3"`). Đã **bỏ hoàn toàn** vì dư thừa: field `C`
(candle) đã có sẵn trong TỪNG event, model hoàn toàn có thể tự so sánh 2
giá trị `C` để suy ra thứ tự thời gian mà không cần 1 field riêng liệt kê
lại quan hệ đó. Cái duy nhất `SEQUENCE` từng cung cấp thêm là rule
tie-break "Swept trước Shift" khi trùng candle — giờ truyền đạt đơn giản
hơn nhiều: **mẫu Tổng hợp sắp xếp event theo đúng thứ tự thời gian
(candle_idx) khi hiển thị** (sort ổn định, nên 2 event cùng candle vẫn giữ
đúng thứ tự Swept-trước-Shift đã quyết định), nên thứ tự xuất hiện trong
text tự nó đã là tín hiệu thứ tự.

Hệ quả: không còn cần dùng `relations` (Tầng 2) trong bước render — toàn
bộ logic remap index (từng cần để giữ `SEQUENCE` đúng sau khi lọc top-K)
cũng biến mất theo, code đơn giản hẳn.

### Ngân sách token (tính toán đã thống nhất — CẦN ĐỌC CÙNG mục dưới)

- 1 chart 20 nến = 20 x 4 token giá + 2 marker = 82 token
- Còn lại ~400 token cho text (yêu cầu + lý giải + eval) trên seq_len 512
- 1 bộ chart (1 lần chạy giải thuật) dùng để render ra nhiều mẫu tin (xem mục 7), không chạy giải thuật lại nhiều lần

**⚠️ Cập nhật quan trọng từ dữ liệu thật:** tính toán ban đầu ("20 nến sẽ
đủ") KHÔNG ĐÚNG trên thực tế — mật độ FVG (~25%/nến) đủ cao để một mình
giới hạn cửa sổ 20 nến không tránh được việc vượt `max_seq=512` (mẫu FVG
đơn lẻ p50=536 token trước khi tối ưu). Đã bổ sung `FVG_TOP_K=4` (chọn
theo gần giá hiện tại + gần thời gian) và rút gọn key như trên — xem
README.md package `app/ict/` "Case study 2" và "Case study 3" để biết chi
tiết số liệu và quá trình xử lý.

---

## 6. Tokenizer — lưu ý kiến trúc quan trọng

- `price_vocab` hiện tại: atomic ID, offset-based, KHÔNG qua BPE, KHÔNG dùng `add_tokens()` (tự cộng `vocab_size` thủ công — quên sẽ gây `IndexError`).
- `Candle.tag()` luôn tự đóng gói cặp `<chart>...</chart>` riêng quanh chính nó → khi trích dẫn lại 1 nến bất kỳ ở đâu trong văn bản (kể cả ngoài block chart chính), nó vẫn đi qua đúng `price_vocab`, KHÔNG bị BPE chẻ vụn. Đây là lý do thiết kế trích dẫn `c.tag()` trong câu mô tả luôn an toàn, không cần thêm regex riêng.
- Nếu sau này cân nhắc thêm tag/marker mới (vd cho OB, HTF...) cần vocab riêng (không dùng BPE), phải nhớ resize Embedding table nếu model đã pretrain.

---

## 7. 4 dạng mẫu tin (curriculum learning)

| Dạng | Yêu cầu (phần 2) | Lý giải (phần 3) | Eval (phần 4) |
|---|---|---|---|
| 1. Swept | "Phân tích Liquidity Sweep..." | CHỈ nói về sweep, bỏ qua FVG/Shift dù chúng có xuất hiện trong cùng chart | Chỉ field SWEPT |
| 2. FVG | "Phân tích Fair Value Gap..." | Chỉ nói về FVG | Chỉ field FVG |
| 3. Shift | "Phân tích Market Structure Shift..." | Chỉ nói về shift | Chỉ field SHIFT |
| 4. Tổng hợp | "Phân tích toàn bộ setup..." | Nói về CẢ 3 + quan hệ thứ tự (Tầng 2) | Đủ field cả 3 + field SEQUENCE |

**Lý do tách 3 dạng đơn (curriculum, không gộp thẳng vào dạng 4):**
- Dạy từng kỹ năng riêng trước, dễ debug nếu model yếu ở đúng 1 loại (không lẫn lỗi "không nhận diện được" với lỗi "không biết liên kết")
- Dạng đơn cố ý chỉ trả lời đúng phạm vi được hỏi dù chart có nhiều pattern khác — giúp model hình thành khái niệm rõ ràng, tránh lẫn lộn khi đang học khái niệm A mà bị nói lẫn khái niệm B

**Lý do dùng chung 1 bộ chart cho cả 4 dạng (không sinh chart riêng từng dạng):**
- Tiết kiệm: 1 lần chạy giải thuật trên 1 chart → ra fact JSON đầy đủ → render thành cả 4 mẫu
- Model thấy cùng 1 chart dưới nhiều góc hỏi khác nhau → học object bền hơn, không lệ thuộc 1 dạng câu hỏi
- Tự động có bộ test đối chiếu chéo (cross-consistency check): field SWEPT trong mẫu 1 phải khớp 100% với field SWEPT trong `<eval>` của mẫu 4. Lệch → lỗi ở bước generate (GPT diễn đạt sai số liệu) → bắt tự động, không cần review tay từng mẫu.

```
Pipeline generate:
1 chart (20 nến)
  -> chạy is_swept(), is_fvg(), is_shift() qua mọi nến -> fact JSON đầy đủ
  -> từ 1 bộ fact, render 4 mẫu tin (mỗi mẫu chỉ lấy đúng phần fact liên quan)
  -> GPT diễn đạt thành văn xuôi (xem mục 8)
  -> validate: số liệu trong câu khớp đúng JSON gốc
  -> validate chéo: field giữa mẫu đơn và mẫu tổng hợp phải khớp nhau
```

---

## 8. Sinh văn xuôi — Template engine nội bộ, KHÔNG dùng GPT

> **Quyết định đã đổi so với bản spec trước:** ban đầu dự kiến dùng GPT để
> diễn đạt JSON fact thành văn xuôi (tăng diversity cách diễn đạt). Sau khi
> pipeline detector (Tầng 1-2) đã vững chắc và test kỹ (test-first, thống
> kê xác nhận, integration test bắt được bug thật ở `relations.py`), quyết
> định **bỏ GPT khỏi bước sinh văn xuôi** — vì đây là bước duy nhất trong
> toàn pipeline có thể làm sai số liệu (hallucinate), trong khi mọi bước
> khác đều deterministic và verify được 100%. Đưa 1 bước non-deterministic
> vào ngay sau khi vừa xây xong 1 pipeline chặt chẽ là tự tạo lỗ hổng mới.

**Thay thế bằng: template engine nội bộ + ngân hàng biến thể.**

| Vai trò | Cách làm | Lý do |
|---|---|---|
| Xác định có pattern hay không, Q-score bao nhiêu, sequence đúng/sai | Giải thuật (không đổi) | Vẫn là target dùng đo tương quan outcome ở Giai đoạn 4 (mục 11) |
| Diễn đạt JSON fact thành văn xuôi | **Nhiều template câu cố định sẵn / loại event**, chọn ngẫu nhiên (`random.choice`) + ngân hàng từ đồng nghĩa | Vẫn đạt mục tiêu tránh model học thuộc 1 khuôn câu, nhưng **100% deterministic** — số liệu luôn lấy trực tiếp từ fact dict qua string interpolation, KHÔNG đi qua bất kỳ mô hình sinh văn bản nào, loại bỏ hoàn toàn khả năng hallucinate số |

**Hệ quả:**
- Bỏ hẳn bước gọi API GPT trong pipeline gen data — nhanh hơn, rẻ hơn, không cần rate-limit/retry logic.
- `validate.py` được giản lược: bỏ phần "đối chiếu số liệu với GPT output" (không còn cần thiết vì số liệu render trực tiếp từ fact, đúng theo construction) — giữ lại phần cross-consistency giữa 4 dạng mẫu tin và kiểm tra rò rỉ Lý giải↔Eval (vẫn có thể xảy ra do bug trong chính template/logic render, không phải do GPT).
- Đa dạng hoá đạt được qua: (a) nhiều template/loại event, (b) ngân hàng từ đồng nghĩa cho thuật ngữ lặp lại (vd "quét thanh khoản"/"phá vỡ đỉnh cũ"/"xuyên qua vùng swing"), (c) random hoá thứ tự câu khi có nhiều event trong mẫu Tổng hợp.

---

## 9. Bảng golden test case — theo từng lớp kỹ năng

Mỗi detector áp dụng khung 5 nhóm case chuẩn: **clear** (rõ ràng), **boundary** (ngay ngưỡng quyết định), **near_miss** (giống pattern nhưng không phải, tránh false positive), **edge_position** (đầu/cuối chuỗi, thiếu context 2 bên), **tie_breaking** (nhiều lựa chọn ngang nhau).

Quy ước đặt tên file: `test_<detector_name>.py`. Quy ước đặt tên hàm: `test_<nhóm>_<mô_tả_ngắn>`.

### Lớp 0 — Hạ tầng (nền cho mọi detector)

| File | Case | Input (mô tả) | Expected |
|---|---|---|---|
| `test_candle_parser_basic.py` | `clear_parse_count` | Chuỗi 5 nến hợp lệ | `len(parser) == 5` |
| | `clear_parse_values` | 1 nến `O_500 H_510 L_490 C_505` | Candle đúng 4 giá trị |
| | `boundary_single_candle` | Chuỗi chỉ 1 nến | Parse thành công, không lỗi index |
| | `near_miss_malformed_token` | Thiếu 1 trong 4 token (vd chỉ có O H L) | Bỏ qua nến lỗi hoặc raise rõ ràng (quyết định behavior, test theo đúng quyết định) |
| | `edge_position_empty_chart` | `<chart></chart>` rỗng | `len(parser) == 0`, không crash |
| `test_slice.py` | `clear_slice_middle` | Cắt `[5:10]` từ chuỗi 20 nến | 5 nến đúng, đúng thứ tự |
| | `boundary_slice_full_range` | Cắt `[0:n]` | Kết quả giống parser gốc |
| | `edge_position_slice_at_end` | Cắt `[n-1:n]` | 1 nến cuối cùng, không lỗi |
| | `tie_breaking_raw_text_rebuild` | Slice rồi build lại raw_text | `raw_text` mới parse lại ra đúng candles đã cắt |

### Lớp 1 — Yếu tố đơn nến

| File | Case | Input (mô tả) | Expected |
|---|---|---|---|
| `test_bull_bear.py` | `clear_bull` | `O=500, C=520` (chênh lệch lớn) | `BULL` |
| | `clear_bear` | `O=520, C=500` | `BEAR` |
| | `boundary_at_threshold` | `C - O` đúng bằng ngưỡng hiện dùng (xem mục 4, cần điền số sau thống kê) | Theo đúng định nghĩa `>`/`<` (không phải `>=`/`<=`) — test phải khẳng định rõ dùng strict hay non-strict |
| | `boundary_just_below_threshold` | `C - O` = ngưỡng - 1 bin | `DOJI` |
| | `near_miss_doji_with_long_wick` | `O=500, C=501` (gần Doji) nhưng `H`, `L` cách xa | Vẫn `DOJI` theo direction (wick không ảnh hưởng kết quả Bull/Bear/Doji) |
| `test_pin_bar.py` | `clear_hammer` | Body nhỏ, lower wick dài, upper wick gần 0 | `HAMMER` |
| | `clear_shooting_star` | Body nhỏ, upper wick dài, lower wick gần 0 | `SHOOTING_STAR` |
| | `boundary_wick_ratio_threshold` | Lower wick đúng bằng `wick_ratio * range` | Theo đúng strict/non-strict đã định nghĩa trong code |
| | `near_miss_large_body` | Wick dài nhưng body cũng lớn (vượt `body_ratio`) | `None` (không phải Pin Bar) |
| | `near_miss_both_wicks_long` | Cả 2 wick đều dài tương đương | `None` (không thỏa điều kiện `upper < lower * 0.5` hoặc ngược lại) |

### Lớp 2 — Yếu tố 2 nến liên tiếp

| File | Case | Input (mô tả) | Expected |
|---|---|---|---|
| `test_engulfing.py` | `clear_bullish_engulfing` | Nến 1 Bear nhỏ, nến 2 Bull lớn nuốt trọn thân nến 1 | `BULLISH_ENGULFING` |
| | `clear_bearish_engulfing` | Nến 1 Bull nhỏ, nến 2 Bear lớn nuốt trọn | `BEARISH_ENGULFING` |
| | `boundary_exact_engulf` | `Open[2] == Close[1]` và `Close[2] == Open[1]` (khớp biên đúng bằng, không vượt) | Theo đúng `<=`/`>=` đã code |
| | `near_miss_almost_engulf` | Thân nến 2 thiếu 1 bin để nuốt trọn nến 1 | `None` |
| | `near_miss_same_direction` | 2 nến cùng hướng (cả 2 Bull hoặc cả 2 Bear) | `None` |
| | `edge_position_first_candle` | `index = 0` (không có nến trước) | `None`, không lỗi index âm |

### Lớp 3 — Yếu tố cửa sổ N nến

| File | Case | Input (mô tả) | Expected |
|---|---|---|---|
| `test_swing.py` | `clear_swing_high` | 1 nến cao vượt hẳn các nến lân cận trong `swing_window` | `is_swing_high == True` |
| | `clear_swing_low` | Tương tự cho đáy | `is_swing_low == True` |
| | `tie_breaking_equal_high` | 2 nến trong window có `High` bằng nhau, cùng là max | Quyết định rõ: cả 2 đều `True`, hay chỉ 1 (theo định nghĩa `target == max(...)` hiện tại → cả 2 sẽ `True`, cần xác nhận đây có phải hành vi mong muốn) |
| | `edge_position_window_start` | `index < swing_window` (không đủ nến bên trái) | `False` (không đủ context, theo code hiện tại) |
| | `edge_position_window_end` | `index + swing_window >= n` (không đủ nến bên phải) | `False` |
| | `near_miss_local_not_global_high` | Nến cao nhất trong `swing_window` hẹp nhưng không phải cao nhất toàn chart | Vẫn `True` (đúng định nghĩa "swing cục bộ", không phải global — test để khẳng định rõ phạm vi) |
| `test_fvg_binary.py` | `clear_bullish_fvg` | `Low[nến 3] > High[nến 1]`, chênh lệch rõ (>10 bin) | `BULL` |
| | `clear_bearish_fvg` | `High[nến 3] < Low[nến 1]`, chênh lệch rõ | `BEAR` |
| | `boundary_gap_1_bin` | Chênh lệch đúng 1 bin | `BULL`/`BEAR` (vẫn hợp lệ theo `>`/`<` strict) — đây là case quan trọng nhất cần xác nhận bằng thống kê (mục 4) |
| | `boundary_gap_0_bin` | `Low[nến 3] == High[nến 1]` (chạm nhau, không vượt) | `None` (không phải FVG, theo `>` strict) |
| | `near_miss_middle_candle_fills` | 2 đầu có gap thật nhưng nến giữa có wick chạm vào vùng gap | Vẫn `BULL`/`BEAR` theo định nghĩa hiện tại (chỉ so nến 1 và nến 3, nến giữa không ảnh hưởng kết quả binary — test để xác nhận rõ phạm vi định nghĩa) |
| | `edge_position_first_two_candles` | `index < 2` (không đủ 3 nến) | `None`, không lỗi index âm |

### Lớp 4 — Yếu tố ICT nâng cao

| File | Case | Input (mô tả) | Expected |
|---|---|---|---|
| `test_swept.py` (đã có 8/8, bổ sung thêm) | *(8 case cũ đã pass — giữ nguyên)* | | |
| | `near_miss_multiple_swings_pick_nearest` | Nhiều swing high/low hợp lệ trong lookback | Chọn đúng swing gần nhất theo thời gian, không phải swing đầu tiên tìm thấy |
| | `tie_breaking_re_sweep_same_swing` | Swing đã bị đánh dấu sweep, giá quay lại sweep lần 2 cùng mức đó | Không tính sweep lần 2 (đã phá thì không tính lại — đúng theo ghi chú đã có trong spec cũ) |
| `test_fvg_graded.py` (`grade_fvg`, chưa code) | `clear_unfilled` | FVG vừa hình thành, chưa có nến nào lấp | `fill_pct == 0` |
| | `clear_fully_filled` | Giá quay lại lấp hoàn toàn vùng gap | `fill_pct == 100`, có thể không còn coi là FVG active |
| | `boundary_partial_fill_50` | Giá lấp đúng nửa vùng gap | `fill_pct ≈ 50` (sai số cho phép theo bin resolution) |
| | `near_miss_fill_then_extend` | Lấp 1 phần, rồi giá đảo chiều mở rộng gap trở lại | `fill_pct` phản ánh đúng trạng thái SAU CÙNG, không phải mức lấp sâu nhất từng đạt được (quyết định rõ trong code, test khẳng định lại) |
| `test_shift.py` (`is_shift`/`is_mss`, chưa code) | `clear_bos_same_direction` | Phá swing nhưng cùng hướng trend hiện tại | `BOS` |
| | `clear_choch_reversal` | Phá swing ngược hướng trend hiện tại | `CHoCH` |
| | `boundary_close_exactly_at_swing` | Giá đóng cửa đúng bằng mức swing (không vượt qua) | `None` (chưa phá, theo `>`/`<` strict) |
| | `near_miss_wick_only_no_close_break` | Wick chạm/vượt swing nhưng giá đóng cửa không vượt | `None` — đây là case "near-miss" quan trọng nhất theo ghi chú đã có trong spec cũ |
| | `tie_breaking_multiple_swings_broken_same_candle` | 1 nến phá nhiều swing cùng lúc | Quyết định rõ: tính shift lớn nhất hay tất cả (cần chốt logic trước khi viết test) |

### Lớp 5 — Quan hệ liên kết (Tầng 2)

| File | Case | Input (mô tả) | Expected |
|---|---|---|---|
| `test_relations.py` (`build_relations`, chưa code) | `clear_sequential_order` | Event A tại nến 5, event B tại nến 10 | `A` trước `B`, đúng thứ tự |
| | `clear_same_direction` | 2 event cùng hướng Bull | `same_direction == True` |
| | `clear_overlap_zone` | 2 event có vùng giá chồng lấp (vd FVG và swept cùng range) | `overlap == True` |
| | `tie_breaking_same_candle_index` | 2 event xảy ra tại đúng cùng 1 nến | Quyết định rõ thứ tự ưu tiên (theo loại event hay theo thứ tự phát hiện) — chốt logic trước khi test |
| | `boundary_adjacent_no_overlap` | 2 vùng giá kề sát nhau nhưng không chồng lấp (chênh đúng 1 bin) | `overlap == False` |

---

## 10. Lộ trình triển khai — Giai đoạn 0 đến 7

> Nguyên tắc xuyên suốt: **test luôn đi trước hoặc song song với code sinh ra nó**, không generate hàng loạt rồi mới test ngược lại. Mỗi giai đoạn có gate rõ ràng — không qua gate thì không sang giai đoạn sau.

### Giai đoạn 0 — Chuẩn bị dữ liệu thô
**Việc làm:** CSV OHLC thô → tính ATR → xác định `scale` (đã có `chart_data_scale_factor.py`) → encode `<chart>` token qua `ChartCodec`.
**Gate:** roundtrip test pass (decode lại đúng giá trong sai số 1 bin) — đã có sẵn.
**Trạng thái:** Xong.

### Giai đoạn 1 — Golden test Lớp 0-3 (hạ tầng + yếu tố cơ bản)
**Việc làm:** Viết đủ 6 file test theo mục 9 (Lớp 0-3), trước hoặc song song với code logic chưa hoàn thiện.
**Gate:** toàn bộ test pass, đặc biệt nhóm `boundary` và `near_miss` — đây là nơi bug thật sự ẩn.
**Trạng thái:** Chưa làm — ưu tiên cao nhất hiện tại.

### Giai đoạn 2 — Thống kê phân phối trên data thật
**Việc làm:** Chạy detector Lớp 1-3 trên sample lớn từ CSV thật, thống kê phân phối (% Doji, size FVG, tần suất swing...) để điền số thật vào các case `boundary` ở mục 9 (xem nguyên tắc mục 4).
**Gate:** có con số cụ thể để quyết định ngưỡng, không còn placeholder cảm tính.
**Trạng thái:** Chưa làm.

### Giai đoạn 3 — Golden test + code Lớp 4 (ICT nâng cao)
**Việc làm:** Hoàn thiện thêm test Swept (case còn thiếu) → `grade_fvg` + test → `is_shift`/`is_mss` (test-first, viết test để định nghĩa rõ behavior trước khi viết logic).
**Gate:** mỗi detector đủ 5 nhóm case, đặc biệt `near_miss` cho Shift.
**Trạng thái:** Swept xong phần cũ, còn lại chưa làm.

### Giai đoạn 4 — `build_relations` (Tầng 2) + golden test
**Việc làm:** Thiết kế struct quan hệ + test (mục 9, Lớp 5). Chạy thử trên output thật của Giai đoạn 3, không chỉ data tay.
**Gate:** test pass + chạy ổn trên case thực tế phức tạp hơn unit test đơn giản.
**Trạng thái:** Chưa làm.

### Giai đoạn 5 — Pipeline render 4 dạng mẫu tin + validate tự động
**Việc làm:** 1 chart → fact JSON (toàn bộ detector) → render 4 dạng → GPT diễn đạt → script validate (đối chiếu số liệu, validate chéo, kiểm tra rò rỉ).
**Gate:** tỷ lệ pass validate trên batch nhỏ (vài trăm mẫu) đủ cao (>95%) trước khi gen hàng loạt.
**Trạng thái:** Chưa làm.

### Giai đoạn 6 — Test kỹ năng model nền (song song, độc lập, có thể bắt đầu từ Giai đoạn 1)
**Việc làm:**
- Test đếm/trích dẫn đúng "nến thứ N" (quan trọng vì spec dùng trích dẫn số thứ tự thay vì lặp OHLC)
- Test model có dựa dẫm vào phần Lý giải để sinh đúng Eval hay tự suy ra từ chart (nhiễu phần Lý giải, xem Eval còn đúng không)
- Test theo từng kỹ năng Lớp 1-4 trên output thật của model (không phải test code detector)
**Gate:** không cứng — benchmark theo dõi liên tục qua các checkpoint.
**Trạng thái:** Chưa làm.

### Giai đoạn 7 — Held-out set + chuẩn bị Tầng 3
**Việc làm:** Chốt 50-100 setup tổng thể làm held-out, không đụng tới từ giờ trở đi. Bắt đầu seed labeling tay 100-200 setup khác (không trùng held-out).
**Gate:** held-out set bị khóa (checksum/git tag) để tránh contamination — làm ngay từ đầu, không trì hoãn.
**Trạng thái:** Chưa làm.

### Sơ đồ phụ thuộc

```
GĐ0 (data thô)
  → GĐ1 (test nền) → GĐ2 (thống kê) → GĐ3 (test ICT) → GĐ4 (relations)
                                                              ↓
                                                        GĐ5 (render + validate)
                                                              ↓
                                                    [train model trên data này]
                                                              ↓
GĐ6 (test model, song song từ GĐ1) ──────────────────────────┘
                                                              ↓
                                                        GĐ7 (held-out, Tầng 3)
```

---

## 11. Pipeline Judge tổng thể (5 giai đoạn sau khi có model, tham chiếu)

> Phân biệt với mục 10 (lộ trình build dataset/detector) — đây là giai đoạn SAU KHI đã có model train xong trên dataset từ Giai đoạn 5.

| Giai đoạn | Việc chính | Trạng thái |
|---|---|---|
| 0.5 | Verify giải thuật bằng golden test set tay (trước khi generate hàng loạt) | Swept: xong |
| 1 | Hard-rule detector + Q-score formula cho từng component | Swept: xong. FVG, Shift: chưa |
| 2 | Seed labeling tay (100-200 setup tổng thể) + chốt held-out set (50-100, không đụng) | Chưa |
| 3 | Active learning mở rộng label (ưu tiên case model lưỡng lự + random check) | Chưa |
| 4 | Gắn outcome thật (R multiple), đo tương quan Q-score - outcome theo cohort, fit lại trọng số (80/20 train/test) | Chưa (cần đủ mẫu) |
| 5 | Epoch hóa: đóng băng rubric, hết epoch mới đánh giá lại, chỉ swap nếu thắng rõ trên test riêng | Chưa |

**Nguyên tắc xuyên suốt mọi giai đoạn:** luôn giữ 1 phần data "chưa đụng tới" để test, ở mọi tầng (label, weight, epoch). Đây là cơ chế duy nhất ngăn rubric/model tự lừa chính nó.

---

## 12. Việc dang dở — cần làm tiếp (cập nhật theo lộ trình mục 10)

1. **(GĐ1)** Viết 6 file golden test Lớp 0-3 (`test_candle_parser_basic`, `test_slice`, `test_bull_bear`, `test_pin_bar`, `test_engulfing`, `test_swing`, `test_fvg_binary`) — ưu tiên cao nhất, chưa làm dù logic đã có sẵn trong code.
2. **(GĐ2)** Script thống kê phân phối (Doji ratio, FVG size distribution, swing frequency) để điền số thật vào ngưỡng `boundary` thay vì cảm tính.
3. **(GĐ3)** `grade_fvg` — size gap (bao nhiêu bin), % đã bị lấp tại thời điểm xét → graded Q-score. Cộng test tương ứng.
4. **(GĐ3)** `is_shift`/`is_mss` — chưa viết. Cần quyết định: chỉ cần phá 1 swing gần nhất (BOS đơn giản), hay phân biệt rõ BOS (tiếp diễn trend) vs CHoCH (đảo chiều trend). Viết test trước khi viết logic.
5. **(GĐ3)** Bổ sung golden test Swept còn thiếu: case sweep nhiều swing cùng lúc, case sweep lại swing đã bị đánh dấu phá.
6. **(GĐ4)** `build_relations` — quan hệ thứ tự/hướng giữa các event đã detect trong cùng 1 chart, có golden test riêng.
7. **(GĐ5)** Script render 4 dạng mẫu tin từ 1 bộ fact JSON.
8. **(GĐ5)** Script validate đối chiếu câu GPT sinh ra với JSON fact gốc + validate chéo giữa 4 dạng mẫu.
9. **(GĐ6)** Test kỹ năng đếm/trích dẫn nến + test "rò rỉ" Lý giải→Eval trên model nền — độc lập, có thể làm song song từ GĐ1.
10. **(GĐ7)** Chốt held-out set (50-100 setup), khóa bằng checksum/git tag ngay khi tạo, không trì hoãn.

**Đã hoãn lại có chủ đích (không phải quên):**
- HTF bias / Premium-Discount — sẽ thêm sau bằng continue-train, không gộp vào batch data hiện tại (model nhỏ, thêm context phức tạp không khả thi ngay)
- Order Block (OB) — đã nhắc tên nhưng chưa thiết kế detector

---

## 13. File đã có (tham chiếu)

- `ict_detectors.py` — `Candle`, `CandleParser` (bao gồm `is_swing_high/low`, `_find_active_swing_high/low`, `is_swept` — trả về `dict` số hóa đầy đủ: `type`, `swept_candle_idx`, `swing_idx`, `swing_level`, `depth`)
- `test_swept.py` — 8 golden test case cho `is_swept`, đã pass trên môi trường thật (Windows, pytest 9.1.1). Bao gồm 2 case quan trọng nhất: swing đã bị phá không tính sweep lại; chọn đúng swing gần nhất khi có nhiều lựa chọn trong lookback.

> Khi ghép `is_swept` + helper vào class `CandleParser` thật của bạn: dùng `CandleParser.from_candles(candles, swing_window=2)` khi viết test (không dùng constructor chính nhận `raw_text`).