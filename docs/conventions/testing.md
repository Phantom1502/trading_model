# Convention: Testing (test-first cho data pipeline)

## Nguyên tắc cốt lõi: Ground truth luôn từ detector thật, không hand-craft

Đây là bài học quan trọng nhất rút ra từ package ICT (107 golden test) và
cũng áp dụng cho `chart_pretrain_pipeline`/`candle_parser`: bất kỳ nơi nào
sinh "đáp án đúng" cho training sample hay benchmark, đáp án đó **phải**
lấy từ output thật của hàm detector (`is_swing_high`, `is_fvg`,
`is_engulfing`, `build_facts()`...) chạy trên dữ liệu — **không viết tay**
label kỳ vọng rồi giả định detector sẽ khớp.

Lý do: hand-crafted unit test dễ bị viết theo đúng *giả định* của người
viết code, nên **cùng 1 bug logic** vừa nằm trong code sinh dữ liệu vừa
"vô tình" khớp với test hand-craft — test pass nhưng dữ liệu vẫn sai. Case
thực tế đã gặp: `_event_direction()` âm thầm trả `None` cho toàn bộ Shift
event, chỉ bị bắt khi có integration test chạy `build_facts()` thật.

## Thứ tự ưu tiên khi viết test cho pipeline dữ liệu mới

1. **Integration test trước**: chạy toàn bộ chuỗi
   `raw data → detector → sample generator` trên 1 đoạn dữ liệu thật nhỏ,
   kiểm tra output cuối cùng hợp lý (không chỉ kiểm tra hàm detector đơn lẻ
   trả đúng kiểu dữ liệu).
2. **Golden test theo case cụ thể**: mỗi pattern cần phát hiện (Swept, FVG,
   Shift, Engulfing...) nên có ít nhất 1 test case dữ liệu tối giản mà con
   người **tính tay được** đáp án đúng, dùng để lock hành vi detector khi
   refactor.
3. **Validate ở quy mô lớn, không chỉ ở quy mô test**: một số lỗi (false
   positive trong `validate_no_leakage()`, ví dụ template literal `"1"`
   trùng với pattern leak dạng số) **chỉ lộ ra ở quy mô hàng triệu dòng**.
   → Sau khi golden test pass, luôn chạy thêm 1 lượt validate trên **toàn
   bộ** tập dữ liệu thật (hoặc mẫu rất lớn) trước khi tin tưởng pipeline.

## Test-first cho data pipeline (không phải cho model code)

Nguyên tắc "viết test trước khi code" áp dụng nhất quán cho **pipeline sinh
dữ liệu** (ICT: 107 test viết trước khi tích hợp) — vì output sai ở đây là
**dữ liệu training sai**, lan truyền âm thầm vào hàng triệu sample mà không
gây lỗi crash rõ ràng nào. Đây khác với code training loop/model
architecture, nơi lỗi thường lộ ra nhanh hơn qua loss NaN, shape mismatch,
hay crash trực tiếp.

## Checklist khi thêm detector/generator mới

- [ ] Có test case tối giản (2-5 nến) tính tay được đáp án đúng?
- [ ] Có integration test chạy qua toàn bộ chuỗi build_facts() (hoặc tương
      đương) trên dữ liệu thật, không chỉ test hàm cô lập?
- [ ] Nếu detector này sinh ra label dùng làm ground truth cho benchmark,
      benchmark đó có tự động lấy lại đúng từ detector (không hard-code
      lại đáp án ở nơi khác)?
- [ ] Đã chạy validate ở quy mô toàn bộ dữ liệu thật ít nhất 1 lần trước
      khi đưa pipeline vào production (không chỉ dựa vào test suite nhỏ)?
