# ICT — 4 loại training sample

> Xem lưu ý về nguồn thông tin ở đầu `docs/ict/detectors.md`.

Package `app/ict/` sinh 4 loại sample huấn luyện từ cùng 1 detector output
(`build_facts()`):

| Loại | Nội dung | Độ khó |
|---|---|---|
| **Swept** | Nhận diện liquidity sweep — giá quét qua vùng thanh khoản trước đó | thấp nhất (100% ở checkpoint đầu) |
| **FVG** | Nhận diện Fair Value Gap | trung bình (95%) |
| **Shift** | Market Structure Shift / Change of Character | cao hơn (80%) — cần xác nhận đúng hướng đảo chiều |
| **Synthesis** | Liên kết nhiều pattern trên cùng vùng/nến (tương tự tầng 5 của curriculum pretrain) | khó nhất — chi phí token cao, dễ bị overflow |

## Vấn đề overflow token ở Synthesis — và cách giải quyết

Sample synthesis ban đầu bị **overflow** (vượt giới hạn token cho phép của
1 sample) ở tỉ lệ rất cao:

```
37% → 21% → 7% overflow  (qua nhiều vòng validate-regenerate)
```

**Thay đổi có tác động lớn nhất**: dùng **abbreviated eval keys** (rút gọn
tên trường trong phần đánh giá/kết quả của sample) thay vì tên đầy đủ —
giảm overflow từ khoảng **82% xuống còn 26%** chỉ với thay đổi này (một
bước riêng lẻ, tách biệt khỏi loạt giảm 37→21→7% ở trên vốn là tổng hợp
nhiều thay đổi khác).

### Bài học quy trình

Giảm overflow token đòi hỏi **nhiều vòng validate → regenerate** liên tục,
không phải 1 lần sửa là xong — mỗi vòng đo lại tỉ lệ overflow thực tế trên
dữ liệu thật rồi mới quyết định thay đổi tiếp theo. Khi tối ưu token budget
cho 1 loại sample mới, nên:
1. Đo overflow rate hiện tại trước khi sửa gì.
2. Thử abbreviate tên field/key trước (thường có tác động lớn, rẻ để làm).
3. Đo lại, lặp — đừng gộp nhiều thay đổi cùng lúc rồi mới đo, sẽ khó biết
   thay đổi nào thực sự có tác dụng.

## Validate no-leakage

`validate_no_leakage()` — kiểm tra sample không vô tình rò rỉ đáp án vào
phần input. **Bài học quan trọng**: false positive trong hàm này (template
literal `"1"` trong câu tiếng Việt bị match nhầm với 1 pattern leak dạng
số) chỉ **lộ ra ở quy mô hàng triệu dòng**, không xuất hiện trong test nhỏ.

→ Khi thêm rule mới vào `validate_no_leakage()`, chạy thử trên **toàn bộ**
tập dữ liệu thật (không chỉ sample nhỏ) trước khi tin tưởng kết quả — quy
mô nhỏ dễ che giấu false positive/negative hiếm gặp.
