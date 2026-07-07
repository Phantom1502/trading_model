# Tokenizer

File: `app/memlm/tokenizer.py` (`VietnameseTokenizer`), train script:
`app/memlm/scripts/train_tokenizer.py`.

## Hai chế độ base tokenizer

| Chế độ | `pretrained_name` | `use_fast` | Vocab |
|---|---|---|---|
| Custom BPE (khuyến nghị) | path local (`"custom_tokenizer"`) | `True` | 16k (mở rộng lên 32k nếu cần) |
| PhoBERT (legacy) | `"vinai/phobert-base"` | `False` (tự fallback nếu truyền `True`, có cảnh báo) | ~64k |

Tổng vocab = BPE base + price vocab (4098 token: `O/H/L/C × 1024` + `<chart>`
+ `</chart>`).

## Train custom tokenizer

Chạy từ **thư mục gốc project** (xem `docs/conventions/running-from-root.md`):

```bash
python app/memlm/scripts/train_tokenizer.py \
    --vocab-size 16000 \
    --wiki-samples 500000 \
    --vtsnlp-samples 500000
```

`--output-dir` mặc định là `app/memlm/custom_tokenizer` — được tính theo
**vị trí file script**, không phụ thuộc thư mục bạn đang đứng khi gọi
lệnh (đã sửa; trước đây default là chuỗi tương đối `"custom_tokenizer"`,
chỉ đúng nếu cwd = `app/memlm/`).

- ByteLevel BPE (`add_prefix_space=False`), train streaming từ
  `wikimedia/wikipedia (20231101.vi)` + `VTSNLP/vietnamese_curated_dataset`
  — không load toàn bộ vào RAM.
- Special tokens theo thứ tự cố định: `<unk> <s> </s> <pad> <mask>` — ID gán
  theo thứ tự này, đừng đổi thứ tự nếu đã có checkpoint dùng vocab cũ.
- `sanity_check()` chạy round-trip test + kiểm tra không có `<unk>` phát sinh.
- Sau khi train tokenizer mới → **phải train model lại từ đầu** (checkpoint
  cũ lệch vocab hoàn toàn).

## Price token — vocab riêng, không qua `add_tokens()`

Token dạng `O_512`, `H_3`, `L_999`, `C_0`, marker `<chart>`/`</chart>` được
nhận diện bằng regex và ánh xạ vào **ID nằm ngoài dải BPE vocab** —
**không** dùng `tokenizer.add_tokens()` để tránh phải train lại BPE mỗi khi
đổi n_price_bins.

```python
PRICE_TOKEN_RE = re.compile(r"<chart>|</chart>|\b[OHLC]_\d{1,4}\b")
```

### 2 lớp bảo vệ chống match nhầm

1. **`strict_chart_mode=True`** (mặc định): chỉ parse price token khi nằm
   trong cặp `<chart>...</chart>`. Tránh việc các ký hiệu khoa học tự nhiên
   trong Wikipedia/VTSNLP (`H_0`, `C_1`, `O_157`...) hay tên chủng vi khuẩn
   bị hiểu nhầm là price token.
2. **Bin range validation** (`_split_segments_loose`): token có bin ngoài
   `[0, n_price_bins-1]` bị bỏ qua dù nằm trong `<chart>`.

Khi mix dữ liệu trading với text tổng quát (Wikipedia, VTSNLP, sách...),
**luôn bật `strict_chart_mode=True`**.

## API chính

```python
tok = VietnameseTokenizer(pretrained_name="custom_tokenizer")
ids = tok.encode(text, add_special_tokens=False)
ids_batch = tok.encode_batch(texts, add_special_tokens=False)  # nhanh hơn cho nhiều câu
text = tok.decode(ids, skip_special_tokens=True)
len(tok)  # == tok.vocab_size
```

- `encode_batch`: gom toàn bộ chunk "text" (không phải price) của nhiều câu
  lại rồi gọi tokenizer 1 lần — giảm overhead đáng kể so với gọi lặp,
  quan trọng khi dùng Fast tokenizer trên chunk lớn.
- `decode`: gom các đoạn ID liên tiếp `< base_len` thành 1 lần decode text,
  price token decode riêng lẻ từ `price_vocab_inv`.

## Việc cần nhớ khi sửa tokenizer

- Đổi `n_price_bins` → đổi toàn bộ price vocab ID → checkpoint cũ lệch,
  không resume được.
- Đổi base tokenizer (PhoBERT ↔ custom BPE, hoặc train lại BPE) → vocab lệch
  hoàn toàn → phải train model từ đầu.
- Nếu thêm token domain-specific kiểu cũ (không phải price token), dùng
  `app/memlm/scripts/add_custom_tokens.py` — **lưu ra thư mục riêng**, không sửa
  tokenizer gốc, để tránh phá checkpoint cũ (xem docstring trong script đó).