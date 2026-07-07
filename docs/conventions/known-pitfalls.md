# Convention: Known Pitfalls (bài học đã trả giá, đừng lặp lại)

Tổng hợp các lỗi/pitfall đã gặp thực tế trong project — mỗi mục nên được
kiểm tra lại khi review code liên quan.

## 1. Effective batch size quá nhỏ → val loss nhiễu (noisy oscillation)

**Triệu chứng**: val loss dao động mạnh giữa các lần eval, không thấy trend
giảm rõ ràng dù model vẫn đang học.

**Nguyên nhân**: batch hiệu dụng (`batch_size * grad_accum` tính theo
token) quá nhỏ so với mức cần thiết cho model size này.

**Fix đã verify**: `batch=32, grad_accum=64` → đạt mốc **~1M token/step**.
Xem `get_100m_config()` trong `config.py`. Khi tạo config mới cho model
size khác, **luôn tính lại** effective token/step và giữ ở mức tương đương
(scale theo Chinchilla-style budget), đừng chỉ copy `batch_size`/`grad_accum`
từ config cũ nếu `seg_len`/`d_model` đã đổi.

## 2. Token-level routing phá vỡ attention

**Nguyên nhân**: nếu route (skip/run) quyết định theo *từng token* nhưng
áp dụng kiểu masking khiến layer sau chỉ thấy context "một phần" (token bị
skip coi như không tồn tại với layer đó) → attention nhận context không
đầy đủ, không nhất quán giữa các layer.

**Giải pháp đã áp dụng**: `DepthRouter` route theo **sequence-level, mỗi
token skip = giữ nguyên hidden state y hệt** (không xoá/che token đó khỏi
attention của layer khác) — xem `docs/model/router-moe.md`. Token bị skip
vẫn "có mặt" đầy đủ cho các token khác attend tới, chỉ là bản thân nó không
được update thêm ở layer đó.

## 3. Memory gradient flow cần write-before-read (áp dụng nếu dựng lại Context Memory)

> Ghi chú: memory module đã được tách ra project riêng, `app/memlm/` hiện
> tại không còn chứa cơ chế này — mục này giữ lại vì bài học vẫn áp dụng
> nếu sau này tích hợp lại 1 cơ chế state xuyên-segment tương tự.

**Nguyên tắc**: `memory_new` (kết quả write, vẫn còn nằm trong computation
graph) phải được dùng làm input cho bước **read** kế tiếp, để gradient chảy
ngược qua `write_attn`. Nếu dùng **EMA update** cho memory thay vì giữ
nguyên trong graph, gradient sẽ **không** chảy qua đúng đường — mâu thuẫn
với mục tiêu `sim_loss` (đang cố dạy memory học biểu diễn hữu ích qua
gradient).

**Hệ quả liên quan**: `init_memory` phải dùng **noise nhỏ**, không phải
zeros — nếu `Wq(zeros) = 0`, các projection `Wq`/`Wk` của `write_attn` bị
**đói gradient** hoàn toàn ở bước đầu (gradient của output theo các weight
này bằng 0 khi input là 0), làm chậm/hỏng quá trình học ngay từ đầu.

**Nếu `alpha` (hệ số EMA) là parameter học được**: phải **fix `alpha`**
(không học) khi dùng `detach_memory()` sau mỗi step — `alpha` học được sẽ
nhận gradient = 0 một khi memory bị detach khỏi graph, khiến việc để nó
"trainable" trở thành ảo giác (không bao giờ thực sự update).

## 4. Checkpoint compatibility — dead parameters

**Vấn đề**: parameter được khai báo trong `__init__` (vd `norm1`, `Wq` cũ)
nhưng **không được gọi** trong `forward` — vẫn tốn VRAM, tốn dung lượng
checkpoint, và gây `strict=True` load failure sau khi refactor xoá param
đó (checkpoint cũ có key thừa/thiếu so với model mới).

**Kỷ luật**: mỗi khi refactor architecture, **grep lại toàn bộ
`self.xxx = nn.Linear(...)` / `nn.Parameter(...)`** và xác nhận từng cái
thực sự được dùng trong `forward()`. Xoá ngay khi phát hiện dead param —
đừng để tích luỹ "để sau dọn", vì càng để lâu càng khó biết checkpoint nào
còn phụ thuộc vào key nào.

## 5. `optimizer.load_state_dict()` âm thầm phục hồi LR cũ

**Triệu chứng**: đổi `cfg.train.lr` cho round train mới, resume từ
checkpoint cũ, nhưng LR thực tế dùng trong training vẫn là LR cũ.

**Nguyên nhân**: `torch.optim.Optimizer.load_state_dict()` khôi phục toàn
bộ state bao gồm cả `lr` đã lưu trong optimizer state — ghi đè lên
`cfg.train.lr` mới truyền vào.

**Fix**: cờ `reset_lr_for_new_round=True` trong `run_pretrain()`/`main()` —
khi bật, optimizer được **tạo lại từ đầu** với `lr` mới, và scheduler được
"tua" (`step()` lặp lại `global_step` lần) để về đúng vị trí lịch trình LR
hiện tại. Xem `docs/training/pretrain-pipeline.md`.

**Quy tắc nhớ**: đổi hyperparameter train (không đổi kiến trúc) + muốn nó
thực sự có hiệu lực khi resume → **luôn** bật `reset_lr_for_new_round=True`.
Chỉ đổi `cfg.train.lr` mà quên bật cờ này là lỗi rất dễ mắc và rất khó phát
hiện (training vẫn chạy bình thường, chỉ là LR không đổi như mong đợi).

## 6. Price token regex quá lỏng lẻo bắt nhầm dữ liệu tổng quát

**Triệu chứng**: text tiếng Việt/Wikipedia lẫn ký hiệu khoa học (`H_0`,
`C_1`, `O_157`) hoặc tên chủng vi khuẩn bị hiểu nhầm thành price token khi
mix dữ liệu trading với corpus tổng quát.

**Fix**: 2 lớp bảo vệ — `strict_chart_mode=True` (chỉ parse trong
`<chart>...</chart>`) + bin range validation (loại token có bin ngoài
`[0, n_price_bins-1]`). Xem `docs/model/tokenizer.md`. **Luôn** bật
`strict_chart_mode=True` khi training data là hỗn hợp (trading + text
tổng quát) — chỉ tắt khi chắc chắn 100% dữ liệu chỉ chứa chart token.

## 7. Ground truth hand-crafted vs detector thật

Xem chi tiết ở `docs/conventions/testing.md` — tóm tắt: mọi label dùng làm
"đáp án đúng" (training sample hay benchmark) phải bắt nguồn từ detector
chạy thật, không viết tay.

## 8. Đường dẫn phụ tính sai cấp thư mục khi copy-paste giữa các script

**Vấn đề cụ thể đã phát hiện**: `app/utils/build_dataset_to_parquet.py` có
đoạn tính đường dẫn tokenizer copy từ 1 script khác nằm ở **thư mục gốc**
(`gen_trading_data.py`):

```python
base_dir = os.path.dirname(os.path.abspath(__file__))
tok_path = os.path.join(base_dir, "app", "memlm", "custom_tokenizer")
```

Đoạn code đúng khi `__file__` nằm ở gốc project, nhưng
`build_dataset_to_parquet.py` lại nằm ở `app/utils/` — nên `base_dir` đã
là `.../app/utils`, và nối thêm `"app/memlm/..."` cho ra đường dẫn sai
`.../app/utils/app/memlm/custom_tokenizer` (không tồn tại).

**Bài học**: khi copy 1 đoạn tính `base_dir`/đường dẫn phụ giữa các file
ở **cấp thư mục khác nhau**, không copy nguyên văn — phải tính lại số
cấp `dirname()` cần thiết, hoặc tốt hơn là dùng chung 1 hàm tìm thư mục
gốc project (đi ngược lên tới khi thấy `app/`) thay vì hard-code số cấp —
xem `docs/conventions/running-from-root.md` cho pattern `_find_repo_root`
hiện đang dùng.

**Đã sửa** cùng đợt chuyển toàn bộ sang chạy từ thư mục gốc — xem
`docs/conventions/running-from-root.md`.

## 9. Script tự chạy code ở module scope thay vì `if __name__ == "__main__":`

**Vấn đề**: `app/utils/chart_generate_ds.py` (bản trước khi sửa) gọi
`builder.build_from_file(...)` ngay ở top-level của module — nghĩa là chỉ
cần `import` file này (dù không cố ý chạy nó) cũng sẽ **tự động thực thi**
toàn bộ pipeline sinh dataset. Đây là side-effect nguy hiểm nếu file bị
import gián tiếp qua 1 module khác trong tương lai.

**Quy tắc**: mọi script sinh dữ liệu/chạy pipeline nên bọc phần thực thi
chính trong `if __name__ == "__main__":`, kể cả khi hiện tại chỉ được
dùng như standalone script — phòng trường hợp sau này có nhu cầu import
1 hàm từ file đó mà không muốn kích hoạt toàn bộ side-effect.