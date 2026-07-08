# Pretrain LLaMA tiếng Việt trên Colab — Tài liệu tổng hợp cấu hình & tối ưu

> Ghi lại các quyết định kỹ thuật đã test và xác nhận hiệu quả trong quá trình debug tốc độ train và thiết kế pipeline xử lý dữ liệu quy mô lớn (14 file × ~1B token ≈ 14B token, GPU Tesla T4 14.5GB VRAM).

---

## 1. Cấu hình model (đã test, chạy ổn định trên T4)

```python
smollm_config = LlamaConfig(
    vocab_size=VOCAB_SIZE,
    hidden_size=576,
    intermediate_size=1536,
    num_hidden_layers=30,
    num_attention_heads=9,
    num_key_value_heads=3,      # GQA tỉ lệ 3:1
    max_position_embeddings=max_seq_length,
    pad_token_id=3,
    bos_token_id=1,
    eos_token_id=2,
    tie_word_embeddings=True,
)
```

- ~130-150M tham số (kiến trúc kiểu SmolLM-135M).
- **Khởi tạo với `attn_implementation="sdpa"` ngay từ đầu**, không dùng `eager`:

```python
tiny_model = LlamaForCausalLM._from_config(
    smollm_config,
    attn_implementation="sdpa"
)
```

> Đo được: chuyển từ `eager` → `sdpa` giúp tốc độ tăng từ `0.12 it/s` → `0.40 it/s` (model nhỏ, seq 2048) — cải thiện hơn 3 lần. Không cần cài `flash-attn`, `sdpa` có sẵn trong PyTorch ≥ 2.0.

### `max_seq_length`
- Ban đầu dùng `2048`, sau giảm xuống `1024` để cân bằng VRAM/tốc độ khi tăng batch size.
- Attention cost tăng theo O(n²) — giảm seq length là đòn bẩy tốc độ mạnh, nhưng đánh đổi khả năng học ngữ cảnh dài. Với model nhỏ (30 layer, hidden 576), `1024` là điểm cân bằng hợp lý đã chọn.

---

## 2. TrainingArguments đã xác nhận hoạt động ổn định

```python
training_args = TrainingArguments(
    output_dir="/content/drive/MyDrive/llama_project/checkpoints",  # PHẢI nằm trên Drive

    per_device_train_batch_size=16,
    gradient_accumulation_steps=8,         # effective batch ≈ 128
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},

    learning_rate=3e-4,
    weight_decay=0.1,
    adam_beta1=0.9,
    adam_beta2=0.95,
    fp16=True,

    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    per_device_eval_batch_size=16,

    save_strategy="steps",
    save_steps=50,                         # ~40 phút/checkpoint ở tốc độ hiện tại — chỉnh theo độ chấp nhận rủi ro ngắt session
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    report_to="none",
)
```

### Vì sao cần `gradient_checkpointing=True`
- Với 30 layer, tắt checkpointing khiến VRAM tăng **~4 lần** → OOM ngay cả ở batch size nhỏ.
- Bắt buộc phải bật, chấp nhận đánh đổi ~20-40% tốc độ để đổi lấy khả năng tăng batch size.
- `use_reentrant=False` là implementation mới hơn, nên dùng thay vì mặc định `True`.

### Về LR scheduler trong `TrainingArguments`
`lr_scheduler_type`, `warmup_ratio`/`warmup_steps` trong `TrainingArguments` **chỉ có tác dụng khi không tự truyền `optimizers=(...)` vào `Trainer`**. Vì pipeline nhiều-shard (mục 5.2) luôn truyền optimizer/scheduler thủ công để giữ LR liên tục xuyên suốt toàn bộ 14B token, các tham số này trong `TrainingArguments` bị bỏ qua hoàn toàn — không cần khai báo ở đây, chỉ cần đúng ở `get_cosine_schedule_with_warmup(...)` (mục 5.2).

### Vì sao batch size không nên chạy sát trần VRAM
- Test batch=16 (dùng gần hết 14.56GB, ở cấu hình `max_seq_length=1024` trước khi bật `use_reentrant=False`) và batch=8 (còn nhiều headroom) cho tốc độ **tương đương nhau** ở cấu hình hiện tại → xác nhận nghẽn không nằm ở VRAM fragmentation, mà là GPU compute-bound thật sự (xem mục 4).
- Sau khi bật `gradient_checkpointing_kwargs={"use_reentrant": False}`, quan sát lại: `batch_size=16` chỉ còn dùng **~8GB VRAM** (còn dư ~6.5GB) → đủ headroom an toàn, đã chốt dùng `per_device_train_batch_size=16, gradient_accumulation_steps=8` (effective batch ≈ 128, giữ nguyên).

---

## 3. Pipeline xử lý dữ liệu

### 3.1 Đọc parquet local (thay cho `load_dataset` từ Hub)

```python
raw_datasets = load_dataset(
    "parquet",
    data_files={
        "train": f"{train_dir}/*.parquet",
        "validation": f"{val_dir}/*.parquet",
    },
)
```

- Dùng `remove_columns=dataset.column_names` thay vì liệt kê cứng tên cột — an toàn hơn khi schema parquet khác Wikipedia gốc (`text`, `source`, `token_length`, `meta`).
- Dataset đã được lọc/chuẩn bị có chủ đích trước khi đóng gói → **không shuffle, không lọc lại**, giữ nguyên thứ tự `select(range(...))`.

### 3.2 Tokenize + group thành block cố định

```python
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=False)

def group_texts(examples):
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated["input_ids"])
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    result = {
        "input_ids": [concatenated["input_ids"][i:i+block_size]
                       for i in range(0, total_length, block_size)]
    }
    return result
    # KHÔNG lưu attention_mask, labels — DataCollatorForLanguageModeling(mlm=False)
    # tự sinh labels từ input_ids, giảm ~2/3 dung lượng cache
```

### 3.3 Validation set — xử lý 1 lần, cố định suốt toàn bộ quá trình

Không thay đổi theo shard/file train. Cache riêng, load lại mỗi session:

```python
if not os.path.exists(val_cache):
    lm_val = ...  # tokenize + group như trên
    lm_val.save_to_disk(val_cache)
lm_val = load_from_disk(val_cache)
```

---

## 4. Kết luận về tốc độ (đã xác nhận qua benchmark cô lập)

- Benchmark forward+backward thuần (bỏ qua Trainer/dataloader) cho tốc độ gần như trùng khớp tốc độ thực tế của Trainer → **xác nhận nghẽn là GPU compute-bound**, không phải do dataloader/CPU.
- GPU xác nhận đúng là Tesla T4 thật (14912MB, 40 SM), không bị giới hạn/chia sẻ tài nguyên.
- Nguyên nhân tốc độ chậm ở mức hợp lý về mặt kỹ thuật, đến từ tổ hợp:
  1. T4 (kiến trúc Turing, 2018) — hiệu suất fp16 thực tế thường chỉ đạt 15-25% TFLOPS lý thuyết.
  2. 30 layer → overhead kernel-launch nhiều lần mỗi step, với `hidden_size=576` chưa đủ lớn để tận dụng hết Tensor Core mỗi lần gọi.
  3. `gradient_checkpointing=True` bắt buộc recompute forward trong backward → gần như nhân đôi phần compute forward.
- Kết luận: **không cần tiếp tục điều tra tốc độ**, đây là giới hạn phần cứng + kiến trúc + ràng buộc VRAM, không phải lỗi cấu hình.

---

## 5. Chiến lược xử lý dữ liệu quy mô lớn (14 file × ~1B token)

### 5.1 Cấu trúc
- 14 file gốc ~1B token/file → tách thành nhiều shard nhỏ hơn (parquet), gộp phẳng vào 1 thư mục chung:
```
train_shards/file01_shard000.parquet
train_shards/file01_shard001.parquet
...
train_shards/file14_shard049.parquet
```
- Vòng lặp train xử lý tuần tự theo danh sách shard đã `sorted()`, không cần phân biệt file gốc.

### 5.2 Optimizer & LR scheduler — tạo 1 lần duy nhất cho toàn bộ 14B token

**Vấn đề nếu làm sai:** nếu mỗi shard tạo `Trainer` mới theo kiểu mặc định, `num_training_steps` sẽ được tính lại riêng cho từng shard → cosine LR decay về 0 rồi reset lại ở shard kế tiếp, gãy đường cong learning rate.

**Giải pháp:**
```python
TOTAL_TOKENS_ESTIMATE = 14_000_000_000
TOTAL_STEPS = TOTAL_TOKENS_ESTIMATE // (block_size * effective_batch)
WARMUP_STEPS = int(0.03 * TOTAL_STEPS)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4,
                               betas=(0.9, 0.95), weight_decay=0.1)
scheduler = get_cosine_schedule_with_warmup(optimizer,
                                             num_warmup_steps=WARMUP_STEPS,
                                             num_training_steps=TOTAL_STEPS)

# Truyền vào MỌI Trainer của MỌI shard:
trainer = Trainer(..., optimizers=(optimizer, scheduler))
```

### 5.3 Quản lý cache dataset theo shard (giới hạn dung lượng Drive)

- Ước tính: 14B token × ~4 byte/token (int) ≈ **52-56GB** nếu giữ cache `input_ids` của toàn bộ — vượt quota Drive thông thường.
- **Chỉ giữ cache của shard đang xử lý**, xoá cache shard trước sau khi đã chuyển tiếp thành công:
```python
if i > 0:
    prev_cache = f"{cache_dir}/shard_{i-1}"
    if os.path.exists(prev_cache):
        shutil.rmtree(prev_cache)
```
- **Không xoá file `.parquet` gốc** (raw shard) — chỉ xoá cache đã tokenize, để có thể tokenize lại bất cứ lúc nào (đổi tokenizer, đổi `max_seq_length`, debug).

### 5.4 Resume/ngắt session (Colab)

- `output_dir` **bắt buộc** nằm trên Google Drive (`/content/drive/...`), không dùng `/content/...` local (mất khi runtime reset).
- File `shard_state.json` lưu `shard_index` hiện tại — chỉ cập nhật **sau khi** shard train xong hoàn toàn:
```python
state["shard_index"] = i + 1
json.dump(state, open(state_path, "w"))
```
- Resume checkpoint trong shard đang dở bằng `get_last_checkpoint(output_dir)` + `trainer.train(resume_from_checkpoint=last_ckpt)`.
- Checkpoint HF lưu đủ `optimizer.pt`, `scheduler.pt`, `rng_state.pth` → resume đúng khôi phục cả vị trí trên đường cong cosine LR, không bị reset.

### 5.5 Vòng lặp tổng thể (khung sườn)

```python
for i in range(state["shard_index"], len(shard_files)):
    shard_cache = f"{cache_dir}/shard_{i}"
    if os.path.exists(shard_cache):
        lm_shard = load_from_disk(shard_cache)
    else:
        raw = load_dataset("parquet", data_files=shard_files[i])["train"]
        tok = raw.map(tokenize_function, batched=True, num_proc=os.cpu_count(),
                       remove_columns=raw.column_names)
        lm_shard = tok.map(group_texts, batched=True, num_proc=os.cpu_count())
        lm_shard.save_to_disk(shard_cache)

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=lm_shard, eval_dataset=lm_val,
        data_collator=data_collator,
        optimizers=(optimizer, scheduler),
    )
    trainer.train(resume_from_checkpoint=get_last_checkpoint(output_dir))

    state["shard_index"] = i + 1
    json.dump(state, open(state_path, "w"))

    if i > 0:
        shutil.rmtree(f"{cache_dir}/shard_{i-1}", ignore_errors=True)
```

---

---

## 6. Backup & resume qua Hugging Face Hub

Bổ sung lớp an toàn thứ 2 ngoài Google Drive — phòng khi Drive hết quota hoặc bị lỗi trong quá trình train kéo dài ~2 tháng.

### 6.1 Đăng nhập (1 lần mỗi session)
```python
from huggingface_hub import login
login(token="hf_xxx")   # ưu tiên notebook_login() để không hardcode token
```

### 6.2 Push checkpoint đầy đủ tự động mỗi lần save
```python
training_args = TrainingArguments(
    ...
    push_to_hub=True,
    hub_model_id="your-username/llama-vi-pretrain",
    hub_private_repo=True,          # optimizer state không nên public
    hub_strategy="checkpoint",      # push full checkpoint (model+optimizer+scheduler+rng) lên branch "last-checkpoint"
    ...
)
```
- `hub_strategy="checkpoint"` tự ghi đè checkpoint cũ trên Hub mỗi lần save, không tích luỹ dung lượng như nhiều checkpoint trên Drive.
- Không cần đổi gì ở vòng lặp nhiều-shard (mục 5.5) — hoạt động độc lập với việc tự truyền `optimizers=(optimizer, scheduler)`.

### 6.2b Tokenizer — push 1 lần duy nhất, không lặp lại theo checkpoint
Tokenizer cố định xuyên suốt toàn bộ quá trình, không cần push lại mỗi lần save:
```python
tokenizer.push_to_hub("your-username/llama-vi-pretrain", private=True)
```
- Chạy 1 lần trước khi vào vòng lặp train (hoặc ngay sau khi setup tokenizer).
- Nằm trên branch `main` (mặc định) của cùng repo, không bị các lần push checkpoint (branch `last-checkpoint`) ghi đè.
- Khi load lại: `AutoModelForCausalLM.from_pretrained(repo_id)` + `AutoTokenizer.from_pretrained(repo_id)` dùng chung 1 `repo_id`, không cần quản lý 2 nơi riêng.

### 6.3 Resume — ưu tiên Drive, fallback sang Hub
```python
from huggingface_hub import snapshot_download
from transformers.trainer_utils import get_last_checkpoint

last_ckpt = get_last_checkpoint(output_dir)   # thử Drive trước, nhanh hơn

if last_ckpt is None:
    try:
        last_ckpt = snapshot_download(
            repo_id="your-username/llama-vi-pretrain",
            revision="last-checkpoint",
        )
    except Exception:
        last_ckpt = None

trainer.train(resume_from_checkpoint=last_ckpt)
```

### 6.4 Lưu ý dung lượng/băng thông
- Checkpoint đầy đủ (~135M params, fp16 weights + Adam optimizer state fp32 x2 + scheduler + rng) ước tính **~1-1.5GB/lần push**.
- Với `save_steps=50` (~40 phút/lần ở tốc độ hiện tại), push liên tục trong ~2 tháng tốn băng thông đáng kể — cân nhắc giãn tần suất push riêng (khác `save_steps` local) nếu cần tiết kiệm.

---

---

## 8. Cách chạy trên Colab (từng bước)

### Cell 1 — Mount Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Cell 2 — Cài đặt thư viện
```python
!pip install -q -U transformers datasets accelerate huggingface_hub
```
> Không cần cài `flash-attn` — `sdpa` có sẵn trong PyTorch ≥ 2.0 (Colab đã có sẵn).

### Cell 3 — Đăng nhập Hugging Face
```python
from huggingface_hub import notebook_login
notebook_login()
```
Nhập token có quyền **write** (tạo tại https://huggingface.co/settings/tokens). Dùng `notebook_login()` thay vì hardcode token vào code.

### Cell 4 — Upload tokenizer & đưa dữ liệu lên đúng vị trí
- Copy thư mục tokenizer (`custom_tokenizer_llama`) vào `/content/custom_tokenizer_llama` (upload thủ công hoặc copy từ Drive):
```python
!cp -r /content/drive/MyDrive/llama_project/custom_tokenizer_llama /content/custom_tokenizer_llama
```
- Đảm bảo cấu trúc thư mục trên Drive khớp với `DRIVE_ROOT` trong script:
```
llama_project/
├── train_shards/          # các file .parquet đã tách shard từ 14 file gốc
├── val/                   # file .parquet validation
├── checkpoints/           # tự tạo, Trainer ghi vào đây
├── train_shards_cache/    # tự tạo, cache dataset đã tokenize
└── lm_val_cache/          # tự tạo, cache val set
```

### Cell 5 — Upload script và chỉnh cấu hình
Upload `train_llama_vi.py` lên Colab (kéo-thả vào file explorer bên trái, hoặc lưu thẳng lên Drive rồi copy), sau đó mở file chỉnh phần **mục 0 — CẤU HÌNH CHUNG**:
- `DRIVE_ROOT`, `TOKENIZER_DIR` khớp đúng path thật.
- `HUB_MODEL_ID` đổi thành repo Hugging Face thật của bạn (ví dụ `"tuanx/llama-vi-135m"`).
- `TOTAL_TOKENS_ESTIMATE` chỉnh lại nếu biết chính xác tổng token sau khi tách shard.

### Cell 6 — Chạy training
```python
import sys
sys.path.append('/content')   # nếu để script ở /content
from train_llama_vi import main

main()
```
Hoặc chạy thẳng qua terminal cell nếu để script trên Drive:
```python
!python /content/drive/MyDrive/llama_project/train_llama_vi.py
```

### Cell 7 (chỉ dùng khi nghi Drive mất checkpoint) — Khôi phục từ Hub trước khi chạy lại
```python
from train_llama_vi import resume_checkpoint_from_hub_if_needed
resume_checkpoint_from_hub_if_needed()

from train_llama_vi import main
main()
```

### Sau khi Colab bị ngắt session
- Mount lại Drive (Cell 1), cài lại thư viện (Cell 2, Colab runtime mới không giữ package đã cài), đăng nhập lại Hub (Cell 3).
- **Không cần** upload lại tokenizer/dữ liệu nếu đã nằm trên Drive từ trước.
- Chạy lại Cell 6 — script tự đọc `shard_state.json` trên Drive để biết đang ở shard nào, và tự phân biệt resume giữa-shard hay bắt đầu shard mới (mục 6 trong script).

### Mẹo giữ session không bị ngắt sớm (Colab free-tier)
- Tránh để tab notebook hoàn toàn không tương tác quá lâu (một số script tự động click page bằng JS console tồn tại nhưng vi phạm ToS Colab, không khuyến khích).
- Cân nhắc Colab Pro/Pro+ nếu cần session dài hơn và ít bị ngắt đột ngột hơn, đặc biệt hợp lý cho lộ trình train ~2 tháng.
- Luôn để `save_steps` đủ nhỏ (hiện `50`, ~40 phút/lần) so với chu kỳ ngắt kỳ vọng, để mỗi lần mất kết nối không mất quá nhiều tiến độ.

---

## 9. Việc chưa làm / cần quyết định thêm

- [ ] Xác định `save_steps` cụ thể theo thời gian thực (không phải số step) để khớp với chu kỳ ngắt session thực tế của Colab.
- [ ] Xác nhận dung lượng Drive khả dụng đủ cho checkpoint (model + optimizer state) lặp lại trong suốt ~2 tháng train.
- [ ] Quyết định số lượng shard/file để cân bằng giữa thời gian tokenize mỗi lần và tần suất checkpoint giữa các shard.
- [ ] Theo dõi `eval_loss` định kỳ để phát hiện sớm nếu có vấn đề (overfitting, LR không phù hợp) trước khi đi hết 14B token.