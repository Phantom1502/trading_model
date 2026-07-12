"""
train.py — Pretrain model đọc giá chuyên biệt (price-only decoder)
================================================================================
KHÁC HẲN bản tiếng Việt cũ ở 2 điểm cốt lõi:

1. KHÔNG group_texts() nối nhiều document rồi cắt lại theo block_size.
   Data giờ đã ĐỀU: mỗi row parquet là ĐÚNG 1 doc 100 nến (402 token cố
   định, xem app.llama.config.DOC_LEN_TOKENS). Nối-cắt theo block_size sẽ
   PHÁ VỠ ranh giới doc, làm mất đúng ý nghĩa "1 chuỗi 100 nến liên tục"
   mà toàn bộ thiết kế (tokenizer, max_position_embeddings, cách dùng lúc
   inference) đang dựa vào — đây là lý do KHÔNG tái dùng group_texts của
   bản cũ, dù trông tiện lợi.

2. KHÔNG dùng hệ thống multi-shard/Drive/state.json của bản cũ (thiết kế
   cho stream 14B token từ HF Hub). Ở đây file train có thể vài GB (đã
   gộp 5 symbol) — dùng `datasets` STREAMING trực tiếp trên file local
   (không cần tự chia shard/quản lý state), Trainer nhận IterableDataset
   bình thường. Val (nhỏ hơn nhiều, tách theo mốc thời gian >= 2026) load
   THƯỜNG — cần biết độ dài để eval loop chạy đúng (IterableDataset không
   có len()). max_steps được tính tự động từ tổng số row trong metadata
   parquet (chỉ đọc footer, không đọc data) — KHÔNG dùng num_train_epochs
   trực tiếp vì streaming dataset không có len().

Giả định cấu trúc data (xem app.llama.config.DataConfig):
    - train_parquet_path : 1 file (hoặc glob pattern) parquet đã shuffle
      trộn 5 symbol, mỗi row là 1 doc "<chart> ... </chart>" đúng 100 nến.
    - val_parquet_path   : cùng format, lấy từ mốc thời gian >= 2026,
      KHÔNG overlap với train (đã tách khi build dataset, xem lịch sử chat).

Chạy:
    from app.llama.config import Config
    from app.llama.train import main

    cfg = Config()
    cfg.data.train_parquet_path = "data/train_price.parquet"
    cfg.data.val_parquet_path   = "data/val_price.parquet"
    main(cfg)
"""

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = _THIS_DIR
while not os.path.isdir(os.path.join(_REPO_ROOT, "app")):
    _parent = os.path.dirname(_REPO_ROOT)
    if _parent == _REPO_ROOT:
        raise RuntimeError("Không tìm thấy thư mục gốc project (thư mục chứa 'app/').")
    _REPO_ROOT = _parent
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    LlamaConfig,
    LlamaForCausalLM,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

from app.llama.config import Config, ModelConfig, DOC_LEN_TOKENS


# =====================================================================================
# 1. TOKENIZER
# =====================================================================================

def load_tokenizer(cfg: Config):
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.tokenizer.pretrained_name,
        use_fast=cfg.tokenizer.use_fast,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<pad>"
    return tokenizer


# =====================================================================================
# 2. DATASET — mỗi row = 1 doc CỐ ĐỊNH, KHÔNG group/nối (xem docstring module)
# =====================================================================================

def _count_parquet_rows(path_or_glob: str) -> int:
    """
    Đếm tổng số row bằng METADATA parquet (chỉ đọc footer, KHÔNG đọc data)
    — rẻ ngay cả với file vài GB, kể cả khi nằm trên HF Hub (path dạng
    "hf://datasets/<repo_id>/<file>.parquet") thay vì local. Hỗ trợ glob
    pattern (nhiều file) ở cả 2 trường hợp.
    """
    import glob
    import pyarrow.parquet as pq

    if path_or_glob.startswith("hf://"):
        from huggingface_hub import HfFileSystem
        fs = HfFileSystem()
        fs_path = path_or_glob[len("hf://"):]
        files = fs.glob(fs_path) if any(ch in fs_path for ch in "*?[") else [fs_path]
        if not files:
            raise FileNotFoundError(f"Không tìm thấy file trên HF Hub khớp: {path_or_glob}")
        return sum(pq.ParquetFile(f, filesystem=fs).metadata.num_rows for f in files)

    files = sorted(glob.glob(path_or_glob)) if any(ch in path_or_glob for ch in "*?[") else [path_or_glob]
    if not files:
        raise FileNotFoundError(f"Không tìm thấy file parquet khớp: {path_or_glob}")
    return sum(pq.ParquetFile(f).metadata.num_rows for f in files)


def tokenize_function(examples, tokenizer, text_col: str):
    """
    Tokenize thẳng, KHÔNG group_texts. Mỗi row ra ĐÚNG DOC_LEN_TOKENS token
    (vì data đã đều 100 nến/doc) — truncation/padding chỉ là lưới an toàn,
    không kỳ vọng thực sự cắt/pad nếu data đúng như thiết kế.
    """
    result = tokenizer(
        examples[text_col],
        truncation=True,
        max_length=DOC_LEN_TOKENS,
        padding="max_length",
    )
    return result


def build_datasets(cfg: Config, tokenizer):
    """
    Train: STREAMING (file lớn, vd 3GB — không load hết vào RAM). Tokenize
    xảy ra "lazy" mỗi khi iterate, không có bước .map() chạy hết 1 lần.

    Val: load THƯỜNG (Dataset, không streaming) — Trainer cần biết độ dài
    để chạy eval loop bình thường (IterableDataset không có len()).
    """
    if not cfg.data.train_parquet_path:
        raise ValueError(
            "cfg.data.train_parquet_path chưa được đặt. "
            "Ví dụ: cfg.data.train_parquet_path = 'data/train_price.parquet'"
        )
    if not cfg.data.val_parquet_path:
        raise ValueError(
            "cfg.data.val_parquet_path chưa được đặt — val PHẢI tách riêng "
            "theo mốc thời gian (>= 2026), không dùng random split trên "
            "cùng file train (xem lịch sử quyết định, tránh leak thời gian)."
        )

    raw = load_dataset(
        "parquet",
        data_files={
            "train": cfg.data.train_parquet_path,
            "validation": cfg.data.val_parquet_path,
        },
    )

    tokenized = raw.map(
        lambda ex: tokenize_function(ex, tokenizer, cfg.data.text_col),
        batched=True,
        num_proc=os.cpu_count(),
        remove_columns=raw["train"].column_names,
        desc="Tokenizing (mỗi row = 1 doc cố định, không group)",
    )

    # Kiểm tra nhanh 1 sample đầu — bắt sớm nếu data không đều như kỳ vọng,
    # thay vì để lộ ra giữa chừng training bằng lỗi khó hiểu.
    sample_len = len(tokenized["train"][0]["input_ids"])
    if sample_len != DOC_LEN_TOKENS:
        print(
            f"  CẢNH BÁO: doc đầu tiên sau tokenize dài {sample_len} token, "
            f"khác DOC_LEN_TOKENS={DOC_LEN_TOKENS} kỳ vọng. Kiểm tra lại "
            f"pipeline build dataset (mỗi row có đúng 100 nến không?)."
        )

    return tokenized["train"], tokenized["validation"]


# =====================================================================================
# 3. MODEL
# =====================================================================================

def build_model(model_cfg: ModelConfig) -> LlamaForCausalLM:
    config = LlamaConfig(
        vocab_size=model_cfg.vocab_size,
        hidden_size=model_cfg.hidden_size,
        intermediate_size=model_cfg.intermediate_size,
        num_hidden_layers=model_cfg.num_hidden_layers,
        num_attention_heads=model_cfg.num_attention_heads,
        num_key_value_heads=model_cfg.num_key_value_heads,
        max_position_embeddings=model_cfg.max_position_embeddings,
        pad_token_id=3,   # khớp SPECIAL_TOKENS trong config.py: <pad>=3
        bos_token_id=1,   # <bos>=1
        eos_token_id=2,   # <eos>=2
        tie_word_embeddings=model_cfg.tie_word_embeddings,
    )
    return LlamaForCausalLM._from_config(config, attn_implementation="sdpa")


# =====================================================================================
# 4. TRAINING ARGS
# =====================================================================================

def build_training_args(cfg: Config) -> TrainingArguments:
    tc = cfg.train
    kwargs = dict(
        output_dir=tc.output_dir,
        per_device_train_batch_size=tc.per_device_train_batch_size,
        per_device_eval_batch_size=tc.per_device_eval_batch_size,
        gradient_accumulation_steps=tc.gradient_accumulation_steps,
        gradient_checkpointing=tc.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if tc.gradient_checkpointing else None,
        learning_rate=tc.learning_rate,
        weight_decay=tc.weight_decay,
        lr_scheduler_type="cosine",
        warmup_ratio=tc.warmup_ratio,
        num_train_epochs=tc.num_train_epochs,
        fp16=tc.fp16,
        logging_steps=tc.logging_steps,
        eval_strategy="steps",
        eval_steps=tc.eval_steps,
        save_strategy="steps",
        save_steps=tc.save_steps,
        save_total_limit=tc.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
    )

    if cfg.hub.repo_id:
        kwargs.update(
            push_to_hub=True,
            hub_model_id=cfg.hub.repo_id,
            hub_private_repo=True,
            hub_strategy="checkpoint",
        )

    return TrainingArguments(**kwargs)


# =====================================================================================
# 5. MAIN
# =====================================================================================

def resume_checkpoint_from_hub_if_needed(cfg: Config):
    """
    Ưu tiên checkpoint LOCAL (output_dir) trước — nhanh hơn, không cần
    mạng. Chỉ tải từ Hub nếu không có checkpoint local nào (vd session
    Colab/Kaggle bị ngắt, output_dir không nằm trên storage sống sót qua
    session).

    QUAN TRỌNG: tùy phiên bản `transformers`, hub_strategy="checkpoint"
    push checkpoint theo 1 trong 2 cách khác nhau:
        (a) Bản MỚI (đã xác nhận qua thực tế) — push vào THƯ MỤC CON
            "last-checkpoint/" NGAY TRONG branch "main" (không phải
            branch/revision riêng). Đây là cách phổ biến hiện tại.
        (b) Bản CŨ hơn — push lên branch/revision RIÊNG tên
            "last-checkpoint" (cách docstring cũ của app/llama/train.py
            gốc — bản tiếng Việt — mô tả).

    Thử (a) trước (khớp thực tế đã quan sát), fallback sang (b) nếu (a)
    không tìm thấy, để code không phụ thuộc cứng vào 1 phiên bản.

    Cần cfg.hub.repo_id đã set VÀ đã login (huggingface_hub.login() hoặc
    HF_TOKEN) với quyền đọc — checkpoint push với hub_private_repo=True
    nên không tải được nếu chưa đăng nhập.
    """
    local_ckpt = get_last_checkpoint(cfg.train.output_dir)
    if local_ckpt:
        return local_ckpt

    if not cfg.hub.repo_id:
        return None

    print("Không có checkpoint local, thử tải từ HF Hub...")
    import shutil
    from huggingface_hub import snapshot_download

    dest = os.path.join(cfg.train.output_dir, "checkpoint-from-hub")

    # (a) Thử subfolder "last-checkpoint/" trong branch main trước
    try:
        snap_dir = snapshot_download(
            repo_id=cfg.hub.repo_id,
            allow_patterns=["last-checkpoint/*"],
        )
        candidate = os.path.join(snap_dir, "last-checkpoint")
        if os.path.isdir(candidate) and os.listdir(candidate):
            shutil.copytree(candidate, dest, dirs_exist_ok=True)
            print(f"  ✓ Đã khôi phục checkpoint từ Hub (subfolder main/last-checkpoint) -> {dest}")
            return dest
    except Exception as e:
        print(f"  (thử subfolder main/last-checkpoint thất bại: {e})")

    # (b) Fallback: branch/revision riêng tên "last-checkpoint" (bản transformers cũ hơn)
    try:
        hub_ckpt_dir = snapshot_download(repo_id=cfg.hub.repo_id, revision="last-checkpoint")
        shutil.copytree(hub_ckpt_dir, dest, dirs_exist_ok=True)
        print(f"  ✓ Đã khôi phục checkpoint từ Hub (revision last-checkpoint) -> {dest}")
        return dest
    except Exception as e:
        print(f"  ⚠ Không có checkpoint trên Hub hoặc lỗi khi tải: {e}")
        return None


def main(cfg: Config = None):
    if cfg is None:
        cfg = Config()

    os.makedirs(cfg.train.output_dir, exist_ok=True)
    torch.manual_seed(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.train.device = device
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU : {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

    if cfg.hub.hf_token:
        from huggingface_hub import login
        login(token=cfg.hub.hf_token)
    elif cfg.hub.repo_id:
        print("Có repo_id nhưng chưa có hf_token — nhớ gọi huggingface_hub.login() thủ công trước khi chạy.")

    # ── Tokenizer ──────────────────────────────────────────────────────────
    tokenizer = load_tokenizer(cfg)
    cfg.model.vocab_size = max(cfg.model.vocab_size, len(tokenizer))
    print(f"Tokenizer vocab size (thực tế): {len(tokenizer)} | model vocab_size (đã pad): {cfg.model.vocab_size}")

    if cfg.hub.repo_id:
        tokenizer.push_to_hub(cfg.hub.repo_id, private=True)

    # ── Datasets ───────────────────────────────────────────────────────────
    train_ds, val_ds = build_datasets(cfg, tokenizer)
    print(f"Train docs: {len(train_ds):,} | Val docs: {len(val_ds):,}")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # ── Model ──────────────────────────────────────────────────────────────
    model = build_model(cfg.model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {n_params/1e6:.1f}M")

    training_args = build_training_args(cfg)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
    )

    # ── Resume — ưu tiên local, fallback sang HF Hub nếu output_dir bị mất
    # (vd session Colab ngắt, output_dir không nằm trên storage bền vững) ──
    resume_ckpt = cfg.train.resume_from_checkpoint
    if resume_ckpt is None:
        resume_ckpt = resume_checkpoint_from_hub_if_needed(cfg)

    if resume_ckpt:
        print(f"Resume từ checkpoint: {resume_ckpt}")
    else:
        print("Không có checkpoint, train từ đầu.")

    trainer.train(resume_from_checkpoint=resume_ckpt)

    print("✓ Train xong.")
    return trainer


if __name__ == "__main__":
    main()