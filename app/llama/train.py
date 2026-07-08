"""
Pretrain LLaMA tiếng Việt — pipeline nhiều-shard cho ~14B token trên Colab.

Thiết kế:
- Model nhỏ (SmolLM-135M style) + sdpa attention + gradient_checkpointing.
- Val set cố định, xử lý 1 lần.
- Optimizer + scheduler (cosine) tạo 1 lần duy nhất cho TOÀN BỘ quá trình,
  dùng chung xuyên suốt mọi shard để LR không bị gãy/reset.
- Cache dataset theo từng shard, tự xoá cache shard trước để tiết kiệm dung lượng Drive.
- Resume phân biệt 2 case:
    (a) Ngắt giữa chừng 1 shard chưa xong -> dùng resume_from_checkpoint bình thường.
    (b) Bắt đầu 1 shard MỚI (shard trước đã xong) -> KHÔNG dùng resume_from_checkpoint
        (Trainer sẽ hiểu nhầm global_step đã vượt max_steps của shard mới và dừng ngay).
        Thay vào đó tự load state_dict optimizer/scheduler/model rồi gọi train() thường.
- Push checkpoint đầy đủ + tokenizer lên Hugging Face Hub làm lớp backup thứ 2.

Chạy trong Colab, từng cell tương ứng các hàm dưới đây, hoặc chạy thẳng cả file
sau khi đã mount Drive và đăng nhập Hugging Face.
"""

import glob
import json
import os
import shutil

import torch
from datasets import load_dataset, load_from_disk
from huggingface_hub import login, snapshot_download
from transformers import (
    DataCollatorForLanguageModeling,
    LlamaConfig,
    LlamaForCausalLM,
    Trainer,
    TrainingArguments,
    get_cosine_schedule_with_warmup,
)
from transformers.trainer_utils import get_last_checkpoint


# =====================================================================================
# 0. CẤU HÌNH CHUNG — chỉnh các giá trị này theo môi trường thực tế trước khi chạy
# =====================================================================================

DRIVE_ROOT = "/content/drive/MyDrive/llama_project"
TOKENIZER_DIR = "/content/custom_tokenizer_llama"
TRAIN_SHARD_DIR = f"{DRIVE_ROOT}/train_shards"          # nhiều file .parquet gốc, đã tách sẵn
VAL_PARQUET_GLOB = f"{DRIVE_ROOT}/val/*.parquet"
CACHE_DIR = f"{DRIVE_ROOT}/train_shards_cache"
VAL_CACHE_DIR = f"{DRIVE_ROOT}/lm_val_cache"
OUTPUT_DIR = f"{DRIVE_ROOT}/checkpoints"
STATE_PATH = f"{OUTPUT_DIR}/shard_state.json"

HUB_MODEL_ID = "your-username/llama-vi-pretrain"   # đổi theo repo thật
HUB_TOKEN = None                                    # để None và dùng login() thủ công, tránh hardcode

VOCAB_SIZE = None          # sẽ lấy từ tokenizer sau khi load
MAX_SEQ_LENGTH = 1024
BLOCK_SIZE = MAX_SEQ_LENGTH

PER_DEVICE_TRAIN_BATCH = 16
GRAD_ACCUM_STEPS = 8       # effective batch ~= 128
EFFECTIVE_BATCH = PER_DEVICE_TRAIN_BATCH * GRAD_ACCUM_STEPS

TOTAL_TOKENS_ESTIMATE = 14_000_000_000     # 14 file x ~1B token
TOTAL_STEPS = TOTAL_TOKENS_ESTIMATE // (BLOCK_SIZE * EFFECTIVE_BATCH)
WARMUP_STEPS = int(0.03 * TOTAL_STEPS)

SAVE_STEPS = 50
EVAL_STEPS = 100
LOGGING_STEPS = 10


def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(VAL_CACHE_DIR, exist_ok=True)


# =====================================================================================
# 1. TOKENIZER
# =====================================================================================

def load_tokenizer():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def tokenize_function(examples, tokenizer):
    result = tokenizer(examples["text"], truncation=False)
    # Thêm eos_token vào cuối mỗi document để đánh dấu ranh giới giữa các doc
    # khi bị nối liền trong group_texts — tránh model học nhầm 2 đoạn không
    # liên quan là 1 mạch văn liên tục.
    result["input_ids"] = [ids + [tokenizer.eos_token_id] for ids in result["input_ids"]]
    return result


def group_texts(examples, block_size=BLOCK_SIZE):
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated["input_ids"])
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    result = {
        "input_ids": [
            concatenated["input_ids"][i : i + block_size]
            for i in range(0, total_length, block_size)
        ]
    }
    # Không lưu attention_mask/labels — DataCollatorForLanguageModeling(mlm=False)
    # tự sinh labels từ input_ids, giảm ~2/3 dung lượng cache.
    return result


# =====================================================================================
# 2. VAL SET — xử lý 1 lần, cố định suốt toàn bộ quá trình
# =====================================================================================

def get_or_build_val_dataset(tokenizer):
    if os.path.exists(VAL_CACHE_DIR) and os.listdir(VAL_CACHE_DIR):
        print("Val set: đã có cache, load lại.")
        return load_from_disk(VAL_CACHE_DIR)

    print("Val set: chưa có cache, xử lý mới...")
    raw_val = load_dataset("parquet", data_files=VAL_PARQUET_GLOB)["train"]
    tok_val = raw_val.map(
        lambda ex: tokenize_function(ex, tokenizer),
        batched=True,
        num_proc=os.cpu_count(),
        remove_columns=raw_val.column_names,
        desc="Tokenizing val set",
    )
    lm_val = tok_val.map(
        group_texts,
        batched=True,
        num_proc=os.cpu_count(),
        desc="Grouping val set",
    )
    lm_val.save_to_disk(VAL_CACHE_DIR)
    return lm_val


# =====================================================================================
# 3. MODEL
# =====================================================================================

def build_model(vocab_size):
    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=576,
        intermediate_size=1536,
        num_hidden_layers=30,
        num_attention_heads=9,
        num_key_value_heads=3,
        max_position_embeddings=MAX_SEQ_LENGTH,
        pad_token_id=3,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=True,
    )
    model = LlamaForCausalLM._from_config(config, attn_implementation="sdpa")
    return model


# =====================================================================================
# 4. STATE (shard đang xử lý) — lưu/khôi phục tiến độ giữa các session Colab
# =====================================================================================

def load_state():
    if os.path.exists(STATE_PATH):
        return json.load(open(STATE_PATH))
    return {"shard_index": 0}


def save_state(state):
    json.dump(state, open(STATE_PATH, "w"))


# =====================================================================================
# 5. TRAINING ARGS (dùng chung cho mọi shard)
# =====================================================================================

def build_training_args():
    return TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        per_device_eval_batch_size=PER_DEVICE_TRAIN_BATCH,
        fp16=True,
        logging_steps=LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        num_train_epochs=1,          # chỉ 1 lượt qua SHARD hiện tại rồi dừng
        report_to="none",
        push_to_hub=True,
        hub_model_id=HUB_MODEL_ID,
        hub_private_repo=True,
        hub_strategy="checkpoint",   # push full checkpoint (model+optimizer+scheduler+rng)
    )
    # Lưu ý: KHÔNG set lr_scheduler_type/warmup_steps ở đây — vì optimizer/scheduler
    # được tạo thủ công và truyền trực tiếp vào Trainer (mục 6), các tham số LR
    # trong TrainingArguments sẽ bị bỏ qua hoàn toàn trong trường hợp đó.


# =====================================================================================
# 6. VÒNG LẶP CHÍNH — xử lý tuần tự từng shard, resume đúng theo 2 trường hợp
# =====================================================================================

def main():
    ensure_dirs()

    # --- đăng nhập Hugging Face (dùng notebook_login() nếu chạy tương tác) ---
    if HUB_TOKEN:
        login(token=HUB_TOKEN)
    else:
        print("Nhớ gọi huggingface_hub.login() hoặc notebook_login() trước khi chạy main().")

    tokenizer = load_tokenizer()
    global VOCAB_SIZE
    VOCAB_SIZE = len(tokenizer)

    # Push tokenizer 1 lần duy nhất — cố định xuyên suốt, không cần lặp lại theo checkpoint.
    tokenizer.push_to_hub(HUB_MODEL_ID, private=True)

    lm_val = get_or_build_val_dataset(tokenizer)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    model = build_model(VOCAB_SIZE)

    # Optimizer + scheduler: tạo 1 LẦN DUY NHẤT cho toàn bộ hành trình 14B token.
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=TOTAL_STEPS
    )

    print(f"Tổng step ước tính: {TOTAL_STEPS} | Warmup: {WARMUP_STEPS}")

    shard_files = sorted(glob.glob(f"{TRAIN_SHARD_DIR}/*.parquet"))
    print(f"Tổng số shard: {len(shard_files)}")

    state = load_state()
    training_args = build_training_args()

    for i in range(state["shard_index"], len(shard_files)):
        shard_path = shard_files[i]
        shard_cache = f"{CACHE_DIR}/shard_{i}"

        # --- load hoặc tokenize shard hiện tại ---
        if os.path.exists(shard_cache) and os.listdir(shard_cache):
            print(f"[Shard {i}] Đã có cache, load lại.")
            lm_shard = load_from_disk(shard_cache)
        else:
            print(f"[Shard {i}] Tokenize + group mới từ {shard_path}")
            raw = load_dataset("parquet", data_files=shard_path)["train"]
            tok = raw.map(
                lambda ex: tokenize_function(ex, tokenizer),
                batched=True,
                num_proc=os.cpu_count(),
                remove_columns=raw.column_names,
                desc=f"Tokenizing shard {i}",
            )
            lm_shard = tok.map(
                group_texts,
                batched=True,
                num_proc=os.cpu_count(),
                desc=f"Grouping shard {i}",
            )
            lm_shard.save_to_disk(shard_cache)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=lm_shard,
            eval_dataset=lm_val,
            data_collator=data_collator,
            optimizers=(optimizer, scheduler),
        )

        # --- xác định đây là "resume giữa chừng shard này" hay "bắt đầu shard mới" ---
        local_ckpt = get_last_checkpoint(OUTPUT_DIR)
        is_fresh_shard_start = state.get("last_completed_shard_ckpt_step") == (
            local_ckpt and int(local_ckpt.split("-")[-1])
        )
        # is_fresh_shard_start = True nghĩa là checkpoint tìm được chính là checkpoint
        # đã lưu NGAY SAU KHI shard trước hoàn thành — tức shard hiện tại (i) là mới,
        # chưa có step nào của riêng nó -> không resume, để Trainer tự train từ đầu shard.
        #
        # Nếu checkpoint mới hơn (do đã train dở shard i rồi bị ngắt) -> resume bình thường.

        if local_ckpt is None:
            print(f"[Shard {i}] Không có checkpoint nào, train từ đầu.")
            trainer.train()
        elif is_fresh_shard_start:
            print(f"[Shard {i}] Checkpoint thuộc về shard trước, bắt đầu shard mới (không resume).")
            trainer.train()
        else:
            print(f"[Shard {i}] Resume giữa chừng từ {local_ckpt}")
            trainer.train(resume_from_checkpoint=local_ckpt)

        # --- cập nhật state SAU KHI shard train xong hoàn toàn ---
        final_ckpt = get_last_checkpoint(OUTPUT_DIR)
        state["shard_index"] = i + 1
        state["last_completed_shard_ckpt_step"] = (
            int(final_ckpt.split("-")[-1]) if final_ckpt else None
        )
        save_state(state)
        print(f"✅ Hoàn thành shard {i}.")

        # --- xoá cache shard TRƯỚC (không xoá shard vừa train, chỉ xoá cái trước đó) ---
        if i > 0:
            prev_cache = f"{CACHE_DIR}/shard_{i - 1}"
            if os.path.exists(prev_cache):
                shutil.rmtree(prev_cache, ignore_errors=True)
                print(f"🗑️  Đã xoá cache shard {i - 1}.")

    print("🎉 Đã train xong toàn bộ shard.")


# =====================================================================================
# 7. HÀM RESUME THỦ CÔNG TỪ HUB (fallback nếu Drive bị mất/hỏng)
# =====================================================================================

def resume_checkpoint_from_hub_if_needed(local_output_dir=OUTPUT_DIR, hub_model_id=HUB_MODEL_ID):
    """Gọi hàm này TRƯỚC khi chạy main() nếu nghi ngờ Drive đã mất checkpoint cục bộ."""
    local_ckpt = get_last_checkpoint(local_output_dir)
    if local_ckpt:
        print(f"Đã có checkpoint cục bộ: {local_ckpt}, không cần pull từ Hub.")
        return local_ckpt

    print("Không tìm thấy checkpoint cục bộ, thử pull từ Hub...")
    try:
        hub_ckpt_dir = snapshot_download(repo_id=hub_model_id, revision="last-checkpoint")
        # copy về đúng vị trí OUTPUT_DIR để get_last_checkpoint() trong main() nhận diện được
        dest = os.path.join(local_output_dir, "checkpoint-from-hub")
        shutil.copytree(hub_ckpt_dir, dest, dirs_exist_ok=True)
        print(f"Đã khôi phục checkpoint từ Hub về {dest}")
        return dest
    except Exception as e:
        print(f"Không có checkpoint trên Hub hoặc lỗi khi tải: {e}")
        return None


if __name__ == "__main__":
    main()