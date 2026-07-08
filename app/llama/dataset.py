"""
dataset.py — Data pipeline (parquet local only) cho app/llama
================================================================================
Tách từ train.py nháp: tokenize_function(), group_texts(), get_or_build_val_dataset(),
và phần tokenize+group từng shard đang lặp lại y hệt trong vòng lặp main().

PHẠM VI: CHỈ đọc parquet local (train_shard_dir/*.parquet + val_parquet_glob).
Không hỗ trợ streaming từ HF Hub (wikipedia/vtsnlp) — data đã có đủ ở local
nên bỏ hẳn nhánh đó cho gọn, tránh design cho use-case chưa cần tới.

Khác biệt quan trọng với bench_pipeline.py (đã dùng để debug tốc độ trước đó):
    - Ở ĐÂY dùng `load_dataset(..., streaming=False)` (mặc định), KHÔNG streaming
      — mỗi shard parquet được load ĐẦY ĐỦ vào bộ nhớ/Arrow cache rồi map với
      `num_proc` song song. Hợp lý vì mỗi shard đã được cắt vừa đủ nhỏ (thiết
      kế trong readme.md mục 5.1), không cần streaming để tiết kiệm RAM.
    - group_texts nối các document trong CÙNG 1 shard lại với nhau rồi cắt
      block_size — giữ nguyên logic gốc, không đổi.
"""

import glob
import os
from typing import List

from datasets import Dataset, load_dataset, load_from_disk

from app.llama.config import Config


# ══════════════════════════════════════════════════════════════════════════
# Tokenize + pack — giữ nguyên logic từ nháp
# ══════════════════════════════════════════════════════════════════════════

def tokenize_function(examples, tokenizer):
    """Tokenize, KHÔNG truncate (group_texts sẽ cắt sau khi nối). Thêm eos_token
    vào cuối mỗi document để đánh dấu ranh giới giữa các doc khi bị nối liền —
    tránh model học nhầm 2 đoạn không liên quan là 1 mạch văn liên tục."""
    result = tokenizer(examples["text"], truncation=False)
    result["input_ids"] = [ids + [tokenizer.eos_token_id] for ids in result["input_ids"]]
    return result


def group_texts(examples, block_size: int):
    """Nối toàn bộ input_ids trong batch lại, cắt thành các đoạn block_size cố
    định. Phần dư cuối (< block_size) bị bỏ — chấp nhận được vì mất rất ít so
    với tổng dữ liệu (readme.md mục 3.2). KHÔNG lưu attention_mask/labels —
    DataCollatorForLanguageModeling(mlm=False) tự sinh labels từ input_ids."""
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated["input_ids"])
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    result = {
        "input_ids": [
            concatenated["input_ids"][i: i + block_size]
            for i in range(0, total_length, block_size)
        ]
    }
    return result


def _tokenize_and_group(raw: Dataset, tokenizer, block_size: int, num_proc: int, desc_prefix: str) -> Dataset:
    """Dùng chung cho cả shard train lẫn val set — tránh lặp lại logic 2 lần
    như bản nháp (get_or_build_val_dataset và vòng lặp shard trong main() gọi
    y hệt 2 bước map này, tách hàm chung để sửa 1 chỗ áp dụng cả 2 nơi)."""
    tok = raw.map(
        lambda ex: tokenize_function(ex, tokenizer),
        batched=True,
        num_proc=num_proc,
        remove_columns=raw.column_names,
        desc=f"Tokenizing {desc_prefix}",
    )
    return tok.map(
        lambda ex: group_texts(ex, block_size),
        batched=True,
        num_proc=num_proc,
        desc=f"Grouping {desc_prefix}",
    )


# ══════════════════════════════════════════════════════════════════════════
# Danh sách shard
# ══════════════════════════════════════════════════════════════════════════

def list_shard_files(cfg: Config) -> List[str]:
    """Danh sách file .parquet trong train_shard_dir, sắp xếp cố định — vòng
    lặp main() xử lý tuần tự theo đúng thứ tự này, PHẢI ổn định qua các lần
    chạy khác nhau để state["shard_index"] (state.py) còn ý nghĩa khi resume."""
    return sorted(glob.glob(f"{cfg.data.train_shard_dir}/*.parquet"))


# ══════════════════════════════════════════════════════════════════════════
# Val set — xử lý 1 lần, cache cố định suốt toàn bộ quá trình
# ══════════════════════════════════════════════════════════════════════════

def get_or_build_val_dataset(cfg: Config, tokenizer) -> Dataset:
    """Load val set đã cache nếu có, không thì tokenize+group mới rồi lưu cache.
    Val set CỐ ĐỊNH xuyên suốt toàn bộ hành trình nhiều-shard — không đổi theo
    shard đang train (readme.md mục 3.3)."""
    val_cache_dir = cfg.data.val_cache_dir

    if val_cache_dir and os.path.exists(val_cache_dir) and os.listdir(val_cache_dir):
        print("Val set: đã có cache, load lại.")
        return load_from_disk(val_cache_dir)

    print("Val set: chưa có cache, xử lý mới...")
    num_proc = cfg.data.map_num_proc or os.cpu_count()
    raw_val = load_dataset("parquet", data_files=cfg.data.val_parquet_glob)["train"]
    lm_val = _tokenize_and_group(raw_val, tokenizer, cfg.data.block_size, num_proc, "val set")

    if val_cache_dir:
        lm_val.save_to_disk(val_cache_dir)
    return lm_val


# ══════════════════════════════════════════════════════════════════════════
# 1 shard train — load cache nếu có, không thì tokenize+group mới
# ══════════════════════════════════════════════════════════════════════════

def get_or_build_shard_dataset(cfg: Config, tokenizer, shard_index: int, shard_path: str) -> Dataset:
    """Tokenize+group đúng 1 shard, cache theo shard_index vào cache_dir/shard_{i}.
    Caller (train.py, vòng lặp main) chịu trách nhiệm xoá cache shard TRƯỚC sau
    khi chuyển tiếp thành công (readme.md mục 5.3) — hàm này chỉ lo phần
    load-hoặc-build, không tự dọn dẹp."""
    shard_cache = f"{cfg.data.cache_dir}/shard_{shard_index}"

    if os.path.exists(shard_cache) and os.listdir(shard_cache):
        print(f"[Shard {shard_index}] Đã có cache, load lại.")
        return load_from_disk(shard_cache)

    print(f"[Shard {shard_index}] Tokenize + group mới từ {shard_path}")
    num_proc = cfg.data.map_num_proc or os.cpu_count()
    raw = load_dataset("parquet", data_files=shard_path)["train"]
    lm_shard = _tokenize_and_group(raw, tokenizer, cfg.data.block_size, num_proc, f"shard {shard_index}")

    lm_shard.save_to_disk(shard_cache)
    return lm_shard


def clear_previous_shard_cache(cfg: Config, shard_index: int) -> None:
    """Xoá cache của shard TRƯỚC shard_index (không xoá shard vừa train xong) —
    giữ nguyên nguyên tắc dọn dẹp trong readme.md mục 5.3, tách hàm riêng để
    main() không phải tự import shutil."""
    import shutil

    if shard_index <= 0:
        return
    prev_cache = f"{cfg.data.cache_dir}/shard_{shard_index - 1}"
    if os.path.exists(prev_cache):
        shutil.rmtree(prev_cache, ignore_errors=True)
        print(f"🗑️  Đã xoá cache shard {shard_index - 1}.")