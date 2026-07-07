"""
dataset_simple.py — Pipeline dữ liệu ĐƠN GIẢN, không pack/nối document
================================================================================
Bản rút gọn của app/llama/dataset.py: BỎ HẲN group_texts (nối nhiều document
lại rồi cắt thành block cố định). Mỗi document được tokenize + cắt ngắn
(truncate) riêng lẻ tại block_size, dùng luôn.

Đánh đổi so với bản pack (dataset.py gốc):
    - ĐƠN GIẢN, ít chỗ có thể sai (không còn `nonlocal remainder`, không còn
      rủi ro lệch cột khi số dòng mỗi batch thay đổi).
    - TỐN COMPUTE HƠN: document ngắn sẽ bị pad nhiều (DataCollatorForLanguageModeling
      tự pad), lãng phí phần lớn context window nếu văn bản ngắn. Chấp nhận
      được ở giai đoạn test pipeline / model nhỏ — quay lại pack sau khi mọi
      thứ khác đã chạy ổn.
    - Không cắt xén document dài thành nhiều mảnh — phần vượt block_size bị
      CẮT BỎ (mất), không giữ lại phần dư như group_texts.

Giữ NGUYÊN interface `build_train_eval_datasets(cfg, tokenizer)` — cùng cách
gọi với app/llama/dataset.py, nên có thể đổi qua lại bằng 1 dòng import mà
không cần sửa app/llama/train.py.

Field cfg.data cần dùng (giống DataConfig hiện có, KHÔNG cần field mới):
    source, dataset_name, dataset_subset, vtsnlp_domains,
    parquet_path, parquet_text_col, mix_sources, mix_stopping,
    min_text_len, n_val_samples, block_size, shuffle_buffer_size,
    map_batch_size

Test nhanh không cần Trainer:
    python dataset_simple.py --source wikipedia --n 3
"""

import glob
import random
from typing import Callable, Optional

from datasets import load_dataset, interleave_datasets, IterableDataset


# ══════════════════════════════════════════════════════════════════════════
# Raw stream theo nguồn — giữ 4 nguồn cũ (wikipedia/vtsnlp/parquet/mix)
# ══════════════════════════════════════════════════════════════════════════

def _load_raw_stream(cfg, text_transform: Optional[Callable[[str], str]] = None) -> IterableDataset:
    source = cfg.data.source

    if source == "wikipedia":
        ds = load_dataset(cfg.data.dataset_name, cfg.data.dataset_subset, split="train", streaming=True)
        text_col = "text"

    elif source == "vtsnlp":
        ds = load_dataset("VTSNLP/vietnamese_curated_dataset", split="train", streaming=True)
        if cfg.data.vtsnlp_domains:
            domains = set(cfg.data.vtsnlp_domains)
            ds = ds.filter(lambda ex: ex.get("domain") in domains)
        text_col = "text"

    elif source == "parquet":
        if not cfg.data.parquet_path:
            raise ValueError("cfg.data.source='parquet' nhưng cfg.data.parquet_path chưa được đặt.")
        ds = load_dataset("parquet", data_files={"train": cfg.data.parquet_path}, split="train", streaming=True)
        text_col = cfg.data.parquet_text_col

    elif source == "mix":
        sources = cfg.data.mix_sources
        if not sources:
            raise ValueError("cfg.data.mix_sources trống.")

        names = list(sources.keys())
        probs = [sources[n][1] for n in names]
        if abs(sum(probs) - 1.0) > 1e-3:
            raise ValueError(f"Tổng probabilities = {sum(probs):.4f}, phải = 1.0")

        print(f"  Mix sources ({len(names)}):")
        parts = []
        for name, prob in zip(names, probs):
            pattern = sources[name][0]
            files = sorted(glob.glob(pattern))
            if not files:
                raise FileNotFoundError(f"Không tìm thấy file: {pattern}")
            random.shuffle(files)
            print(f"    {name:<14} {prob*100:.0f}%  ({len(files)} files)")
            parts.append(load_dataset("parquet", data_files={"train": files}, split="train", streaming=True))

        ds = interleave_datasets(parts, probabilities=None, seed=cfg.seed, stopping_strategy=cfg.data.mix_stopping)
        text_col = cfg.data.parquet_text_col

    else:
        raise ValueError(f"cfg.data.source='{source}' không hợp lệ")

    # QUAN TRỌNG: strip TOÀN BỘ cột khác ngoài "text" NGAY TẠI ĐÂY, một lần
    # duy nhất. Đây là bước dễ bị quên nhất (đã gây bug IndexError khi test
    # nhanh với bench_pipeline.py) — mọi bước .map() sau này chỉ còn thấy
    # đúng 1 cột "text", không còn rủi ro lệch cột khi số dòng đổi.
    min_len = cfg.data.min_text_len

    def _normalize(ex):
        t = ex.get(text_col)
        if not isinstance(t, str):
            return {"text": ""}
        t = t.strip()
        if text_transform:
            t = text_transform(t)
        return {"text": t}

    other_cols = [c for c in (ds.column_names or []) if c != text_col]
    ds = ds.map(_normalize, remove_columns=other_cols or None)
    ds = ds.filter(lambda ex: len(ex["text"]) >= min_len)
    return ds


# ══════════════════════════════════════════════════════════════════════════
# Tokenize — KHÔNG pack, mỗi document 1 sample riêng, truncate tại block_size
# ══════════════════════════════════════════════════════════════════════════

def _make_tokenize_fn(tokenizer, block_size: int):
    def tokenize_function(examples):
        enc = tokenizer(
            examples["text"],
            add_special_tokens=False,
            truncation=True,
            max_length=block_size,
            return_attention_mask=False,
        )
        return {"input_ids": enc["input_ids"]}
    return tokenize_function


def build_train_eval_datasets(cfg, tokenizer, text_transform: Optional[Callable[[str], str]] = None):
    """Entry point chính — CÙNG SIGNATURE với app/llama/dataset.py, trả về
    (train_dataset, eval_dataset) dạng datasets.IterableDataset, dùng thẳng
    cho transformers.Trainer. KHÔNG pack — mỗi phần tử là 1 document đã
    tokenize + truncate, độ dài <= block_size (có thể ngắn hơn nhiều)."""
    raw = _load_raw_stream(cfg, text_transform=text_transform)

    n_val = cfg.data.n_val_samples
    val_raw = raw.take(n_val)
    train_raw = raw.skip(n_val)

    tokenize_fn = _make_tokenize_fn(tokenizer, cfg.data.block_size)
    batch_size = cfg.data.map_batch_size

    def _pipeline(ds):
        return ds.map(tokenize_fn, batched=True, batch_size=batch_size, remove_columns=["text"])

    train_ds = _pipeline(train_raw)
    if cfg.data.shuffle_buffer_size:
        # Lưu ý: buffer_size ở đây tính theo SỐ DOCUMENT (không phải số block
        # đã pack như bản group_texts) — buffer-fill sẽ nhanh hơn nhiều vì
        # không cần tích lũy hàng triệu token trước khi yield.
        train_ds = train_ds.shuffle(seed=cfg.seed, buffer_size=cfg.data.shuffle_buffer_size)

    eval_ds = _pipeline(val_raw)

    return train_ds, eval_ds


# ══════════════════════════════════════════════════════════════════════════
# Test nhanh không cần Trainer
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    from dataclasses import dataclass, field
    from typing import Optional as _Optional
    from transformers import AutoTokenizer

    p = argparse.ArgumentParser()
    p.add_argument("--source", type=str, default="wikipedia",
                   choices=["wikipedia", "vtsnlp", "parquet", "mix"])
    p.add_argument("--dataset-name", type=str, default="wikimedia/wikipedia")
    p.add_argument("--dataset-subset", type=str, default="20231101.vi")
    p.add_argument("--block-size", type=int, default=512)
    p.add_argument("--tokenizer", type=str, default="gpt2")
    p.add_argument("--n", type=int, default=3, help="Số sample muốn in thử")
    args = p.parse_args()

    @dataclass
    class _DataCfg:
        source: str = args.source
        dataset_name: str = args.dataset_name
        dataset_subset: str = args.dataset_subset
        vtsnlp_domains: _Optional[list] = None
        parquet_path: str = ""
        parquet_text_col: str = "text"
        mix_sources: dict = field(default_factory=dict)
        mix_stopping: str = "first_exhausted"
        min_text_len: int = 50
        n_val_samples: int = 5
        block_size: int = args.block_size
        shuffle_buffer_size: int = 0
        map_batch_size: int = 100

    @dataclass
    class _Cfg:
        data: _DataCfg = field(default_factory=_DataCfg)
        seed: int = 42

    cfg = _Cfg()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds, eval_ds = build_train_eval_datasets(cfg, tokenizer)

    print(f"\n── In thử {args.n} sample từ train_ds ──")
    for i, ex in enumerate(train_ds):
        if i >= args.n:
            break
        ids = ex["input_ids"]
        print(f"  [{i}] len={len(ids)}  decoded[:200]={tokenizer.decode(ids[:60])!r}...")