"""
app/llama/dataset.py — Streaming dataset chuẩn HF, dùng thẳng cho transformers.Trainer
==========================================================================================
Thay toàn bộ cơ chế "chunk thủ công" (_BaseChunkedLoader/TokenChunkDataset/
SequentialDocumentDataset ở bản cũ) bằng pipeline streaming chuẩn của thư viện
`datasets`:

    raw (IterableDataset, streaming=True)
      └─▶ .map(tokenize_function, batched=True)      # tokenize, bỏ cột "text"
            └─▶ .map(group_texts, batched=True)       # PACKING: nối các doc
                                                        # liên tiếp, cắt thành
                                                        # block cố định block_size

KHÔNG bao giờ load hết file vào RAM — kể cả parquet local >1GB — vì
`load_dataset(..., streaming=True)` đọc theo kiểu iterator, và .map() ở trên
cũng chạy lazy theo từng batch nhỏ (map_batch_size), giải phóng ngay sau khi
xử lý xong. Đây chính là pattern RAM-safe tương đương "chunk thủ công" bản cũ,
nhưng để `datasets`/`Trainer` tự lo, không cần tự quản lý vòng lặp chunk nữa.

Vì sao PACKING (group_texts) thay vì pad-per-batch như DataCollator thường làm:
    - Pretrain trên document ngắn nếu chỉ pad riêng lẻ sẽ lãng phí rất nhiều
      compute vào token pad. group_texts nối nhiều doc lại (ngăn cách bởi
      eos_token) rồi cắt thành khối đúng block_size → gần như không có pad,
      tận dụng tối đa context window (giữ đúng tinh thần TokenChunkDataset cũ).
    - Nhãn (labels) KHÔNG tự shift tay ở đây nữa — để DataCollatorForLanguageModeling
      (mlm=False) gán labels = input_ids, và LlamaForCausalLM.forward() tự
      shift nội bộ khi nhận labels=... Đây là convention chuẩn của HF Trainer/
      run_clm.py, khác hẳn bản cũ (phải tự shift để né double-shift vì trainer
      tự viết không truyền labels vào forward).

Train/val split khi streaming: KHÔNG thể chia theo tỉ lệ % vì không biết
trước tổng số mẫu. Dùng .take(n)/.skip(n): n mẫu ĐẦU TIÊN của stream làm val,
phần còn lại làm train.
"""

import glob
import random
from typing import Callable, Optional

from datasets import load_dataset, interleave_datasets, IterableDataset


# ══════════════════════════════════════════════════════════════════════════
# Raw stream theo nguồn — giữ nguyên 4 nguồn cũ (wikipedia/vtsnlp/parquet/mix)
# ══════════════════════════════════════════════════════════════════════════

def _load_raw_stream(cfg, text_transform: Optional[Callable[[str], str]] = None) -> IterableDataset:
    source = cfg.data.source

    if source == "wikipedia":
        ds = load_dataset(
            cfg.data.dataset_name, cfg.data.dataset_subset,
            split="train", streaming=True,
        )
        text_col = "text"

    elif source == "vtsnlp":
        ds = load_dataset(
            "VTSNLP/vietnamese_curated_dataset", split="train", streaming=True,
        )
        if cfg.data.vtsnlp_domains:
            domains = set(cfg.data.vtsnlp_domains)
            ds = ds.filter(lambda ex: ex.get("domain") in domains)
        text_col = "text"

    elif source == "parquet":
        if not cfg.data.parquet_path:
            raise ValueError("cfg.data.source='parquet' nhưng cfg.data.parquet_path chưa được đặt.")
        ds = load_dataset(
            "parquet", data_files={"train": cfg.data.parquet_path},
            split="train", streaming=True,
        )
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

    # Chuẩn hoá tên cột về "text" + lọc rỗng/quá ngắn — làm ngay ở tầng raw để
    # tokenize_function bên dưới không phải quan tâm tên cột gốc theo nguồn.
    min_len = cfg.data.min_text_len

    def _normalize(ex):
        t = ex.get(text_col)
        if not isinstance(t, str):
            return {"text": ""}
        t = t.strip()
        if text_transform:
            t = text_transform(t)
        return {"text": t}

    ds = ds.map(_normalize, remove_columns=[c for c in ds.column_names or [] if c != text_col])
    ds = ds.filter(lambda ex: len(ex["text"]) >= min_len)
    return ds


# ══════════════════════════════════════════════════════════════════════════
# Tokenize + Pack (group_texts) — chạy lazy theo batch, RAM-safe
# ══════════════════════════════════════════════════════════════════════════

def _make_tokenize_fn(tokenizer):
    def tokenize_function(examples):
        # CHỈ trả "input_ids" — nếu để tokenizer tự trả thêm "attention_mask"
        # thì cột đó sẽ bị LỆCH ĐỘ DÀI so với "input_ids" sau bước group_texts
        # (group_texts đổi hẳn số dòng), khiến datasets báo lỗi khi ghép batch.
        enc = tokenizer(examples["text"], add_special_tokens=False, return_attention_mask=False)
        return {"input_ids": enc["input_ids"]}
    return tokenize_function


def _make_group_texts_fn(block_size: int, eos_token_id: Optional[int]):
    # Dùng list closure để lưu vết các token còn dư từ batch trước truyền sang batch sau
    remainder = []

    def group_texts(examples):
        nonlocal remainder
        ids_lists = examples["input_ids"]
        
        # Nối phần dư cũ với toàn bộ dữ liệu của batch mới
        concatenated = remainder + [i for ids in ids_lists for i in ids + ([eos_token_id] if eos_token_id is not None else [])]
        
        total_len = (len(concatenated) // block_size) * block_size
        
        if total_len > 0:
            result = [concatenated[i : i + block_size] for i in range(0, total_len, block_size)]
            # Giữ lại phần thừa cuối cùng (< block_size) cho lần gọi tiếp theo
            remainder = concatenated[total_len:]
            return {"input_ids": result}
        else:
            # Nếu vẫn chưa đủ 2048 tokens, tích lũy tiếp và trả về danh sách trống tạm thời
            remainder = concatenated
            return {"input_ids": []}

    return group_texts


def build_train_eval_datasets(cfg, tokenizer, text_transform: Optional[Callable[[str], str]] = None):
    """Entry point chính — trả về (train_dataset, eval_dataset) dạng
    datasets.IterableDataset, dùng thẳng cho transformers.Trainer.

    text_transform: convert_legacy_price_tokens khi cfg.data.source in
    ("parquet", "mix") — truyền từ train.py giống hệt convention cũ.
    """
    raw = _load_raw_stream(cfg, text_transform=text_transform)

    n_val = cfg.data.n_val_samples
    val_raw   = raw.take(n_val)
    train_raw = raw.skip(n_val)

    tokenize_fn = _make_tokenize_fn(tokenizer)
    group_fn    = _make_group_texts_fn(cfg.data.block_size, tokenizer.eos_token_id)
    batch_size  = cfg.data.map_batch_size

    def _pipeline(ds):
        ds = ds.map(tokenize_fn, batched=True, batch_size=batch_size, remove_columns=["text"])
        ds = ds.map(group_fn, batched=True, batch_size=batch_size)
        return ds

    train_ds = _pipeline(train_raw)
    if cfg.data.shuffle_buffer_size:
        train_ds = train_ds.shuffle(seed=cfg.seed, buffer_size=cfg.data.shuffle_buffer_size)

    eval_ds = _pipeline(val_raw)

    return train_ds, eval_ds
