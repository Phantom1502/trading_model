"""
dataset.py — Incremental loading cho dữ liệu tiếng Việt
============================================================
RAM hạn chế nên không load toàn bộ dataset một lần.
Thay vào đó: load từng chunk N sample, train xong, giải phóng, load chunk tiếp.

Hỗ trợ 3 nguồn dữ liệu:
    ChunkedWikiLoader    — wikimedia/wikipedia (bản gốc, field "text")
    ChunkedVTSNLPLoader  — VTSNLP/vietnamese_curated_dataset (đã curate,
                           field "text" + "domain", chất lượng cao hơn,
                           12.2M rows, có thể lọc theo domain cụ thể)
    ChunkedParquetLoader — file .parquet local (sách, corpus nội bộ, ...)
                           field "text" + metadata tuỳ chọn (title, author, ...)
                           hỗ trợ lọc theo metadata qua filter_fn

Cả ba đều là generator — mỗi lần `next()` trả về (train_loader, val_loader)
đã tokenize sẵn, sẵn sàng đưa vào train_one_chunk().
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset


class TokenChunkDataset(Dataset):
    """
    Dataset đơn giản: nhận list token_ids đã nối sẵn, cắt thành các
    đoạn cố định length `seg_len + 1` (input + 1 token để làm label).

    Mỗi document được giữ riêng biệt (không nối lẫn giữa các document)
    để document boundary rõ ràng — quan trọng cho việc reset M đúng chỗ.
    """

    def __init__(self, documents: list[list[int]], seg_len: int):
        self.samples = []

        for doc in documents:
            if len(doc) < 2:
                continue
            n_full = max(1, len(doc) // (seg_len + 1))
            for i in range(n_full):
                start = i * (seg_len + 1)
                end   = start + seg_len + 1
                chunk = doc[start:end]
                if len(chunk) < 2:
                    continue
                self.samples.append({
                    "ids"        : chunk,
                    "is_doc_start": (i == 0),
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        ids  = torch.tensor(item["ids"], dtype=torch.long)
        return {
            "input_ids"   : ids[:-1],
            "labels"      : ids[1:],
            "is_doc_start": item["is_doc_start"],
        }


def collate_fn(batch, pad_id: int = 0):
    """Pad các sample trong batch về cùng độ dài."""
    max_len = max(item["input_ids"].size(0) for item in batch)

    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels    = torch.full((len(batch), max_len), -100,   dtype=torch.long)
    is_doc_start = []

    for i, item in enumerate(batch):
        L = item["input_ids"].size(0)
        input_ids[i, :L] = item["input_ids"]
        labels[i, :L]    = item["labels"]
        is_doc_start.append(item["is_doc_start"])

    return {
        "input_ids"   : input_ids,
        "labels"      : labels,
        "is_doc_start": torch.tensor(is_doc_start, dtype=torch.bool),
    }


# ══════════════════════════════════════════════════════════════════════════
# Base class — logic chung cho mọi nguồn streaming dataset
# ══════════════════════════════════════════════════════════════════════════

class _BaseChunkedLoader:
    """
    Logic chung: load N sample từ streaming dataset, tokenize, chia
    train/val, trả về DataLoader. Subclass chỉ cần override
    `_load_dataset()` và `_extract_text(sample)`.
    """

    def __init__(self, cfg, tokenizer, start_chunk: int = 0):
        self.cfg           = cfg
        self.tokenizer     = tokenizer
        self.chunk_size    = cfg.data.chunk_size
        self.seg_len       = cfg.data.seg_len
        self.min_text_len  = cfg.data.min_text_len
        self.val_ratio     = cfg.data.val_ratio
        self.batch_size    = cfg.train.batch_size
        self.total_chunks  = cfg.train.total_chunks
        self.start_chunk   = start_chunk

        self.raw_stream = self._load_dataset()

        if start_chunk > 0:
            n_skip = start_chunk * self.chunk_size
            print(f"Resume: skip {n_skip:,} sample đầu (tương ứng {start_chunk} chunk đã train)...")
            self.raw_stream = self.raw_stream.skip(n_skip)

        self.stream_iter = iter(self.raw_stream)
        self.exhausted   = False
        self.chunk_count = start_chunk

    def _load_dataset(self):
        raise NotImplementedError

    def _extract_text(self, sample: dict) -> str | None:
        raise NotImplementedError

    def _load_one_chunk(self) -> list[list[int]] | None:
        texts = []

        while len(texts) < self.chunk_size:
            try:
                sample = next(self.stream_iter)
            except StopIteration:
                self.exhausted = True
                break

            text = self._extract_text(sample)
            if text is None or len(text) < self.min_text_len:
                continue

            texts.append(text)

        if not texts:
            return None

        token_lists = self.tokenizer.encode_batch(texts, add_special_tokens=False)
        documents = [ids for ids in token_lists if len(ids) >= 2]
        return documents if documents else None

    def __iter__(self):
        return self

    def __next__(self):
        if self.exhausted:
            raise StopIteration
        if self.total_chunks > 0 and self.chunk_count >= self.total_chunks:
            raise StopIteration

        documents = self._load_one_chunk()
        if documents is None:
            raise StopIteration

        self.chunk_count += 1
        current_chunk_idx = self.chunk_count

        split_idx  = int(len(documents) * (1 - self.val_ratio))
        train_docs = documents[:split_idx]
        val_docs   = documents[split_idx:] if split_idx < len(documents) else documents[-5:]

        train_ds = TokenChunkDataset(train_docs, self.seg_len)
        val_ds   = TokenChunkDataset(val_docs,   self.seg_len)

        pad_id = self.tokenizer.pad_id

        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True,
            collate_fn=lambda b: collate_fn(b, pad_id),
            num_workers=0,
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.batch_size, shuffle=False,
            collate_fn=lambda b: collate_fn(b, pad_id),
            num_workers=0,
        )

        print(
            f"[Chunk {current_chunk_idx}] "
            f"docs: {len(documents)} | "
            f"train samples: {len(train_ds)} | "
            f"val samples: {len(val_ds)}"
        )

        return train_loader, val_loader


# ══════════════════════════════════════════════════════════════════════════
# ChunkedWikiLoader — wikimedia/wikipedia
# ══════════════════════════════════════════════════════════════════════════

class ChunkedWikiLoader(_BaseChunkedLoader):
    """
    Load Wikipedia tiếng Việt theo từng chunk N article.

    Usage:
        loader = ChunkedWikiLoader(cfg, tokenizer)
        for train_loader, val_loader in loader:
            train_one_chunk(model, train_loader, val_loader)

    Resume từ chunk cụ thể:
        loader = ChunkedWikiLoader(cfg, tokenizer, start_chunk=20)
    """

    def _load_dataset(self):
        return load_dataset(
            self.cfg.data.dataset_name,
            self.cfg.data.dataset_subset,
            split="train",
            streaming=True,
        )

    def _extract_text(self, sample: dict) -> str | None:
        text = sample.get("text", "").strip()
        return text if text else None


# ══════════════════════════════════════════════════════════════════════════
# ChunkedVTSNLPLoader — VTSNLP/vietnamese_curated_dataset
# ══════════════════════════════════════════════════════════════════════════

class ChunkedVTSNLPLoader(_BaseChunkedLoader):
    """
    Load VTSNLP/vietnamese_curated_dataset — dataset tiếng Việt đã curate,
    12.2M rows, field: text, id, domain (25 domain).

    Usage — dùng toàn bộ domain:
        loader = ChunkedVTSNLPLoader(cfg, tokenizer)

    Usage — chỉ lấy một số domain cụ thể:
        loader = ChunkedVTSNLPLoader(cfg, tokenizer, domains=["Science", "Books_and_Literature"])

    Resume từ chunk cụ thể:
        loader = ChunkedVTSNLPLoader(cfg, tokenizer, start_chunk=20)
    """

    HF_DATASET_NAME = "VTSNLP/vietnamese_curated_dataset"

    def __init__(self, cfg, tokenizer, start_chunk: int = 0, domains: list[str] = None):
        self.domains = set(domains) if domains else None
        super().__init__(cfg, tokenizer, start_chunk=start_chunk)

    def _load_dataset(self):
        return load_dataset(
            self.HF_DATASET_NAME,
            split="train",
            streaming=True,
        )

    def _extract_text(self, sample: dict) -> str | None:
        if self.domains is not None and sample.get("domain") not in self.domains:
            return None
        text = sample.get("text", "").strip()
        return text if text else None


# ══════════════════════════════════════════════════════════════════════════
# ChunkedParquetLoader — file .parquet local (sách, corpus nội bộ, ...)
# ══════════════════════════════════════════════════════════════════════════

class ChunkedParquetLoader(_BaseChunkedLoader):
    """
    Load dataset từ 1 file .parquet LOCAL theo từng chunk — phù hợp với
    file lớn (>1GB) trên RAM thấp (Colab free tier).

    Dùng HuggingFace datasets streaming mode đọc theo row group của
    parquet, KHÔNG load toàn bộ file vào RAM một lần.

    Hỗ trợ:
        - Chọn tên cột text (mặc định "text")
        - Lọc theo metadata tuỳ ý qua filter_fn (lambda sample → bool)
        - Resume từ chunk cụ thể (giống ChunkedWikiLoader)

    Usage — đơn giản nhất:
        loader = ChunkedParquetLoader(cfg, tokenizer, "data/books.parquet")

    Chỉ lấy sách của một tác giả cụ thể:
        loader = ChunkedParquetLoader(
            cfg, tokenizer, "data/books.parquet",
            filter_fn=lambda s: s.get("author") == "Nguyễn Du",
        )

    Cột text tên khác (ví dụ "content"):
        loader = ChunkedParquetLoader(
            cfg, tokenizer, "data/books.parquet",
            text_col="content",
        )

    Resume từ chunk 20:
        loader = ChunkedParquetLoader(
            cfg, tokenizer, "data/books.parquet",
            start_chunk=20,
        )

    Dùng qua config (không có filter_fn):
        cfg.data.source           = "parquet"
        cfg.data.parquet_path     = "data/books.parquet"
        cfg.data.parquet_text_col = "text"
        main(cfg)

    ────────────────────────────────────────────────────────────────────────
    Lưu ý kỹ thuật — vì sao dùng HuggingFace thay vì pandas:

    pandas.read_parquet() load TOÀN BỘ file vào RAM một lần — không khả thi
    với file >1GB trên Colab (thường chỉ có 12GB RAM, còn phải chia cho model
    và batch). HuggingFace datasets streaming đọc tuần tự theo row group của
    parquet, mỗi lúc chỉ giữ 1 row group trong RAM (~vài MB).

    skip() khi resume tính theo SỐ ROW THÔ (kể cả row bị filter bỏ) —
    sai lệch nhỏ này chấp nhận được để tránh scan lại toàn bộ file.
    ────────────────────────────────────────────────────────────────────────
    """

    def __init__(
        self,
        cfg,
        tokenizer,
        parquet_path : str,
        text_col     : str = "text",
        start_chunk  : int = 0,
        filter_fn    = None,   # Callable[[dict], bool] | None
    ):
        """
        parquet_path : đường dẫn tới file .parquet (absolute hoặc relative)
        text_col     : tên cột chứa nội dung văn bản (mặc định "text")
        filter_fn    : hàm lọc sample — nhận dict 1 row, trả về True để giữ.
                       None = lấy tất cả.
                       Ví dụ: filter_fn=lambda s: s.get("author") == "Nguyễn Du"
                       Ví dụ: filter_fn=lambda s: s.get("genre") in {"fiction", "history"}
        """
        self.parquet_path = parquet_path
        self.text_col     = text_col
        self.filter_fn    = filter_fn
        super().__init__(cfg, tokenizer, start_chunk=start_chunk)

    def _load_dataset(self):
        return load_dataset(
            "parquet",
            data_files={"train": self.parquet_path},
            split="train",
            streaming=True,
        )

    def _extract_text(self, sample: dict) -> str | None:
        # Bước 1: áp filter metadata nếu có
        if self.filter_fn is not None and not self.filter_fn(sample):
            return None

        # Bước 2: lấy text từ đúng cột
        text = sample.get(self.text_col)
        if not isinstance(text, str):
            return None
        return text.strip() or None