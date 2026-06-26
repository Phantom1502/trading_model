"""
dataset.py — Incremental loading cho dữ liệu tiếng Việt
============================================================
RAM hạn chế nên không load toàn bộ dataset một lần.
Thay vào đó: load từng chunk N sample, train xong, giải phóng, load chunk tiếp.

Hỗ trợ 4 nguồn dữ liệu:
    ChunkedWikiLoader    — wikimedia/wikipedia (bản gốc, field "text")
    ChunkedVTSNLPLoader  — VTSNLP/vietnamese_curated_dataset (đã curate,
                           field "text" + "domain", chất lượng cao hơn,
                           12.2M rows, có thể lọc theo domain cụ thể)
    ChunkedParquetLoader — file .parquet local (sách, corpus nội bộ, ...)
                           field "text" + metadata tuỳ chọn (title, author, ...)
    ChunkedMixLoader     — interleave nhiều nguồn parquet local theo tỷ lệ
                           định sẵn (wiki 30%, books 70%, ...)

────────────────────────────────────────────────────────────────────────────
FIX RAM: tokenize streaming theo batch nhỏ (TOKENIZE_BATCH=500)

Phiên bản cũ gom đủ chunk_size text rồi tokenize một lần:
    - 20k text × ~1000 token × 4 bytes = ~80MB token lists
    - + raw text ~200-400MB
    - + Python list overhead → dễ lên 12GB RAM

Phiên bản mới tokenize từng 500 text, giải phóng ngay:
    - RAM peak = 500 text × size thay vì 20k × size → giảm ~40x peak
    - Tổng documents cuối chunk giữ nguyên, chỉ peak RAM khi tokenize giảm

────────────────────────────────────────────────────────────────────────────
2 chế độ Dataset:

    TokenChunkDataset (mặc định, cfg.data.sequential_mode=False):
        Mỗi document bị cắt thành các đoạn độc lập seg_len token.
        DataLoader shuffle tự do — đa dạng batch tốt.

    SequentialDocumentDataset (cfg.data.sequential_mode=True):
        Shuffle ở cấp DOCUMENT, các segment của cùng 1 document
        được xử lý TUẦN TỰ — M thực sự carry-over xuyên suốt document.
────────────────────────────────────────────────────────────────────────────
"""

import random
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import glob as _glob

# Số text tokenize mỗi lần — giữ RAM peak thấp
TOKENIZE_BATCH = 500


# ══════════════════════════════════════════════════════════════════════════
# Dataset classes
# ══════════════════════════════════════════════════════════════════════════

class TokenChunkDataset(Dataset):
    """
    Chế độ mặc định: mỗi document bị cắt thành các đoạn ĐỘC LẬP.
    DataLoader có thể shuffle tự do — đa dạng batch tốt.
    M không carry-over giữa các segment của cùng document.
    """

    def __init__(
        self,
        documents    : list[list[int]],
        seg_len      : int,
        min_tail_len : int = 64,
    ):
        self.samples = []
        n_short_skipped = 0

        for doc in documents:
            if len(doc) < (seg_len + 1) // 2:
                n_short_skipped += 1
                continue

            n_full = len(doc) // (seg_len + 1)
            chunks = []

            for i in range(n_full):
                start = i * (seg_len + 1)
                end   = start + seg_len + 1
                chunks.append(doc[start:end])

            tail_start = n_full * (seg_len + 1)
            tail       = doc[tail_start:]
            if len(tail) >= min_tail_len + 1:
                chunks.append(doc[-(seg_len + 1):])

            for i, chunk in enumerate(chunks):
                self.samples.append({
                    "ids"         : chunk,
                    "is_doc_start": (i == 0),
                    "is_doc_end"  : (i == len(chunks) - 1),
                })

        if n_short_skipped > 0:
            print(f"  [TokenChunkDataset] Bỏ qua {n_short_skipped} doc ngắn hơn {(seg_len+1)//2} token")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        ids  = torch.tensor(item["ids"], dtype=torch.long)
        return {
            "input_ids"   : ids[:-1],
            "labels"      : ids[1:],
            "is_doc_start": item["is_doc_start"],
            "is_doc_end"  : item["is_doc_end"],
        }


class SequentialDocumentDataset(Dataset):
    """
    Chế độ sequential: documents được shuffle ngẫu nhiên, nhưng các
    segment của cùng 1 document được xếp TUẦN TỰ liền nhau.

    Dùng với DataLoader(shuffle=False) — thứ tự samples quan trọng.
    """

    def __init__(
        self,
        documents   : list[list[int]],
        seg_len     : int,
        stride      : int  = None,
        shuffle_docs: bool = True,
    ):
        if stride is None:
            stride = seg_len

        doc_list = [d for d in documents if len(d) >= (seg_len + 1) // 2]
        n_skipped = len(documents) - len(doc_list)
        if n_skipped > 0:
            print(f"  [SequentialDocumentDataset] Bỏ qua {n_skipped} doc ngắn hơn {(seg_len+1)//2} token")

        if shuffle_docs:
            random.shuffle(doc_list)

        self.samples = []

        for doc in doc_list:
            windows = []
            start = 0
            while start + seg_len + 1 <= len(doc):
                windows.append(doc[start : start + seg_len + 1])
                start += stride

            tail_start = (len(windows) - 1) * stride if windows else 0
            tail_covered_end = tail_start + seg_len + 1
            if tail_covered_end < len(doc) and len(doc) - tail_covered_end >= 64:
                windows.append(doc[-(seg_len + 1):])

            for i, chunk in enumerate(windows):
                self.samples.append({
                    "ids"         : chunk,
                    "is_doc_start": (i == 0),
                    "is_doc_end"  : (i == len(windows) - 1),
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
            "is_doc_end"  : item["is_doc_end"],
        }


# ══════════════════════════════════════════════════════════════════════════
# Collate
# ══════════════════════════════════════════════════════════════════════════

def collate_fn(batch, pad_id: int = 0):
    """Pad các sample trong batch về cùng độ dài."""
    max_len = max(item["input_ids"].size(0) for item in batch)

    input_ids    = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels       = torch.full((len(batch), max_len), -100,   dtype=torch.long)
    is_doc_start = []
    is_doc_end   = []

    for i, item in enumerate(batch):
        L = item["input_ids"].size(0)
        input_ids[i, :L] = item["input_ids"]
        labels[i, :L]    = item["labels"]
        is_doc_start.append(item["is_doc_start"])
        is_doc_end.append(item["is_doc_end"])

    return {
        "input_ids"   : input_ids,
        "labels"      : labels,
        "is_doc_start": torch.tensor(is_doc_start, dtype=torch.bool),
        "is_doc_end"  : torch.tensor(is_doc_end,   dtype=torch.bool),
    }


# ══════════════════════════════════════════════════════════════════════════
# Factory
# ══════════════════════════════════════════════════════════════════════════

def make_dataloaders(
    train_docs : list[list[int]],
    val_docs   : list[list[int]],
    cfg,
    pad_id     : int,
):
    seg_len = cfg.data.seg_len
    bs      = cfg.train.batch_size
    collate = lambda b: collate_fn(b, pad_id)

    if getattr(cfg.data, "sequential_mode", False):
        stride = getattr(cfg.data, "window_stride", seg_len)

        train_ds = SequentialDocumentDataset(
            train_docs, seg_len, stride=stride, shuffle_docs=True,
        )
        val_ds = TokenChunkDataset(val_docs, seg_len)

        train_loader = DataLoader(
            train_ds, batch_size=bs, shuffle=False,
            collate_fn=collate, num_workers=0,
        )
        val_loader = DataLoader(
            val_ds, batch_size=bs, shuffle=False,
            collate_fn=collate, num_workers=0,
        )
    else:
        train_ds = TokenChunkDataset(train_docs, seg_len)
        val_ds   = TokenChunkDataset(val_docs,   seg_len)

        train_loader = DataLoader(
            train_ds, batch_size=bs, shuffle=True,
            collate_fn=collate, num_workers=0,
        )
        val_loader = DataLoader(
            val_ds, batch_size=bs, shuffle=False,
            collate_fn=collate, num_workers=0,
        )

    return train_loader, val_loader


# ══════════════════════════════════════════════════════════════════════════
# Base class
# ══════════════════════════════════════════════════════════════════════════

class _BaseChunkedLoader:
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
        """
        Tokenize streaming theo batch nhỏ (TOKENIZE_BATCH) thay vì gom
        hết chunk_size rồi tokenize một lần.

        RAM peak = TOKENIZE_BATCH × text_size thay vì chunk_size × text_size
        → giảm ~40x peak RAM khi tokenize với chunk_size=20_000.
        """
        documents  = []
        batch_texts = []

        while len(documents) + len(batch_texts) < self.chunk_size:
            try:
                sample = next(self.stream_iter)
            except StopIteration:
                self.exhausted = True
                break

            text = self._extract_text(sample)
            if text is None or len(text) < self.min_text_len:
                continue

            batch_texts.append(text)

            # Tokenize và giải phóng ngay khi đủ TOKENIZE_BATCH
            if len(batch_texts) >= TOKENIZE_BATCH:
                token_lists = self.tokenizer.encode_batch(
                    batch_texts, add_special_tokens=False
                )
                documents.extend(ids for ids in token_lists if len(ids) >= 2)
                batch_texts = []   # giải phóng RAM ngay

        # Flush phần còn lại (< TOKENIZE_BATCH)
        if batch_texts:
            token_lists = self.tokenizer.encode_batch(
                batch_texts, add_special_tokens=False
            )
            documents.extend(ids for ids in token_lists if len(ids) >= 2)

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

        train_loader, val_loader = make_dataloaders(
            train_docs, val_docs, self.cfg, self.tokenizer.pad_id,
        )

        mode = "sequential" if getattr(self.cfg.data, "sequential_mode", False) else "chunked"
        print(
            f"[Chunk {current_chunk_idx}] "
            f"docs: {len(documents)} | "
            f"train samples: {len(train_loader.dataset)} | "
            f"val samples: {len(val_loader.dataset)} | "
            f"mode: {mode}"
        )

        return train_loader, val_loader


# ══════════════════════════════════════════════════════════════════════════
# ChunkedWikiLoader
# ══════════════════════════════════════════════════════════════════════════

class ChunkedWikiLoader(_BaseChunkedLoader):
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
# ChunkedVTSNLPLoader
# ══════════════════════════════════════════════════════════════════════════

class ChunkedVTSNLPLoader(_BaseChunkedLoader):
    HF_DATASET_NAME = "VTSNLP/vietnamese_curated_dataset"

    def __init__(self, cfg, tokenizer, start_chunk: int = 0, domains: list[str] = None):
        self.domains = set(domains) if domains else None
        super().__init__(cfg, tokenizer, start_chunk=start_chunk)

    def _load_dataset(self):
        return load_dataset(self.HF_DATASET_NAME, split="train", streaming=True)

    def _extract_text(self, sample: dict) -> str | None:
        if self.domains is not None and sample.get("domain") not in self.domains:
            return None
        text = sample.get("text", "").strip()
        return text if text else None


# ══════════════════════════════════════════════════════════════════════════
# ChunkedParquetLoader
# ══════════════════════════════════════════════════════════════════════════

class ChunkedParquetLoader(_BaseChunkedLoader):
    """
    Load dataset từ 1 file .parquet LOCAL theo từng chunk.
    Dùng HuggingFace streaming — KHÔNG load toàn bộ file vào RAM.
    """

    def __init__(
        self,
        cfg,
        tokenizer,
        parquet_path : str,
        text_col     : str = "text",
        start_chunk  : int = 0,
        filter_fn    = None,
    ):
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
        if self.filter_fn is not None and not self.filter_fn(sample):
            return None
        text = sample.get(self.text_col)
        if not isinstance(text, str):
            return None
        return text.strip() or None


# ══════════════════════════════════════════════════════════════════════════
# ChunkedMixLoader
# ══════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════
# PATCH 1: dataset.py — ChunkedMixLoader với shuffle file order
# Thay thế class ChunkedMixLoader hiện tại bằng version này
# ══════════════════════════════════════════════════════════════════════════

class ChunkedMixLoader(_BaseChunkedLoader):
    """
    Interleave nhiều nguồn parquet local theo tỷ lệ định sẵn.
    File list được shuffle 1 lần lúc init và lưu vào checkpoint —
    đảm bảo resume đúng vị trí, không đọc lại file đã qua.

    Ví dụ setup:
        cfg.data.source = "mix"
        cfg.data.mix.sources = {
            "wiki_vi": ("/content/HLLMDS/wiki_vi/*.parquet", 0.01),
            "wiki_en": ("/content/HLLMDS/wiki_en/*.parquet", 0.20),
            "math"   : ("/content/HLLMDS/math/*.parquet",    0.10),
        }
        cfg.data.mix.stopping_strategy = "all_exhausted"
        cfg.data.mix.shuffle_buffer    = 10_000
    """

    def __init__(self, cfg, tokenizer, start_chunk: int = 0,
                 file_order: dict = None):
        """
        file_order: dict {source_name: [path1, path2, ...]} từ checkpoint.
                    None = shuffle mới (lần đầu train).
        """
        self._file_order = file_order   # lưu trước khi super().__init__ gọi _load_dataset
        super().__init__(cfg, tokenizer, start_chunk=start_chunk)

    def _resolve_file_order(self) -> dict:
        """
        Nếu có file_order từ checkpoint → dùng lại (resume đúng thứ tự).
        Không có → glob + shuffle mới, lưu lại để checkpoint sau.
        """
        mix = self.cfg.data.mix
        if self._file_order is not None:
            print("  [MixLoader] Resume: dùng lại file order từ checkpoint")
            return self._file_order

        order = {}
        for name, (pattern, _) in mix.sources.items():
            files = sorted(_glob.glob(pattern))
            if not files:
                raise FileNotFoundError(
                    f"Không tìm thấy file nào khớp pattern: {pattern}"
                )
            import random as _random
            _random.shuffle(files)
            order[name] = files
            print(f"  [MixLoader] {name}: {len(files)} files (shuffled)")

        self._file_order = order   # lưu lại để caller có thể checkpoint
        return order

    def _load_dataset(self):
        from datasets import interleave_datasets

        mix   = self.cfg.data.mix
        order = self._resolve_file_order()

        if not mix.sources:
            raise ValueError("cfg.data.mix.sources trống")

        names = list(mix.sources.keys())
        probs = [mix.sources[n][1] for n in names]

        total = sum(probs)
        if abs(total - 1.0) > 1e-3:
            raise ValueError(f"Tổng probabilities = {total:.4f}, phải = 1.0")

        print(f"  Mix sources ({len(names)}):")
        for name, prob in zip(names, probs):
            print(f"    {name:<12} {prob*100:.0f}%  ({len(order[name])} files)")
        print(f"  stopping_strategy : {mix.stopping_strategy}")
        print(f"  shuffle_buffer    : {mix.shuffle_buffer:,}")

        datasets = [
            load_dataset(
                "parquet",
                data_files={"train": order[name]},   # list files theo đúng order
                split="train",
                streaming=True,
            )
            for name in names
        ]

        mixed = interleave_datasets(
            datasets,
            probabilities=probs,
            seed=42,
            stopping_strategy=mix.stopping_strategy,
        )

        return mixed.shuffle(seed=42, buffer_size=mix.shuffle_buffer)

    def _extract_text(self, sample: dict) -> str | None:
        text = sample.get(self.cfg.data.parquet_text_col)
        if not isinstance(text, str):
            return None
        return text.strip() or None

    @property
    def file_order(self) -> dict:
        """Trả về file order hiện tại — pretrain.py lưu vào checkpoint."""
        return self._file_order or {}