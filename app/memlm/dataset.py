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

────────────────────────────────────────────────────────────────────────────
2 chế độ Dataset:

    TokenChunkDataset (mặc định, cfg.data.sequential_mode=False):
        Mỗi document bị cắt thành các đoạn độc lập seg_len token.
        DataLoader shuffle tự do — đa dạng batch tốt.
        Nhược điểm: M chỉ tích lũy trong phạm vi 1 segment (512 token),
        không carry-over giữa các segment của cùng document.

    SequentialDocumentDataset (cfg.data.sequential_mode=True):
        Shuffle ở cấp DOCUMENT, nhưng các segment của cùng 1 document
        được xử lý TUẦN TỰ trong batch — M thực sự carry-over xuyên
        suốt document, đúng mục đích thiết kế Context Memory.

        Cách hoạt động:
            - Documents được shuffle ngẫu nhiên (đảm bảo diversity)
            - Mỗi document tạo ra N window tuần tự (sliding window với stride)
            - DataLoader KHÔNG shuffle (shuffle=False)
            - Trainer dùng is_doc_start / is_doc_end để biết khi nào
              reset M (đầu doc mới) và khi nào carry-over (giữa doc)

        Phù hợp khi:
            - Muốn M học tích lũy context dài thật sự
            - Document dài (sách, bài báo dài) — nhiều window/doc
            - half_life đủ lớn để M có thời gian học qua nhiều window

        Không phù hợp khi:
            - Document rất ngắn (1-2 segment) — lợi ích không đáng
            - Batch size lớn — các batch liên tiếp ít diverse hơn
────────────────────────────────────────────────────────────────────────────
"""

import random
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset


# ══════════════════════════════════════════════════════════════════════════
# Dataset classes
# ══════════════════════════════════════════════════════════════════════════

class TokenChunkDataset(Dataset):
    """
    Chế độ mặc định: mỗi document bị cắt thành các đoạn ĐỘC LẬP.
    DataLoader có thể shuffle tự do — đa dạng batch tốt.
    M không carry-over giữa các segment của cùng document.

    [FIX 4a] Bỏ qua document ngắn hơn 50% seg_len+1 token — tránh batch có
    padding >80% gây GPU utilization kém và loss không nhất quán.

    [FIX 4b] Giữ lại phần đuôi của document (token cuối không đủ seg_len)
    nếu dài hơn min_tail_len. Phiên bản cũ bỏ mất phần này hoàn toàn.
    Ví dụ: doc 1200 token, seg_len=512:
        Cũ:  chunk[0:512], chunk[512:1024] — bỏ 176 token cuối
        Mới: chunk[0:512], chunk[512:1024], chunk[688:1200] (overlap OK)
    """

    def __init__(
        self,
        documents    : list[list[int]],
        seg_len      : int,
        min_tail_len : int = 64,   # giữ lại đuôi nếu >= min_tail_len token (sau khi trừ label shift)
    ):
        self.samples = []
        n_short_skipped = 0

        for doc in documents:
            # [FIX 4a] Bỏ doc ngắn hơn 50% seg_len+1 — quá nhiều padding, ít thông tin
            if len(doc) < (seg_len + 1) // 2:
                n_short_skipped += 1
                continue

            n_full = len(doc) // (seg_len + 1)
            chunks = []

            for i in range(n_full):
                start = i * (seg_len + 1)
                end   = start + seg_len + 1
                chunks.append(doc[start:end])

            # [FIX 4b] Giữ phần đuôi nếu đủ dài
            tail_start = n_full * (seg_len + 1)
            tail       = doc[tail_start:]
            if len(tail) >= min_tail_len + 1:   # +1 vì cần ít nhất 1 label token
                # Lấy seg_len token cuối để đuôi có độ dài chuẩn (tránh batch lệch)
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

    Sliding window:
        Mỗi document tạo ra các window chồng lên nhau với stride < seg_len:
            window 0: token[0 : seg_len]
            window 1: token[stride : stride + seg_len]
            window 2: token[2*stride : 2*stride + seg_len]
            ...
        Stride nhỏ -> nhiều window hơn, M được train lâu hơn trên mỗi doc.
        Stride = seg_len -> không overlap (giống TokenChunkDataset về số lượng).

    is_doc_start = True  -> đầu document mới, trainer reset M
    is_doc_end   = True  -> cuối document, trainer có thể flush M
    """

    def __init__(
        self,
        documents   : list[list[int]],
        seg_len     : int,
        stride      : int  = None,    # None = seg_len (không overlap)
        shuffle_docs: bool = True,
    ):
        """
        stride: số token dịch chuyển giữa 2 window liên tiếp.
                Khuyến nghị:
                    seg_len // 2  — overlap 50%, M học kỹ hơn mỗi đoạn văn
                    seg_len // 4  — overlap 75%, phù hợp document rất dài
                    seg_len       — không overlap (mặc định, nhanh nhất)
        """
        if stride is None:
            stride = seg_len

        # [FIX 4a] Filter nhất quán với TokenChunkDataset — bỏ doc không đủ
        # 50% seg_len+1. Comment cũ đã đúng hướng nhưng
        # bị comment out; giờ bật lại và thêm warning.
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

            # [FIX 4b] Giữ phần đuôi nếu đủ dài — nhất quán với TokenChunkDataset
            # Lấy seg_len+1 token cuối để window có độ dài chuẩn
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
# Factory — chọn dataset class theo config
# ══════════════════════════════════════════════════════════════════════════

def make_dataloaders(
    train_docs : list[list[int]],
    val_docs   : list[list[int]],
    cfg,
    pad_id     : int,
):
    """
    Tạo train/val DataLoader theo cfg.data.sequential_mode.

    sequential_mode=False (mặc định):
        TokenChunkDataset + shuffle=True — hành vi gốc

    sequential_mode=True:
        SequentialDocumentDataset + shuffle=False — M carry-over đúng cách
        Val loader luôn dùng TokenChunkDataset (đánh giá độc lập từng segment)
    """
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
            train_ds, batch_size=bs, shuffle=False,   # KHÔNG shuffle — thứ tự quan trọng
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
# Base class — logic chung cho mọi nguồn streaming dataset
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

    Kết hợp tốt nhất với sequential_mode=True cho document dài (sách):
        cfg.data.sequential_mode = True
        cfg.data.window_stride   = cfg.data.seg_len // 2
        loader = ChunkedParquetLoader(cfg, tokenizer, "data/books.parquet")
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