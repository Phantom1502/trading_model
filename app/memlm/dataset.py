"""
dataset.py — Incremental loading cho dữ liệu tiếng Việt
============================================================
RAM hạn chế nên không load toàn bộ dataset một lần.
Thay vào đó: load từng chunk N sample, train xong, giải phóng, load chunk tiếp.

Hỗ trợ 2 nguồn dữ liệu:
    ChunkedWikiLoader   — wikimedia/wikipedia (bản gốc, field "text")
    ChunkedVTSNLPLoader — VTSNLP/vietnamese_curated_dataset (đã curate,
                          field "text" + "domain", chất lượng cao hơn,
                          12.2M rows, có thể lọc theo domain cụ thể)

Cả hai đều là generator — mỗi lần `next()` trả về (train_loader, val_loader)
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
        """
        documents: list các document, mỗi document là list token_ids
        seg_len  : độ dài mỗi sample (input). Label = input dịch 1 vị trí.
        """
        self.samples = []   # list of (token_ids, is_doc_start)

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
    `_load_dataset()` (cách load dataset gốc) và `_extract_text(sample)`
    (cách lấy text + có filter/skip sample đó không).

    Subclass KHÔNG cần override __iter__/__next__/skip-resume logic —
    toàn bộ phần đó dùng chung, tránh trùng code giữa Wikipedia và VTSNLP.
    """

    def __init__(self, cfg, tokenizer, start_chunk: int = 0):
        self.cfg          = cfg
        self.tokenizer     = tokenizer
        self.chunk_size     = cfg.data.chunk_size
        self.seg_len         = cfg.data.seg_len
        self.min_text_len     = cfg.data.min_text_len
        self.val_ratio         = cfg.data.val_ratio
        self.batch_size         = cfg.train.batch_size
        self.total_chunks       = cfg.train.total_chunks
        self.start_chunk        = start_chunk

        self.raw_stream = self._load_dataset()

        # Resume: skip qua các sample thuộc các chunk đã train.
        # Đây là skip THEO SỐ SAMPLE THÔ (kể cả sample bị lọc bỏ vì quá
        # ngắn hoặc không khớp domain), KHÔNG phải theo số document hợp lệ
        # đã đưa vào chunk trước đó. Sai lệch nhỏ này là đánh đổi chấp nhận
        # được để tránh tokenize lại toàn bộ dữ liệu chỉ để đếm chính xác.
        if start_chunk > 0:
            n_skip = start_chunk * self.chunk_size
            print(f"Resume: skip {n_skip:,} sample đầu (tương ứng {start_chunk} chunk đã train)...")
            self.raw_stream = self.raw_stream.skip(n_skip)

        self.stream_iter = iter(self.raw_stream)
        self.exhausted   = False
        self.chunk_count = start_chunk

    # ── Subclass PHẢI override 2 hàm này ───────────────────────────────────
    def _load_dataset(self):
        """Trả về streaming dataset gốc (chưa skip, chưa iter)."""
        raise NotImplementedError

    def _extract_text(self, sample: dict) -> str | None:
        """
        Lấy text từ 1 sample thô. Trả về None nếu sample này nên bị bỏ qua
        (ví dụ không khớp domain mong muốn, hoặc field text rỗng).
        """
        raise NotImplementedError

    # ── Logic chung — không cần override ───────────────────────────────────
    def _load_one_chunk(self) -> list[list[int]] | None:
        """
        Tách 2 pha rõ ràng:
            Pha 1 — thu thập đủ chunk_size text hợp lệ (lọc theo min_text_len,
                    domain...), KHÔNG tokenize gì trong pha này.
            Pha 2 — tokenize TOÀN BỘ list text bằng MỘT lệnh batch duy nhất.

        Lý do quan trọng (đặc biệt với PhoBERT — KHÔNG có Fast tokenizer,
        chỉ có Slow/Python-backend, xem tokenizer.py để biết chi tiết):
        gọi tokenizer.encode() RIÊNG LẺ hàng nghìn lần có overhead Python
        rất lớn (mỗi lệnh phải qua toàn bộ pipeline normalize/pre-tokenize/
        BPE-merge từ đầu). Gọi MỘT lần với list text tận dụng được tối ưu
        nội bộ của HuggingFace cho xử lý hàng loạt, nhanh hơn đáng kể dù
        vẫn cùng một backend Python — đặc biệt quan trọng trên CPU yếu
        (Colab free tier) nơi single-core clock thấp.
        """
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

        # ── Batch tokenize MỘT LẦN cho toàn bộ chunk ──────────────────────
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
# ChunkedWikiLoader — wikimedia/wikipedia (giữ nguyên hành vi cũ)
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
    12.2M rows, field: text, id, domain (25 domain: Science, Internet_and_Telecom,
    Books_and_Literature, ...).

    Chất lượng văn bản tốt hơn Wikipedia thô (đã qua lọc/làm sạch), và có
    thể lọc theo domain cụ thể nếu muốn tập trung train vào một lĩnh vực
    (ví dụ chỉ lấy domain "Science" để tăng tỷ trọng kiến thức khoa học).

    Usage — dùng toàn bộ domain:
        loader = ChunkedVTSNLPLoader(cfg, tokenizer)

    Usage — chỉ lấy một số domain cụ thể:
        loader = ChunkedVTSNLPLoader(cfg, tokenizer, domains=["Science", "Books_and_Literature"])

    Resume từ chunk cụ thể (giống ChunkedWikiLoader):
        loader = ChunkedVTSNLPLoader(cfg, tokenizer, start_chunk=20)
    """

    HF_DATASET_NAME = "VTSNLP/vietnamese_curated_dataset"

    def __init__(self, cfg, tokenizer, start_chunk: int = 0, domains: list[str] = None):
        """
        domains: list tên domain muốn giữ lại (None = lấy tất cả).
                 Tên domain xem trong dataset card, ví dụ:
                 "Science", "Internet_and_Telecom", "Books_and_Literature", ...
        """
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
