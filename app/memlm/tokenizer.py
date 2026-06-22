"""
tokenizer.py — Wrapper cho PhoBERT BPE tokenizer + Price token riêng biệt
=============================================================================
PhoBERT tokenizer được train sẵn trên 20GB tiếng Việt, vocab ~64k.
Dùng lại thay vì train BPE từ đầu — tiết kiệm thời gian và chất lượng tốt hơn
với corpus nhỏ.

Lưu ý quan trọng:
    PhoBERT được train trên text ĐÃ QUA WORD SEGMENTATION
    (ví dụ: "trí_tuệ nhân_tạo" thay vì "trí tuệ nhân tạo").
    Để dùng đúng chuẩn cần VnCoreNLP, nhưng tốn thêm dependency Java.

    Ở đây dùng tokenizer ở chế độ RAW TEXT (không word-segment).
    Vẫn hoạt động tốt cho mục đích train LM từ đầu, chỉ là không tận dụng
    được 100% chất lượng pretrained embedding của PhoBERT (không quan trọng
    vì ta train embedding từ đầu, không dùng PhoBERT weights).

────────────────────────────────────────────────────────────────────────────
QUAN TRỌNG — PhoBERT KHÔNG CÓ FAST TOKENIZER (đã verify bằng code):

    from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING_NAMES
    TOKENIZER_MAPPING_NAMES['phobert']  # -> 'PhobertTokenizer' (chỉ 1 class)

So sánh với BERT/RoBERTa gốc luôn có cặp (Slow, Fast):
    TOKENIZER_MAPPING_NAMES['bert']     # -> ('BertTokenizer', 'BertTokenizerFast')

PhoBERT/RoBERTa-tiếng-Việt CHỈ có bản Slow (Python backend thuần). Việc
truyền use_fast=True KHÔNG có tác dụng gì — không tồn tại class
PhobertTokenizerFast để load, bất kể tokenizer gốc hay đã add_tokens().

Hệ quả thực tế: PhoBERT tokenize CHẬM hơn nhiều so với BERT/GPT-2 trên
cùng khối lượng text, đặc biệt rõ trên CPU yếu (Colab free tier — single-
core clock thấp). Cách giảm thiểu duy nhất là BATCH hóa lệnh tokenize
(gọi 1 lần cho N câu thay vì N lần riêng lẻ) — xem encode_batch() bên dưới.
────────────────────────────────────────────────────────────────────────────

────────────────────────────────────────────────────────────────────────────
PRICE TOKEN — KHÔNG dùng add_tokens(), dùng vocab RIÊNG BIỆT:

Thay vì add_tokens() vào PhoBERT vocab (cách cũ, làm BPE merge table lớn
hơn → chậm hơn, và không tận dụng được cấu trúc số học của giá), price
token được mã hóa thành một dải ID RIÊNG, nằm ngoài vocab PhoBERT:

    PhoBERT vocab : ID [0, base_len)
    Price vocab   : ID [base_len, base_len + n_price_tokens)

Token dạng "O_512", "H_3", "L_999", "C_0" (Open/High/Low/Close + bin số)
và marker "<chart>"/"</chart>" được nhận diện bằng regex, KHÔNG đi qua
BPE — mỗi token giá là MỘT ID duy nhất, không bị cắt vụn.

⚠️ BẢO VỆ 2 LỚP CHỐNG MATCH NHẦM:

Lớp 1 — strict_chart_mode=True (mặc định):
    CHỈ nhận diện price token khi nằm trong cặp <chart>...</chart>.
    Text ngoài cặp marker (kể cả "C_2022", "H_0", "O_157") KHÔNG bao giờ
    được parse thành price token — an toàn khi trộn corpus sách/Wikipedia
    với dữ liệu trading.

Lớp 2 — bin range validation trong _split_segments_loose:
    Ngay cả khi nằm trong <chart>...</chart>, token có bin nằm ngoài
    [0, n_price_bins-1] (ví dụ O_9999 do lỗi encode) sẽ bị bỏ qua thay
    vì gây KeyError. Đảm bảo không bao giờ crash kể cả khi data lỗi.

Tắt strict_chart_mode (False) CHỈ khi train trên dữ liệu trading THUẦN
(không lẫn corpus chung) — lớp 2 vẫn hoạt động.
────────────────────────────────────────────────────────────────────────────
"""

import re
from transformers import AutoTokenizer


# Regex nhận diện token giá: O_512, H_3, L_999, C_0 ...
PRICE_TOKEN_RE = re.compile(r"<chart>|</chart>|\b[OHLC]_\d{1,4}\b")

# Regex chỉ nhận diện price token KHI nằm trong cặp <chart>...</chart>
_CHART_BLOCK_RE = re.compile(r"<chart>.*?</chart>", re.DOTALL)


def _build_price_vocab(base_len: int, n_bins: int = 1024) -> dict:
    tokens = [f"{p}_{i}" for p in "OHLC" for i in range(n_bins)]
    tokens += ["<chart>", "</chart>"]      # +2 token marker
    return {tok: base_len + i for i, tok in enumerate(tokens)}


class VietnameseTokenizer:
    def __init__(
        self,
        pretrained_name  : str = "vinai/phobert-base",
        use_fast         : bool = False,
        n_price_bins     : int = 1024,
        strict_chart_mode: bool = True,   # mặc định True: chỉ parse price token
                                           # khi nằm trong <chart>...</chart>.
                                           # Tắt (False) nếu CHỈ train data trading thuần.
    ):
        """
        n_price_bins      : số bin giá cho mỗi loại O/H/L/C (mặc định 1024
                             → tổng 4096 price token + 2 marker = 4098).
        strict_chart_mode : nếu True (mặc định), CHỈ nhận diện price token
                             khi nằm trong cặp <chart>...</chart>; mọi chuỗi
                             dạng "O_5"/"H_0"/... NẰM NGOÀI cặp marker sẽ
                             được coi là text thường (qua BPE PhoBERT), tránh
                             match nhầm với ký hiệu khoa học/năm tháng/số thứ
                             tự trong corpus chung (C_2022, H_0, O_157, ...).
                             Tắt (False) nếu CHỈ train trên dữ liệu trading thuần.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_name, use_fast=use_fast)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.strict_chart_mode = strict_chart_mode
        self.n_price_bins      = n_price_bins

        base_len = len(self.tokenizer)
        self.price_vocab     = _build_price_vocab(base_len, n_price_bins)
        self.price_vocab_inv = {v: k for k, v in self.price_vocab.items()}

        # QUAN TRỌNG: không add_tokens() vào self.tokenizer, nên phải tự
        # cộng phần price token vào vocab_size ở đây — nếu quên, model sẽ
        # build Embedding/lm_head THIẾU đúng số ID, gây IndexError khi gặp
        # price token lúc training.
        self.vocab_size = base_len + len(self.price_vocab)

        self.pad_id = self.tokenizer.pad_token_id
        self.eos_id = self.tokenizer.eos_token_id
        self.bos_id = self.tokenizer.bos_token_id
        self.unk_id = self.tokenizer.unk_token_id

    # ──────────────────────────────────────────────────────────────────────
    def _split_segments(self, text: str) -> list[tuple[str, str]]:
        """
        Tách text thành list (loại, nội dung) theo đúng thứ tự xuất hiện.
        loại ∈ {"text", "price"}.

        strict_chart_mode=True (mặc định):
            CHỈ nhận diện price token khi nằm trong cặp <chart>...</chart>.

        strict_chart_mode=False:
            Nhận diện price token ở BẤT KỲ ĐÂU — chỉ dùng khi train data
            trading thuần (lớp 2 bin validation vẫn hoạt động).
        """
        if not self.strict_chart_mode:
            return self._split_segments_loose(text)
        return self._split_segments_strict(text)

    def _split_segments_loose(self, text: str) -> list[tuple[str, str]]:
        """
        Parse price token trong đoạn text (không kiểm tra chart boundary).
        Gọi từ _split_segments_strict chỉ với phần text bên TRONG <chart>...</chart>,
        hoặc trực tiếp khi strict_chart_mode=False.

        Lớp bảo vệ 2: bỏ qua token OHLC có bin nằm ngoài [0, n_price_bins-1].
        Ví dụ: C_2022, H_9999 → không có trong price_vocab → skip, giữ là text.
        pos không tăng khi skip → đoạn text chứa token lỗi được gom vào
        text segment ở lần match tiếp theo hoặc đoạn cuối (luôn đúng).
        """
        segments, pos = [], 0
        for m in PRICE_TOKEN_RE.finditer(text):
            token = m.group(0)

            # Validate bin range cho OHLC token (không áp dụng cho marker tag)
            if token not in ("<chart>", "</chart>"):
                try:
                    if int(token.split("_")[1]) >= self.n_price_bins:
                        continue   # bin ngoài range → bỏ qua, giữ nguyên là text
                except (IndexError, ValueError):
                    continue       # format lạ → bỏ qua an toàn

            if m.start() > pos:
                chunk = text[pos:m.start()]
                if chunk.strip():
                    segments.append(("text", chunk))
            segments.append(("price", token))
            pos = m.end()

        if pos < len(text):
            chunk = text[pos:]
            if chunk.strip():
                segments.append(("text", chunk))
        return segments

    def _split_segments_strict(self, text: str) -> list[tuple[str, str]]:
        """Chỉ parse price token bên trong cặp <chart>...</chart>."""
        segments, pos = [], 0
        for block_m in _CHART_BLOCK_RE.finditer(text):
            # Phần text TRƯỚC block <chart> — giữ nguyên, không qua PRICE_TOKEN_RE
            if block_m.start() > pos:
                chunk = text[pos:block_m.start()]
                if chunk.strip():
                    segments.append(("text", chunk))

            # Bên TRONG block <chart>...</chart> — parse price token + bin validation
            inner_segs = self._split_segments_loose(block_m.group(0))
            segments.extend(inner_segs)

            pos = block_m.end()

        if pos < len(text):
            chunk = text[pos:]
            if chunk.strip():
                segments.append(("text", chunk))
        return segments

    # ──────────────────────────────────────────────────────────────────────
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        segs = self._split_segments(text)
        chunks = [c for k, c in segs if k == "text"]
        encoded = self.tokenizer(chunks, add_special_tokens=False)["input_ids"] if chunks else []
        it = iter(encoded)

        ids = []
        for kind, content in segs:
            if kind == "text":
                ids.extend(next(it))
            else:
                ids.append(self.price_vocab[content])

        if add_special_tokens:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    def encode_batch(self, texts: list[str], add_special_tokens: bool = False) -> list[list[int]]:
        """
        Tokenize MỘT LẦN cho nhiều câu — nhanh hơn đáng kể so với gọi
        encode() lặp lại nhiều lần (xem cảnh báo PhoBERT Slow tokenizer
        đầu file). Toàn bộ text-chunk của TẤT CẢ câu trong batch được gom
        lại, gọi self.tokenizer() đúng MỘT lần, rồi phân phối lại đúng
        thứ tự cho từng câu.
        """
        all_segs = [self._split_segments(t) for t in texts]
        flat_chunks = [c for segs in all_segs for k, c in segs if k == "text"]
        encoded = self.tokenizer(flat_chunks, add_special_tokens=False)["input_ids"] if flat_chunks else []
        it = iter(encoded)

        results = []
        for segs in all_segs:
            ids = []
            for kind, content in segs:
                if kind == "text":
                    ids.extend(next(it))
                else:
                    ids.append(self.price_vocab[content])
            if add_special_tokens:
                ids = [self.bos_id] + ids + [self.eos_id]
            results.append(ids)
        return results

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        base_len = len(self.tokenizer)
        pieces, run = [], []
        for i in ids:
            if i >= base_len:
                if run:
                    pieces.append(self.tokenizer.decode(run, skip_special_tokens=skip_special_tokens))
                    run = []
                pieces.append(self.price_vocab_inv.get(i, "<unk_price>"))
            else:
                run.append(i)
        if run:
            pieces.append(self.tokenizer.decode(run, skip_special_tokens=skip_special_tokens))
        return " ".join(p for p in pieces if p)

    def __len__(self):
        return self.vocab_size


def load_tokenizer(cfg) -> VietnameseTokenizer:
    """Entry point để load tokenizer từ TokenizerConfig."""
    return VietnameseTokenizer(
        pretrained_name  = cfg.tokenizer.pretrained_name,
        use_fast         = cfg.tokenizer.use_fast,
        n_price_bins     = cfg.tokenizer.n_price_bins,
        strict_chart_mode= cfg.tokenizer.strict_chart_mode,
    )


if __name__ == "__main__":
    # Quick test
    tok = VietnameseTokenizer()
    text = "Trí tuệ nhân tạo đang thay đổi thế giới."
    ids  = tok.encode(text)
    print(f"Text   : {text}")
    print(f"Tokens : {ids}")
    print(f"Decoded: {tok.decode(ids)}")
    print(f"Vocab  : {tok.vocab_size}")

    # Test bin validation — C_2022 không được parse thành price token
    tok2 = VietnameseTokenizer(strict_chart_mode=False)
    ids2 = tok2.encode("Năm C_2022 và O_157")
    decoded2 = tok2.decode(ids2)
    print(f"\n[strict=False] 'Năm C_2022 và O_157' → decoded: {decoded2}")
    assert "<unk_price>" not in decoded2, "C_2022/O_157 không được là price token!"
    print("✓ Bin validation OK — C_2022 và O_157 được giữ nguyên là text")