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

⚠️ RỦI RO ĐàVERIFY — REGEX CÓ THỂ MATCH NHẦM TRONG TEXT TỰ NHIÊN:

    "Hằng số H_0 (Hubble)"        -> bị hiểu nhầm thành price token H_0
    "Phương trình C_1, C_2"       -> bị hiểu nhầm thành price token C_1, C_2
    "Mẫu vật L_99 tại bảo tàng"   -> bị hiểu nhầm thành price token L_99
    "Chủng vi khuẩn O_157"        -> bị hiểu nhầm thành price token O_157

Đây là vấn đề THẬT nếu bạn train trên dữ liệu CHUNG (Wikipedia/VTSNLP)
lẫn với dữ liệu trading. Hai hướng xử lý:

    1. Nếu CHỈ train trên dữ liệu trading thuần (không lẫn Wikipedia) —
       rủi ro này gần như không đáng kể, vì văn bản trading hiếm khi chứa
       các ký hiệu khoa học dạng "C_1", "H_0" với nghĩa khác.

    2. Nếu TRỘN dữ liệu trading với Wikipedia/corpus chung — nên thêm
       một marker bắt buộc bao quanh price token (ví dụ chỉ nhận diện
       price token khi nằm trong cặp <chart>...</chart>, KHÔNG nhận diện
       price token đứng tự do ngoài cặp marker này). Xem
       `strict_chart_mode` bên dưới — mặc định TẮT để giữ tương thích với
       thiết kế gốc, BẬT nếu bạn train trên dữ liệu trộn.
────────────────────────────────────────────────────────────────────────────
"""

import re
from transformers import AutoTokenizer


# Regex nhận diện token giá: O_512, H_3, L_999, C_0 ...
PRICE_TOKEN_RE = re.compile(r"<chart>|</chart>|\b[OHLC]_\d{1,4}\b")

# Regex chỉ nhận diện price token KHI nằm trong cặp <chart>...</chart>
# Dùng khi strict_chart_mode=True (xem cảnh báo phía trên).
_CHART_BLOCK_RE = re.compile(r"<chart>.*?</chart>", re.DOTALL)


def _build_price_vocab(base_len: int, n_bins: int = 1024) -> dict:
    tokens = [f"{p}_{i}" for p in "OHLC" for i in range(n_bins)]
    tokens += ["<chart>", "</chart>"]      # +2 token marker
    return {tok: base_len + i for i, tok in enumerate(tokens)}


class VietnameseTokenizer:
    def __init__(
        self,
        pretrained_name : str = "vinai/phobert-base",
        use_fast        : bool = False,
        n_price_bins    : int = 1024,
        strict_chart_mode: bool = False,
    ):
        """
        n_price_bins      : số bin giá cho mỗi loại O/H/L/C (mặc định 1024
                             → tổng 4096 price token + 2 marker = 4098).
        strict_chart_mode : nếu True, CHỈ nhận diện price token khi nằm
                             trong cặp <chart>...</chart>; mọi chuỗi dạng
                             "O_5"/"H_0"/... NẰM NGOÀI cặp marker sẽ được
                             coi là text thường (qua BPE PhoBERT), tránh
                             match nhầm với ký hiệu khoa học/toán học
                             trong corpus chung. BẬT tham số này nếu bạn
                             train lẫn dữ liệu trading với Wikipedia/VTSNLP.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_name, use_fast=use_fast)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.strict_chart_mode = strict_chart_mode

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

        strict_chart_mode=False (mặc định, GIỮ NGUYÊN hành vi gốc):
            Nhận diện price token ở BẤT KỲ ĐÂU trong text — rủi ro match
            nhầm ký hiệu khoa học (xem cảnh báo đầu file).

        strict_chart_mode=True:
            CHỈ nhận diện price token khi nằm trong cặp <chart>...</chart>.
            Phần text ngoài cặp marker được giữ nguyên dạng "text" kể cả
            khi trùng pattern O_x/H_x/L_x/C_x.
        """
        if not self.strict_chart_mode:
            return self._split_segments_loose(text)
        return self._split_segments_strict(text)

    def _split_segments_loose(self, text: str) -> list[tuple[str, str]]:
        segments, pos = [], 0
        for m in PRICE_TOKEN_RE.finditer(text):
            if m.start() > pos:
                chunk = text[pos:m.start()]
                if chunk.strip():
                    segments.append(("text", chunk))
            segments.append(("price", m.group(0)))
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

            # Bên TRONG block <chart>...</chart> — parse price token bình thường
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
        thứ tự cho từng câu — đã verify bằng test kể cả trường hợp một câu
        toàn price token (không có text chunk nào).
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
    """
    Entry point để load tokenizer từ TokenizerConfig.

    cfg.tokenizer.strict_chart_mode điều khiển việc price token chỉ được
    nhận diện trong cặp <chart>...</chart> hay ở bất kỳ đâu — xem cảnh báo
    đầu file để biết khi nào cần bật True.
    """
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
