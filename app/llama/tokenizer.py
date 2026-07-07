"""
app/llama/tokenizer.py — Tokenizer THẬT cho nhánh Llama (tương thích HF/TRL)
===============================================================================
Khác app/memlm/tokenizer.py (wrapper tự chế — price token cộng ID ẢO bên
ngoài tokenizer thật, KHÔNG hoạt động khi Trainer/TRL gọi tokenizer(text)
trực tiếp), file này:

    1. Load base BPE tokenizer (train bằng app/memlm/scripts/train_tokenizer.py)
    2. add_tokens() THẬT SỰ cho toàn bộ price vocab — trở thành 1 phần vocab
       chính thức, ID liên tục, nằm TRONG chính tokenizer.
    3. Lưu ra bản riêng (output_dir) — không đụng bản BPE gốc, không phá
       checkpoint cũ của nhánh app/memlm/.

Sau bước này, tokenizer trả về là PreTrainedTokenizerFast THẬT, dùng trực
tiếp được với transformers.Trainer, trl.SFTTrainer/DPOTrainer/RewardTrainer,
và model.resize_token_embeddings(len(tokenizer)).

────────────────────────────────────────────────────────────────────────────
QUYẾT ĐỊNH THIẾT KẾ QUAN TRỌNG — ĐỔI ĐỊNH DẠNG PRICE TOKEN:

Bản cũ dùng "O_512", "H_800", ... trần trụi. Với add_tokens() thật (không
còn strict_chart_mode để giới hạn phạm vi <chart>...</chart> tại thời điểm
encode), token trần "O_512" có nguy cơ:
    - Trùng ngẫu nhiên với ký hiệu khoa học/code trong text tự nhiên
      (Wikipedia, VTSNLP) mà không có cách nào phân biệt ngữ cảnh.
    - BPE base tokenizer có thể đã học "O", "_5", "12" thành sub-token khác
      nhau — add_tokens() có ưu tiên cao hơn BPE nên VẪN match đúng, nhưng
      rủi ro va chạm ngữ nghĩa với text thường vẫn còn.

Giải pháp: đổi sang "<px_O_512>" (có dấu ngoặc nhọn bao quanh, prefix "px_")
— gần như KHÔNG THỂ xuất hiện ngẫu nhiên trong text tự nhiên. An toàn nằm
NGAY TRONG HÌNH DẠNG TOKEN, không phụ thuộc parse boundary lúc runtime nữa.

Hệ quả: mọi text sinh từ ChartCodec/CurriculumGenerator/ActionDataGen (định
dạng "O_512 H_800 ...") cần được convert bằng convert_legacy_price_tokens()
TRƯỚC KHI tokenize. Việc này được làm tự động trong app/llama/dataset.py khi
source="parquet"/"mix" — KHÔNG cần sửa lại chartcodec.py/curriculum_generator.py.
────────────────────────────────────────────────────────────────────────────
"""

import os
import re
from typing import List

from transformers import AutoTokenizer, PreTrainedTokenizerFast


# Bỏ hẳn tiền tố "px" và dấu ngoặc nhọn đi, trả về đúng định dạng cũ trong file Parquet
def _price_token(channel: str, bin_idx) -> str:
    return f"{channel}_{bin_idx}"  # Kết quả: "O_512", "H_800",...


def build_price_tokens(n_bins: int = 1024) -> List[str]:
    tokens = [_price_token(c, i) for c in "OHLC" for i in range(n_bins)]
    tokens += ["<chart>", "</chart>"]  # Giữ lại các thẻ bao bọc nếu data có dùng
    return tokens


def build_llama_tokenizer(
    base_tokenizer_dir: str,
    output_dir        : str,
    n_price_bins       : int = 1024,
) -> PreTrainedTokenizerFast:
    """
    Load base BPE tokenizer, add_tokens() thật cho price vocab, lưu bản riêng.

    Idempotent: nếu output_dir đã tồn tại VÀ đã có price vocab, load thẳng
    từ đó thay vì add lại — tránh lệch ID nếu gọi hàm này nhiều lần (mỗi lần
    add_tokens() thêm token đã tồn tại sẽ bị bỏ qua tự động bởi HF, nhưng an
    toàn hơn là short-circuit sớm để không phụ thuộc hành vi đó).
    """
    if os.path.isdir(output_dir):
        tok = AutoTokenizer.from_pretrained(output_dir)
        if _price_token("O", 0) in tok.get_vocab():
            print(f"  Tokenizer đã có price vocab sẵn, load thẳng từ {output_dir}")
            return tok
        print(f"  {output_dir} tồn tại nhưng thiếu price vocab — build lại.")

    print(f"  Loading base tokenizer: {base_tokenizer_dir}")
    tok = AutoTokenizer.from_pretrained(base_tokenizer_dir)

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    price_tokens = build_price_tokens(n_price_bins)
    vocab_before = len(tok)
    num_added    = tok.add_tokens(price_tokens)
    vocab_after  = len(tok)

    print(f"  Vocab trước: {vocab_before:,} | thêm: {num_added:,} | sau: {vocab_after:,}")
    assert vocab_after == vocab_before + num_added, (
        "Số token thêm không khớp — có thể base tokenizer đã chứa sẵn một số "
        "price token trùng tên (không nên xảy ra với base BPE mới train)."
    )

    os.makedirs(output_dir, exist_ok=True)
    tok.save_pretrained(output_dir)
    print(f"  ✓ Đã lưu tokenizer (BPE + price token thật) → {output_dir}/")
    return tok


def load_llama_tokenizer(cfg) -> PreTrainedTokenizerFast:
    """Entry point từ Config — build nếu cần, load nếu đã có sẵn."""
    return build_llama_tokenizer(
        base_tokenizer_dir=cfg.tokenizer.base_tokenizer_dir,
        output_dir=cfg.tokenizer.output_dir,
        n_price_bins=cfg.tokenizer.n_price_bins,
    )



if __name__ == "__main__":
    import sys

    base_dir = sys.argv[1] if len(sys.argv) > 1 else "custom_tokenizer"
    out_dir  = sys.argv[2] if len(sys.argv) > 2 else "custom_tokenizer_llama"