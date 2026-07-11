"""
build_price_tokenizer.py — Build tokenizer vocab ĐÓNG cho model đọc giá
================================================================================
KHÁC HẲN train_tokenizer.py (app/memlm) — ở đó cần TRAIN BPE trên corpus
lớn vì vocab tiếng Việt không biết trước hết. Ở đây vocab đã BIẾT TRƯỚC
HOÀN TOÀN (4 kênh O/H/L/C x 1024 bin + vài marker cấu trúc) — không cần
train gì cả, chỉ cần LIỆT KÊ hết token rồi build WordLevel tokenizer trực
tiếp. Chạy 1 lần, ra kết quả giống hệt mỗi lần chạy lại (deterministic).

Price token dùng format BRACKET "<px_O_512>" (đã chốt trước đó trong dự
án) — phải add qua add_tokens() vì '<'/'>' bị pre-tokenizer coi là ký tự
tách từ riêng, không thể để tự nhận diện nguyên khối như "O_512" (không
ngoặc) được.

Thứ tự id PHẢI khớp đúng config.py (SPECIAL_TOKENS, STRUCTURE_MARKERS) —
và khớp với id hardcode sẵn trong train.py::build_model()
(pad_token_id=3, bos_token_id=1, eos_token_id=2).

Chạy:
    python app/llama/build_price_tokenizer.py
    # hoặc chỉ định output khác:
    python app/llama/build_price_tokenizer.py --output-dir custom_tokenizer_price
"""

import argparse
import os

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from app.llama.config import (
    SPECIAL_TOKENS,
    STRUCTURE_MARKERS,
    PRICE_LETTERS,
    N_PRICE_BINS,
    REAL_VOCAB_SIZE,
)


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "custom_tokenizer_price")


def _price_token(letter: str, bin_idx: int) -> str:
    """Format bracket đã chốt: <px_O_512>, <px_H_0>, ..."""
    return f"<px_{letter}_{bin_idx}>"


def build_price_tokenizer(output_dir: str) -> PreTrainedTokenizerFast:
    """
    Build tokenizer vocab đóng, KHÔNG cần corpus train.

    Thứ tự add token QUYẾT ĐỊNH id — giữ đúng thứ tự dưới đây để khớp
    config.py và các id hardcode sẵn trong train.py::build_model():
        0: <unk>   1: <bos>   2: <eos>   3: <pad>
        4: <chart> 5: </chart>
        6..4101   : price token (O rồi H rồi L rồi C, mỗi loại 1024 bin)
    """
    core = Tokenizer(WordLevel(unk_token="<unk>"))
    core.pre_tokenizer = Whitespace()

    # add_tokens() theo ĐÚNG thứ tự mong muốn -> id tăng dần từ 0
    special_and_markers = list(SPECIAL_TOKENS) + list(STRUCTURE_MARKERS)
    core.add_tokens(special_and_markers)

    price_tokens = [
        _price_token(letter, i)
        for letter in PRICE_LETTERS
        for i in range(N_PRICE_BINS)
    ]
    core.add_tokens(price_tokens)

    # Đánh dấu special tokens (để tokenizer xử lý đúng add_special_tokens=True,
    # skip_special_tokens khi decode, v.v.) — <chart>/</chart> KHÔNG đánh dấu
    # special vì chúng là marker cấu trúc, không phải special token chuẩn HF.
    core.add_special_tokens(list(SPECIAL_TOKENS))

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=core,
        unk_token="<unk>",
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
    )

    os.makedirs(output_dir, exist_ok=True)
    fast_tokenizer.save_pretrained(output_dir)

    print(f"✓ Đã build tokenizer vocab đóng -> {output_dir}/")
    print(f"  Vocab size thực tế : {len(fast_tokenizer)}  (kỳ vọng {REAL_VOCAB_SIZE})")
    for name, tok in [("unk", "<unk>"), ("bos", "<bos>"), ("eos", "<eos>"), ("pad", "<pad>")]:
        print(f"  {name:<4} '{tok}' -> id={fast_tokenizer.convert_tokens_to_ids(tok)}")

    return fast_tokenizer


def _sanity_check(tokenizer_dir: str):
    """Roundtrip đơn giản — encode rồi decode phải khớp lại đúng token gốc,
    không có <unk> nào lọt vào (vì vocab đóng, mọi input HỢP LỆ đều có ID)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_dir)

    sample = "<chart> <px_O_512> <px_H_780> <px_L_490> <px_C_650> </chart>"
    ids     = tok.encode(sample, add_special_tokens=False)
    decoded = tok.decode(ids, skip_special_tokens=False)

    unk_id = tok.convert_tokens_to_ids("<unk>")
    n_unk  = ids.count(unk_id)

    print(f"\n  Sanity check:")
    print(f"  Input  : {sample}")
    print(f"  Tokens : {ids}  ({len(ids)} token)")
    print(f"  Decoded: {decoded}")
    print(f"  <unk> count: {n_unk}  {'✓' if n_unk == 0 else '✗ WARN'}")
    assert n_unk == 0, "Vocab đóng nhưng vẫn ra <unk> — kiểm tra lại format bracket."
    assert len(ids) == 6, f"Kỳ vọng đúng 6 token (2 marker + 4 price), nhận {len(ids)}"
    print("  ✓ Sanity check PASSED")


def main():
    parser = argparse.ArgumentParser(description="Build tokenizer vocab đóng cho model đọc giá")
    parser.add_argument("--output-dir", type=str, default=_DEFAULT_OUTPUT_DIR)
    parser.add_argument("--skip-sanity", action="store_true")
    args = parser.parse_args()

    build_price_tokenizer(args.output_dir)
    if not args.skip_sanity:
        _sanity_check(args.output_dir)

    print(f"\nCập nhật config nếu output-dir khác mặc định:")
    print(f"  cfg.tokenizer.pretrained_name = '{args.output_dir}'")


if __name__ == "__main__":
    main()