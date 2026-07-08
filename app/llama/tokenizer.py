"""
tokenizer.py — Load tokenizer cho pipeline pretrain LLaMA tiếng Việt (app/llama)
================================================================================
Tách từ hàm load_tokenizer() trong bản nháp train.py. Logic giữ NGUYÊN, chỉ
đổi tham số đầu vào từ TokenizerConfig rời sang Config đầy đủ (để nhất quán
với các module khác sẽ tách sau — tất cả đều nhận `cfg: Config`).
"""

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from app.llama.config import Config


def load_tokenizer(cfg: Config) -> PreTrainedTokenizerBase:
    """
    Load tokenizer theo cfg.tokenizer.pretrained_name (mặc định
    "custom_tokenizer_llama" — path local, không phải tên trên HF Hub).

    Tự set pad_token = eos_token nếu tokenizer chưa có pad_token (nhiều BPE
    tokenizer kiểu GPT-2/LLaMA không có sẵn pad_token, nhưng
    DataCollatorForLanguageModeling / Trainer cần pad_token_id hợp lệ để pad
    batch khi các sample khác độ dài).
    """
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.tokenizer.pretrained_name,
        use_fast=cfg.tokenizer.use_fast,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def check_tokenizer_matches_model_config(tokenizer: PreTrainedTokenizerBase, cfg: Config) -> None:
    """
    Cảnh báo sớm nếu special token id của tokenizer KHÔNG khớp
    cfg.model.pad/bos/eos_token_id — build_model() (model.py) dùng thẳng
    3 giá trị này để build LlamaConfig, lệch sẽ khiến model học sai ranh
    giới câu mà không có lỗi rõ ràng nào báo ra (fail âm thầm).

    Gọi hàm này ngay sau load_tokenizer(), trước khi build model.
    """
    mismatches = []
    if tokenizer.pad_token_id is not None and tokenizer.pad_token_id != cfg.model.pad_token_id:
        mismatches.append(f"pad_token_id: tokenizer={tokenizer.pad_token_id} vs cfg.model={cfg.model.pad_token_id}")
    if tokenizer.bos_token_id is not None and tokenizer.bos_token_id != cfg.model.bos_token_id:
        mismatches.append(f"bos_token_id: tokenizer={tokenizer.bos_token_id} vs cfg.model={cfg.model.bos_token_id}")
    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id != cfg.model.eos_token_id:
        mismatches.append(f"eos_token_id: tokenizer={tokenizer.eos_token_id} vs cfg.model={cfg.model.eos_token_id}")

    if mismatches:
        print("  ⚠ CẢNH BÁO: special token id của tokenizer lệch với cfg.model:")
        for m in mismatches:
            print(f"      {m}")
        print("      -> Sửa cfg.model.pad/bos/eos_token_id cho khớp tokenizer thật, "
              "hoặc build_model() sẽ dùng nhầm id.")


def sync_vocab_size(tokenizer: PreTrainedTokenizerBase, cfg: Config) -> None:
    """
    Cập nhật cfg.model.vocab_size = len(tokenizer) — PHẢI gọi trước
    build_model(), vì ModelConfig.vocab_size mặc định (24000) chỉ là giá trị
    khởi tạo, không đảm bảo khớp tokenizer thật đang dùng.
    """
    cfg.model.vocab_size = len(tokenizer)