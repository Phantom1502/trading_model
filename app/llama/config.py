"""
config.py — Config cho model đọc giá chuyên biệt (price-only decoder)
==============================================================================
Ghi đè hoàn toàn default cũ (nhánh tiếng Việt của app/llama chưa dùng thật,
xem quyết định trong lịch sử chat). Model này KHÔNG có bất kỳ thành phần
ngôn ngữ tự nhiên nào — vocab đóng, chỉ gồm price token + marker cấu trúc.

QUYẾT ĐỊNH ĐÃ CHỐT (không tự ý đổi nếu không có lý do mới):
    - Vocab đóng ~4102 token: <unk>=0, <bos>=1, <eos>=2, <pad>=3 (GIỮ NGUYÊN
      id này vì build_model() hiện tại đã hardcode đúng thứ tự), <chart>=4,
      </chart>=5, rồi 4096 price token (O/H/L/C x 1024 bin) từ id 6.
    - Price token dùng format bracket "<px_O_512>" (đã chốt trước đó) —
      PHẢI add qua add_tokens() vì '<'/'>' là ký tự tách từ, không thể để
      pre-tokenizer tự nhận diện nguyên khối.
    - 1 document = ĐÚNG 100 nến cố định (không group/nối nhiều doc lại) —
      max_position_embeddings tính theo 100*4 + 2 = 402, có buffer.
    - KV cache bật cho Monte Carlo rollout — mặc định của LlamaForCausalLM
      (use_cache=True), không cần cấu hình thêm ở đây.
    - Vocab padding: 4102 -> làm tròn lên bội số 64 = 4160, CHỈ để tối ưu
      tensor-core alignment cho lm_head/embedding (tied weights), KHÔNG có
      lý do nào khác để pad thêm (xem app/memlm — cùng nguyên tắc).

Kích thước model (hidden/layers/heads) là ƯỚC LƯỢNG BAN ĐẦU dựa trên vocab
hẹp + task hẹp (chỉ học động lực giá, không phải ngôn ngữ tự nhiên rộng) —
CẦN điều chỉnh lại theo lượng dữ liệu thật khi đã có số liệu (nhiều dữ liệu
hơn có thể tăng, ít hơn nên giảm để tránh overfit vô nghĩa).
"""

from dataclasses import dataclass, field
from typing import Optional


# ══════════════════════════════════════════════════════════════════════
# Vocab — tính sẵn hằng số dùng chung giữa config.py và tokenizer builder
# ══════════════════════════════════════════════════════════════════════

N_PRICE_BINS       = 1024                          # bin mỗi kênh O/H/L/C
SPECIAL_TOKENS     = ["<unk>", "<bos>", "<eos>", "<pad>"]   # id 0-3, GIỮ NGUYÊN thứ tự
STRUCTURE_MARKERS  = ["<chart>", "</chart>"]                # id 4-5
PRICE_LETTERS      = ["O", "H", "L", "C"]

REAL_VOCAB_SIZE = len(SPECIAL_TOKENS) + len(STRUCTURE_MARKERS) + len(PRICE_LETTERS) * N_PRICE_BINS
# = 4 + 2 + 4096 = 4102


def _pad_vocab_size(n: int, multiple: int = 64) -> int:
    """Làm tròn lên bội số `multiple` — CHỈ để alignment tensor-core, không
    phải lý do nào khác (xem app/memlm — cùng nguyên tắc đã chốt)."""
    return multiple * ((n + multiple - 1) // multiple)


PADDED_VOCAB_SIZE = _pad_vocab_size(REAL_VOCAB_SIZE)   # 4160


# ══════════════════════════════════════════════════════════════════════
# Số nến / độ dài chuỗi — tính sẵn để không hardcode rải rác
# ══════════════════════════════════════════════════════════════════════

CANDLES_PER_DOC   = 100                                  # 1 doc = đúng 100 nến, KHÔNG group nhiều doc
TOKENS_PER_CANDLE = 4                                     # O H L C
DOC_LEN_TOKENS    = CANDLES_PER_DOC * TOKENS_PER_CANDLE + 2   # +2 cho <chart>/</chart> = 402


@dataclass
class TokenizerConfig:
    """Cấu hình tokenizer — vocab đóng, không có base BPE."""
    pretrained_name: str = "custom_tokenizer_price"
    use_fast        : bool = True


@dataclass
class ModelConfig:
    """
    Kiến trúc model — ƯỚC LƯỢNG BAN ĐẦU cho task hẹp (chỉ đọc giá).
    So với config LLaMA-VN cũ (hidden=576, layers=30, vocab=24000):
    nhỏ hơn nhiều vì vocab hẹp hơn ~6 lần và task không cần hiểu ngôn ngữ
    tự nhiên rộng — chỉ cần đủ sâu để nắm bắt candlestick pattern + cấu
    trúc thị trường (Swept/FVG/Shift) trong cửa sổ ngắn.
    """
    vocab_size              : int  = PADDED_VOCAB_SIZE          # 4160 (real=4102, xem docstring)
    hidden_size              : int  = 256
    intermediate_size         : int  = 704                        # SwiGLU-style: 64*ceil(8/3*256/64)
    num_hidden_layers          : int  = 12
    num_attention_heads         : int  = 8
    num_key_value_heads           : int  = 4                        # GQA 2:1
    max_position_embeddings         : int  = 512                      # buffer so với DOC_LEN_TOKENS=402
    tie_word_embeddings                : bool = True


@dataclass
class TrainConfig:
    """
    Cấu hình quá trình train. Khác bản tiếng Việt cũ (không cần shard-state
    tracking/multi-file Drive loop) — data đều 100 nến/doc, dùng thẳng HF
    Trainer + resume_from_checkpoint chuẩn, không cần cơ chế riêng.
    """
    device                     : str   = "cuda"
    output_dir                 : str   = "./checkpoints_price"

    per_device_train_batch_size: int   = 128
    per_device_eval_batch_size : int   = 128
    gradient_accumulation_steps: int   = 8

    learning_rate              : float = 3e-4
    weight_decay               : float = 0.1
    warmup_ratio               : float = 0.03
    num_train_epochs           : int   = 1

    logging_steps              : int   = 50
    eval_steps                 : int   = 200
    save_steps                 : int   = 200
    save_total_limit           : int   = 3

    resume_from_checkpoint     : Optional[str] = None
    fp16                       : bool  = True
    gradient_checkpointing     : bool  = True


@dataclass
class DataConfig:
    """Cấu hình dữ liệu — 1 doc = 1 row parquet, KHÔNG group/nối nhiều doc
    (khác hẳn pipeline group_texts của nhánh tiếng Việt cũ — xem ghi chú
    trong train.py: group_texts nối-cắt theo block_size sẽ PHÁ vỡ ranh
    giới doc 100-nến cố định, KHÔNG được dùng lại nguyên xi cho nhánh này)."""
    train_parquet_path : str = ""
    val_parquet_path   : str = ""
    text_col           : str = "text"


@dataclass
class HubConfig:
    """Cấu hình huggingface_hub."""
    repo_id : Optional[str] = None
    hf_token: Optional[str] = None


@dataclass
class Config:
    """Gộp tất cả config con."""
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    model    : ModelConfig     = field(default_factory=ModelConfig)
    train    : TrainConfig     = field(default_factory=TrainConfig)
    data     : DataConfig      = field(default_factory=DataConfig)
    hub      : HubConfig       = field(default_factory=HubConfig)
    seed     : int             = 42