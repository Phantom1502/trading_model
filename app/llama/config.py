"""
app/llama/config.py — Config cho nhánh pretrain dùng HF LlamaForCausalLM + Trainer chuẩn
============================================================================================
So với bản trước (custom training loop tự viết), bản này chuyển hẳn sang
transformers.Trainer + TrainingArguments — TrainConfig bên dưới vì vậy có
thêm các field ánh xạ 1-1 sang TrainingArguments (xem train.py: build_training_args()).

vocab_size trong LlamaConfig chỉ là PLACEHOLDER — bị ghi đè bởi
model.build_model(cfg, tokenizer) ngay sau khi tokenizer thật được build
(xem tokenizer.py: build_llama_tokenizer). KHÔNG tự sửa tay giá trị này khi
đổi price bin count — sửa cfg.tokenizer.n_price_bins là đủ.

LƯU Ý QUAN TRỌNG — vì sao có max_steps thay vì num_train_epochs:
Dataset nạp theo kiểu streaming (datasets.IterableDataset) để KHÔNG load hết
file parquet (>1GB) vào RAM. IterableDataset không có __len__, nên Trainer
không tự tính được "1 epoch = bao nhiêu step" → BẮT BUỘC khai rõ
cfg.train.max_steps thay vì dựa vào num_train_epochs.
"""

from dataclasses import dataclass, field
from typing import Optional
from transformers import LlamaConfig


@dataclass
class DataConfig:
    # source: "wikipedia" | "vtsnlp" | "parquet" | "mix"
    source           : str  = "wikipedia"

    dataset_name     : str  = "wikimedia/wikipedia"
    dataset_subset   : str  = "20231101.vi"
    vtsnlp_domains   : Optional[list] = None

    # Local parquet (dùng cho nguồn "parquet", vd data trading chart/action/books)
    parquet_path      : str = ""
    parquet_text_col  : str = "text"

    # Mix config — chỉ dùng khi source="mix"
    mix_sources   : dict = field(default_factory=dict)   # {name: (glob_pattern, prob)}
    mix_stopping  : str  = "first_exhausted"              # | "all_exhausted"

    min_text_len  : int   = 200

    # Số document đầu tiên của stream dùng làm tập validation (streaming
    # split — không thể chia theo tỉ lệ % vì không biết trước tổng số mẫu).
    n_val_samples : int = 2_000

    # Block size để "pack" nhiều document liên tiếp thành 1 sequence độ dài
    # cố định (group_texts) — tận dụng tối đa context window, không lãng phí
    # padding. Mặc định = max_position_embeddings của model.
    block_size    : int = 1024

    # Buffer cho streaming shuffle xấp xỉ (Dataset.shuffle(buffer_size=...))
    # — không shuffle được toàn bộ vì dữ liệu không nằm hết trong RAM.
    shuffle_buffer_size : int = 10_000

    # Batch size khi .map() tokenize/group — batch càng lớn thì packing càng
    # ít lãng phí phần dư cuối mỗi batch, nhưng tốn RAM tạm thời hơn.
    map_batch_size : int = 1_000


@dataclass
class TrainConfig:
    # ── Ánh xạ thẳng sang TrainingArguments ─────────────────────────────────
    per_device_train_batch_size : int   = 16
    per_device_eval_batch_size  : int   = 16
    gradient_accumulation_steps : int   = 32
    learning_rate                : float = 3e-4
    weight_decay                  : float = 0.01
    max_grad_norm                  : float = 1.0
    warmup_ratio                     : float = 0.03

    lr_scheduler_type : str = "cosine"   # dùng scheduler có sẵn của Trainer

    max_steps : int = 100_000   # BẮT BUỘC vì train_dataset là IterableDataset

    logging_steps : int = 100
    eval_steps    : int = 500
    save_steps    : int = 1000
    save_total_limit : int = 3

    output_dir  : str            = "./checkpoints_llama"
    resume_from : Optional[str]  = None   # checkpoint-N/ (do Trainer tự lưu) hoặc None

    gradient_checkpointing : bool = True   # tiết kiệm VRAM, đánh đổi ~20% tốc độ
    fp16 : bool = False
    bf16 : bool = True   # ưu tiên bf16 nếu GPU hỗ trợ (Ampere+); đổi fp16=True nếu chạy T4

    # 0 là BẮT BUỘC nếu muốn resume dataset streaming đúng vị trí (xem
    # DatasetStateCallback trong train.py) — num_workers>0 khiến PyTorch fork
    # tiến trình con giữ bản sao dataset riêng, tiến trình chính không thấy
    # được vị trí thật đã đọc tới đâu.
    dataloader_num_workers : int = 2

    hf_repo_id : Optional[str] = None
    hf_token   : Optional[str] = None
    push_to_hub : bool = False

    # ── Benchmark callback (run_llama_benchmark) ────────────────────────────
    run_benchmark_every_eval : bool = True
    n_language_samples       : int = 5
    best_benchmark_dir       : str = "./checkpoints_llama/best_benchmark"


@dataclass
class TokenizerConfig:
    """
    base_tokenizer_dir : tokenizer BPE gốc, train bằng app/memlm/scripts/train_tokenizer.py
    output_dir         : nơi lưu bản đã add_tokens() price vocab THẬT
                         (khác app/memlm/tokenizer.py — không cộng ID ảo bên
                         ngoài tokenizer thật nữa)
    n_price_bins       : số bin mỗi kênh O/H/L/C (mặc định 1024, khớp ChartCodec)
    """
    base_tokenizer_dir : str = "custom_tokenizer"
    output_dir          : str = "custom_tokenizer_llama"
    n_price_bins         : int = 1024


@dataclass
class Config:
    llama     : Optional[LlamaConfig] = None
    data      : DataConfig            = field(default_factory=DataConfig)
    train     : TrainConfig           = field(default_factory=TrainConfig)
    tokenizer : TokenizerConfig       = field(default_factory=TokenizerConfig)
    seed      : int = 42

    def __post_init__(self):
        if self.llama is None:
            self.llama = get_110m_llama_config()
        # block_size mặc định khớp context window của model, tránh lệch tay
        if self.data.block_size is None:
            self.data.block_size = self.llama.max_position_embeddings


def get_110m_llama_config() -> LlamaConfig:
    """
    ~110-125M params (cỡ SmolLM2-135M), phù hợp Colab T4 / Kaggle P100.
    vocab_size=24000 là placeholder — sẽ bị build_model() resize đúng theo
    tokenizer thật (base BPE 16k + price vocab ~4098 + special tokens).
    """
    return LlamaConfig(
        vocab_size             = 24_000,
        hidden_size             = 576,
        intermediate_size        = 1536,
        num_hidden_layers         = 22,
        num_attention_heads        = 9,
        num_key_value_heads         = 3,     # GQA: 9 Q head / 3 KV group
        max_position_embeddings      = 2048,
        rms_norm_eps                  = 1e-5,
        rope_theta                     = 10000.0,
        tie_word_embeddings              = True,
        use_cache                         = False,   # tắt khi train (đi cùng gradient_checkpointing)
        attention_bias                     = False,
        mlp_bias                            = False,
    )


def get_small_llama_config() -> LlamaConfig:
    """Config nhỏ để test nhanh pipeline (không dùng để train thật)."""
    return LlamaConfig(
        vocab_size             = 24_000,
        hidden_size              = 64,
        intermediate_size         = 128,
        num_hidden_layers          = 2,
        num_attention_heads         = 2,
        num_key_value_heads           = 1,
        max_position_embeddings        = 128,
        tie_word_embeddings              = True,
        use_cache                         = False,
    )


def get_default_config() -> Config:
    return Config()
