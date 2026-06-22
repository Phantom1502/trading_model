"""
config.py — Toàn bộ hyperparameter của project
================================================
Sửa giá trị ở đây, không sửa rải rác trong code.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Kiến trúc model."""
    vocab_size  : int = 64001        # PhoBERT vocab size (set lại sau khi load tokenizer)
    d_model     : int = 512
    n_heads     : int = 8
    n_layers    : int = 8
    num_slots   : int = 4            # số memory slot — nhiều "ngăn nhớ" độc lập (khuyên 4 hoặc 8)
    half_life   : int = 100          # M nhớ tương đương ~100 token cũ (alpha = 0.5^(1/half_life))
    max_seq     : int = 512
    dropout     : float = 0.1
    use_memory  : bool = True        # False để train baseline so sánh


@dataclass
class DataConfig:
    """Cấu hình dữ liệu và cách load incremental."""
    # source: "wikipedia" | "vtsnlp" | "parquet"
    #   "wikipedia" — wikimedia/wikipedia, raw, dùng dataset_name/dataset_subset
    #   "vtsnlp"    — VTSNLP/vietnamese_curated_dataset, đã curate, 12.2M rows,
    #                 chất lượng tốt hơn, có field domain để lọc (xem
    #                 dataset.py::ChunkedVTSNLPLoader để biết danh sách domain)
    #   "parquet"   — file .parquet local (sách, corpus nội bộ, ...) — xem
    #                 dataset.py::ChunkedParquetLoader, cần đặt parquet_path
    source         : str = "wikipedia"

    dataset_name   : str = "wikimedia/wikipedia"
    dataset_subset : str = "20231101.vi"

    # Chỉ áp dụng khi source="vtsnlp". None = lấy tất cả 25 domain.
    # Ví dụ: ["Science", "Books_and_Literature"]
    vtsnlp_domains  : list = None

    # ── Local parquet (sách, corpus nội bộ) ─────────────────────────────
    # Chỉ áp dụng khi source="parquet".
    # parquet_path     : đường dẫn tới file .parquet (absolute hoặc relative)
    # parquet_text_col : tên cột chứa văn bản (mặc định "text")
    # Lọc theo metadata (author, genre, ...): dùng filter_fn khi khởi tạo
    # ChunkedParquetLoader trực tiếp thay vì qua main() — xem train.py.
    parquet_path     : str = ""       # ví dụ: "data/books.parquet"
    parquet_text_col : str = "text"   # đổi nếu cột text tên khác

    chunk_size     : int = 10_000     # số sample load mỗi lần (do RAM ít)
    seg_len        : int = 512        # độ dài 1 segment train (truncated BPTT)
    min_text_len   : int = 200        # bỏ qua sample quá ngắn
    val_ratio      : float = 0.01
    cache_dir      : str = "./data_cache"


@dataclass
class TrainConfig:
    """Cấu hình quá trình train."""
    batch_size        : int = 8
    grad_accum         : int = 4
    lr                 : float = 3e-4
    warmup_steps        : int = 500
    weight_decay        : float = 0.1
    max_grad_norm        : float = 1.0
    epochs_per_chunk     : int = 1     # số epoch train trên mỗi chunk data
    total_chunks         : int = -1    # -1 = train hết toàn bộ dataset

    # LR decay — KHÔNG biết trước tổng số step thật (streaming dataset),
    # nên dùng "chu kỳ giả định": coi như cứ sau `lr_decay_cycle_steps` step
    # thì lr đã decay hết cosine một vòng, rồi WARM RESTART (quay lại đỉnh,
    # decay tiếp). Đây là kỹ thuật SGDR (cosine annealing with warm restarts).
    lr_decay_cycle_steps : int = 5000
    lr_min_ratio          : float = 0.1   # lr thấp nhất = 0.1 * lr (không về 0 tuyệt đối)

    log_every            : int = 100
    eval_every            : int = 500
    save_every            : int = 1000

    save_dir              : str = "./checkpoints"
    resume_from            : Optional[str] = None   # path checkpoint để resume

    device                 : str = "cuda"            # tự detect trong code
    mixed_precision         : bool = True

    persist_memory_across_chunks: bool = True
    reset_memory_per_document    : bool = True


@dataclass
class TokenizerConfig:
    """Cấu hình tokenizer."""
    pretrained_name : str = "vinai/phobert-base"

    # use_fast KHÔNG có tác dụng với PhoBERT — đã verify
    use_fast        : bool = False

    # strict_chart_mode=True (mặc định): price token (O_x/H_x/L_x/C_x) CHỈ được
    # nhận diện khi nằm trong cặp <chart>...</chart>. An toàn khi train lẫn
    # dữ liệu trading với sách/Wikipedia/VTSNLP — tránh match nhầm các ký
    # hiệu xuất hiện tự nhiên trong corpus (C_2022 = năm, H_0 = hằng số
    # Hubble, O_157 = chủng vi khuẩn, ...).
    #
    # Tắt (False) CHỈ khi train trên dữ liệu trading THUẦN — lớp 2 bin
    # validation [0, n_price_bins-1] trong _split_segments_loose vẫn hoạt
    # động để chặn KeyError kể cả khi data lỗi.
    strict_chart_mode: bool = True   # mặc định True — an toàn khi trộn corpus

    # Số bin giá cho mỗi loại O/H/L/C. Tổng price token = n_price_bins*4 + 2
    n_price_bins     : int = 1024


@dataclass
class Config:
    """Gộp tất cả config con."""
    model     : ModelConfig     = field(default_factory=ModelConfig)
    data      : DataConfig      = field(default_factory=DataConfig)
    train     : TrainConfig     = field(default_factory=TrainConfig)
    tokenizer : TokenizerConfig = field(default_factory=TokenizerConfig)

    seed: int = 42


def get_default_config() -> Config:
    return Config()


def get_small_config() -> Config:
    """Config nhỏ để test nhanh trên máy yếu / Colab free tier."""
    cfg = Config()
    cfg.model.d_model   = 256
    cfg.model.n_heads   = 4
    cfg.model.n_layers  = 4
    cfg.model.max_seq   = 256
    cfg.data.chunk_size = 2_000
    cfg.data.seg_len    = 256
    cfg.train.batch_size = 4
    return cfg


def get_100m_config() -> Config:
    """Config ~100M params cho Colab T4."""
    cfg = Config()
    cfg.model.d_model   = 512
    cfg.model.n_heads   = 8
    cfg.model.n_layers  = 8
    cfg.model.max_seq   = 512
    cfg.data.chunk_size = 10_000
    cfg.train.batch_size = 8
    cfg.train.grad_accum = 4
    return cfg