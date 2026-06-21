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
    # source: "wikipedia" hoặc "vtsnlp"
    #   "wikipedia" — wikimedia/wikipedia, raw, dùng dataset_name/dataset_subset
    #   "vtsnlp"    — VTSNLP/vietnamese_curated_dataset, đã curate, 12.2M rows,
    #                 chất lượng tốt hơn, có field domain để lọc (xem
    #                 dataset.py::ChunkedVTSNLPLoader để biết danh sách domain)
    source         : str = "wikipedia"

    dataset_name   : str = "wikimedia/wikipedia"
    dataset_subset : str = "20231101.vi"

    # Chỉ áp dụng khi source="vtsnlp". None = lấy tất cả 25 domain.
    # Ví dụ: ["Science", "Books_and_Literature"]
    vtsnlp_domains  : list = None

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
    # Nếu train ngắn hơn 1 cycle: lr decay dần như cosine thường, không vấn đề.
    # Nếu train dài hơn nhiều cycle: lr lặp lại nhịp tăng/giảm — giúp tránh
    # forget vì lr không bao giờ "đứng yên ở mức cao" mãi mãi.
    lr_decay_cycle_steps : int = 5000
    lr_min_ratio          : float = 0.1   # lr thấp nhất = 0.1 * lr (không về 0 tuyệt đối)

    log_every            : int = 100
    eval_every            : int = 500
    save_every            : int = 1000

    save_dir              : str = "./checkpoints"
    resume_from            : Optional[str] = None   # path checkpoint để resume

    device                 : str = "cuda"            # tự detect trong code
    mixed_precision         : bool = True

    # M (memory) có giữ nguyên xuyên suốt các chunk hay reset mỗi chunk
    # True  = đúng ý tưởng "M tích lũy xuyên suốt"
    # False = M reset mỗi document mới (an toàn hơn, dễ train hơn)
    persist_memory_across_chunks: bool = True
    reset_memory_per_document    : bool = True   # reset M khi bắt đầu document mới trong cùng chunk

    # GHI CHÚ: bptt_window đã bỏ — model dùng EMA write với alpha CỐ ĐỊNH
    # (xem model/block.py: half_life trong ModelConfig), nên detach M ngay
    # mỗi batch là an toàn, không cần BPTT window phức tạp.


@dataclass
class TokenizerConfig:
    """Cấu hình tokenizer."""
    # PhoBERT tokenizer — pretrained BPE cho tiếng Việt, vocab ~64k
    # Yêu cầu: pip install transformers
    # Lưu ý: PhoBERT tokenizer cần input đã qua word-segmentation (VnCoreNLP)
    #        để đạt hiệu quả tốt nhất, nhưng vẫn chạy được ở chế độ raw text.
    #
    # ĐỂ DÙNG TOKENIZER ĐÃ MỞ RỘNG VOCAB QUA add_tokens()
    # (scripts/add_custom_tokens.py — cách cũ, mở rộng trực tiếp vocab
    # PhoBERT): đổi pretrained_name thành đường dẫn local tới thư mục
    # output của script đó:
    #
    #     cfg.tokenizer.pretrained_name = "custom_tokenizer"
    #
    # AutoTokenizer.from_pretrained() tự nhận diện đây là local path
    # (không phải tên HuggingFace) nếu thư mục tồn tại — không cần đổi gì
    # thêm ở chỗ khác trong code.
    #
    # ĐỂ DÙNG PRICE TOKEN RIÊNG BIỆT (cách mới — KHÔNG qua add_tokens(),
    # token giá nằm trong dải ID riêng ngoài vocab PhoBERT gốc, xem
    # tokenizer.py::VietnameseTokenizer để biết chi tiết thiết kế):
    # KHÔNG cần đổi pretrained_name — giữ nguyên "vinai/phobert-base" hoặc
    # tokenizer custom khác, price vocab tự động được cộng thêm vào.
    pretrained_name : str = "vinai/phobert-base"

    # use_fast KHÔNG có tác dụng với PhoBERT — đã verify
    # TOKENIZER_MAPPING_NAMES['phobert'] chỉ map tới PhobertTokenizer
    # (không có bản Fast). Giữ False để rõ ràng, tránh hiểu lầm True sẽ
    # nhanh hơn.
    use_fast        : bool = False

    # strict_chart_mode: nếu True, price token (O_x/H_x/L_x/C_x) CHỈ được
    # nhận diện khi nằm trong cặp <chart>...</chart>; mọi chuỗi trùng
    # pattern NẰM NGOÀI cặp marker được coi là text thường.
    #
    # BẬT (True) nếu train LẪN dữ liệu trading với Wikipedia/VTSNLP —
    # tránh match nhầm các ký hiệu khoa học tự nhiên trong corpus chung
    # (ví dụ "H_0" = hằng số Hubble, "C_1, C_2" = hằng số tích phân,
    # "O_157" = chủng vi khuẩn — đã verify các case này bị match nhầm khi
    # strict_chart_mode=False).
    #
    # TẮT (False, mặc định) nếu CHỈ train trên dữ liệu trading thuần —
    # rủi ro match nhầm gần như không đáng kể trong domain này.
    strict_chart_mode: bool = False

    # Số bin giá cho mỗi loại O/H/L/C. Tổng price token = n_price_bins*4 + 2
    # (4 loại O/H/L/C + 2 marker <chart>/</chart>). Mặc định 1024 bin/loại
    # → 4098 token, khớp với danh sách 4098 token bạn đã add trước đó qua
    # add_tokens(). Đổi giá trị này làm vocab_size đổi theo — phải train
    # lại từ đầu nếu đổi giữa các round (giống mọi thay đổi vocab khác).
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
