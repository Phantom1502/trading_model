"""
config.py — Cấu hình cho pipeline pretrain LLaMA tiếng Việt (app/llama)
==============================================================================
Gộp toàn bộ hằng số đang nằm rải rác dạng module-level constant trong bản
nháp train.py (DRIVE_ROOT, BLOCK_SIZE, PER_DEVICE_TRAIN_BATCH, TOTAL_TOKENS_ESTIMATE...)
vào đây, theo đúng cấu trúc dataclass đã dùng ở app/memlm/config.py.

TRẠNG THÁI: mới xong phần config. train.py CHƯA được sửa để đọc từ đây —
làm tuần tự từng phần theo yêu cầu, bước sau mới refactor train.py.

Đổi tên so với bản nháp gốc:
    - `val_cache_path` -> `val_cache_dir` (nhất quán với train_shard_dir/cache_dir,
      đều là thư mục chứa cache, không phải 1 file)
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TokenizerConfig:
    """Cấu hình tokenizer."""
    pretrained_name: str = "custom_tokenizer_llama"
    use_fast: bool = True


@dataclass
class ModelConfig:
    """Cấu hình kiến trúc model — mặc định kiểu SmolLM-135M (30 layer, hidden 576),
    đã test ổn định trên Colab T4 (xem readme.md mục 1)."""
    vocab_size: int = 24000               # bị ghi đè runtime = len(tokenizer) sau khi load
    hidden_size: int = 576
    intermediate_size: int = 1536
    num_hidden_layers: int = 30
    num_attention_heads: int = 9
    num_key_value_heads: int = 3          # GQA tỉ lệ 3:1
    max_position_embeddings: int = 1024
    tie_word_embeddings: bool = True

    # Special token id — khớp custom_tokenizer_llama hiện tại (bos=1, eos=2, pad=3).
    # Đưa vào config thay vì hardcode trong build_model() để đổi tokenizer sau
    # này chỉ cần sửa config, không đụng code.
    pad_token_id: int = 3
    bos_token_id: int = 1
    eos_token_id: int = 2

    # Đã benchmark: sdpa nhanh hơn eager ~3 lần (0.12 it/s -> 0.40 it/s), xem
    # readme.md mục 1 — KHÔNG dùng "eager".
    attn_implementation: str = "sdpa"


@dataclass
class DataConfig:
    """Cấu hình dữ liệu — pipeline nhiều-shard đọc parquet local (readme.md mục 5)."""
    drive_root: str = "/content/drive/MyDrive/llama_project"

    # Các path con mặc định suy ra từ drive_root (giữ đúng cấu trúc thư mục
    # trong nháp: train_shards/, val/, train_shards_cache/, lm_val_cache/) —
    # để None và gọi Config.resolve_paths() sau khi chỉnh drive_root, hoặc
    # set tay riêng từng path nếu muốn đặt khác chỗ.
    train_shard_dir: Optional[str] = None    # None -> f"{drive_root}/train_shards"
    val_parquet_glob: Optional[str] = None   # None -> f"{drive_root}/val/*.parquet"
    cache_dir: Optional[str] = None          # None -> f"{drive_root}/train_shards_cache"
    val_cache_dir: Optional[str] = None      # None -> f"{drive_root}/lm_val_cache"

    # = model.max_position_embeddings hiện tại, nhưng tách field riêng để có
    # thể thử block_size khác (vd rút ngắn để tăng tốc) mà không đổi kiến trúc.
    block_size: int = 1024

    map_num_proc: Optional[int] = None   # None -> os.cpu_count() lúc dùng


@dataclass
class TrainConfig:
    """Cấu hình quá trình train — mặc định lấy từ bản nháp đã test ổn định
    trên Colab T4 (readme.md mục 2 và 4)."""
    device: str = "cuda"

    output_dir: Optional[str] = None   # None -> f"{data.drive_root}/checkpoints"
    state_path: Optional[str] = None   # None -> f"{train.output_dir}/shard_state.json"

    per_device_train_batch: int = 16
    per_device_eval_batch: int = 16
    grad_accum_steps: int = 8           # effective batch = 16 * 8 = 128

    lr: float = 3e-4
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95

    # Optimizer/scheduler tạo 1 LẦN cho toàn bộ hành trình nhiều-shard (readme.md
    # mục 5.2) — cosine cần biết trước tổng số step, suy ra từ ước lượng tổng
    # token thay vì đặt tay total_steps (dễ quên cập nhật khi đổi batch/block_size).
    total_tokens_estimate: int = 14_000_000_000   # 14 file x ~1B token
    warmup_ratio: float = 0.03

    fp16: bool = True
    gradient_checkpointing: bool = True
    # Bắt buộc False, không dùng mặc định True của PyTorch — xem readme.md mục 2.
    gradient_checkpointing_use_reentrant: bool = False

    # QUAN TRỌNG: field này PHẢI nằm ở TrainConfig, không phải DataConfig — bug
    # thật đã gặp: gán nhầm cfg.data.dataloader_num_workers không có tác dụng gì,
    # Trainer vẫn đọc cfg.train.dataloader_num_workers (mặc định cũ = 2).
    # Giữ 0 cho IterableDataset nhiều-shard để resume đúng vị trí (xem docstring
    # train.py nháp, mục "Resume phân biệt 2 case").
    dataloader_num_workers: int = 0

    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 50              # ~40 phút/lần ở tốc độ hiện tại trên T4
    save_total_limit: int = 3

    num_train_epochs: int = 1   # chỉ 1 lượt / shard hiện tại rồi dừng (vòng lặp nhiều-shard)


@dataclass
class HubConfig:
    """Cấu hình huggingface_hub — backup checkpoint + tokenizer (readme.md mục 6)."""
    repo_id: Optional[str] = None
    hf_token: Optional[str] = None
    private_repo: bool = True
    push_to_hub: bool = True
    hub_strategy: str = "checkpoint"   # push full checkpoint (model+optimizer+scheduler+rng)


@dataclass
class Config:
    """Gộp tất cả config con."""
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    hub: HubConfig = field(default_factory=HubConfig)
    seed: int = 42

    def resolve_paths(self) -> None:
        """Điền các path còn None bằng giá trị suy ra từ drive_root/output_dir.

        Gọi 1 LẦN sau khi cfg đã được chỉnh (vd đổi drive_root) — KHÔNG tự động
        chạy trong __post_init__, vì dataclass không tự re-resolve khi field
        khác (drive_root) đổi sau đó; gọi ngầm dễ gây bug "đổi drive_root mà
        path con không đổi theo" nếu resolve chạy quá sớm.
        """
        root = self.data.drive_root
        if self.data.train_shard_dir is None:
            self.data.train_shard_dir = f"{root}/train_shards"
        if self.data.val_parquet_glob is None:
            self.data.val_parquet_glob = f"{root}/val/*.parquet"
        if self.data.cache_dir is None:
            self.data.cache_dir = f"{root}/train_shards_cache"
        if self.data.val_cache_dir is None:
            self.data.val_cache_dir = f"{root}/lm_val_cache"

        if self.train.output_dir is None:
            self.train.output_dir = f"{root}/checkpoints"
        if self.train.state_path is None:
            self.train.state_path = f"{self.train.output_dir}/shard_state.json"


def get_default_config() -> Config:
    """Config mặc định — khớp 1:1 giá trị hardcode trong bản nháp train.py hiện
    tại (SmolLM-135M style, Colab T4, 14B token). Nhớ gọi lại cfg.resolve_paths()
    nếu chỉnh drive_root/output_dir sau khi lấy cfg từ hàm này."""
    cfg = Config()
    cfg.resolve_paths()
    return cfg