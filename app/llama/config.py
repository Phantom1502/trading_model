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
    grad_accum_steps: int = 64           # effective batch = 16 * 64 = 1024

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
    # QUAN TRỌNG: khi load_best_model_at_end=True (hardcode trong
    # trainer_utils.build_training_args), TrainingArguments BẮT BUỘC
    # save_steps là bội số của eval_steps, nếu không raise ValueError ngay
    # khi khởi tạo. Bản nháp gốc để save_steps=50/eval_steps=100 (không phải
    # bội số) — bug có sẵn từ nháp, chỉ chưa lộ vì bản transformers cũ hơn
    # validate lỏng hơn. Giữ cả 2 = 50 để chắc chắn tương thích; xem thêm
    # validate_config() ở config.py — sẽ báo lỗi sớm nếu sau này đổi 2 số
    # này mà quên giữ đúng quan hệ bội số.
    eval_steps: int = 50
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


def validate_config(cfg: Config) -> None:
    """
    Kiểm tra sớm các lỗi cấu hình PHỔ BIẾN nhất trước khi tốn thời gian
    build model/tokenizer/dataset — raise ValueError với thông báo cụ thể
    thay vì để lỗi xuất hiện mù mờ giữa chừng training (hoặc tệ hơn: không
    lỗi gì cả mà chạy sai âm thầm, như trường hợp total_tokens_estimate lệch
    thực tế khiến cosine LR decay sai tốc độ).

    Gọi hàm này ngay đầu train.py::main(), TRƯỚC hub_login()/load_tokenizer().
    """
    import glob
    import os

    errors = []

    # ── Data paths ───────────────────────────────────────────────────────
    if not cfg.data.train_shard_dir or not os.path.isdir(cfg.data.train_shard_dir):
        errors.append(f"data.train_shard_dir không tồn tại: {cfg.data.train_shard_dir!r}")
    elif not glob.glob(f"{cfg.data.train_shard_dir}/*.parquet"):
        errors.append(f"Không tìm thấy file .parquet nào trong data.train_shard_dir: {cfg.data.train_shard_dir!r}")

    if not cfg.data.val_parquet_glob or not glob.glob(cfg.data.val_parquet_glob):
        errors.append(f"Không tìm thấy file nào khớp data.val_parquet_glob: {cfg.data.val_parquet_glob!r}")

    # ── block_size phải khớp kiến trúc model, nếu không context bị cắt sai ─
    if cfg.data.block_size != cfg.model.max_position_embeddings:
        errors.append(
            f"data.block_size ({cfg.data.block_size}) != "
            f"model.max_position_embeddings ({cfg.model.max_position_embeddings}) "
            f"— 2 giá trị này phải khớp nhau, nếu không model sẽ không tận dụng "
            f"hết context hoặc lỗi khi seq dài hơn max_position_embeddings."
        )

    # ── save_steps phải là bội số của eval_steps (ràng buộc cứng của
    # TrainingArguments khi load_best_model_at_end=True — luôn True trong
    # trainer_utils.build_training_args). Bug thật đã gặp: nháp gốc để
    # save_steps=50/eval_steps=100, không phải bội số -> crash ngay khi tạo
    # TrainingArguments với ValueError khó đoán nếu không biết trước ràng buộc này.
    if cfg.train.eval_steps <= 0 or cfg.train.save_steps % cfg.train.eval_steps != 0:
        errors.append(
            f"train.save_steps ({cfg.train.save_steps}) phải là BỘI SỐ của "
            f"train.eval_steps ({cfg.train.eval_steps}) — bắt buộc vì "
            f"load_best_model_at_end=True. Ví dụ hợp lệ: eval_steps=50, save_steps=50/100/150..."
        )

    # ── Hub: push_to_hub=True nhưng thiếu repo_id -> TrainingArguments sẽ lỗi
    # ngay khi Trainer khởi tạo, thà báo sớm ở đây còn hơn để crash lúc đó.
    if cfg.hub.push_to_hub and not cfg.hub.repo_id:
        errors.append(
            "hub.push_to_hub=True nhưng hub.repo_id trống — set cfg.hub.repo_id "
            "thành 'username/repo-name', hoặc set cfg.hub.push_to_hub=False nếu "
            "chưa muốn dùng Hub."
        )

    if errors:
        msg = "Config chưa sẵn sàng, cần sửa trước khi chạy:\n  - " + "\n  - ".join(errors)
        raise ValueError(msg)

    print("✓ Config hợp lệ — đã kiểm tra data path, block_size, hub.repo_id.")


def print_config_summary(cfg: Config) -> None:
    """In ra các giá trị THỰC TẾ sẽ được dùng (sau khi resolve_paths) — để
    nhìn 1 lần biết chắc mình đang trỏ đúng folder/repo nào, không cần nhớ
    field nào derive từ field nào."""
    print("── Config summary ──────────────────────────────────────────")
    print(f"  drive_root         : {cfg.data.drive_root}")
    print(f"  train_shard_dir    : {cfg.data.train_shard_dir}")
    print(f"  val_parquet_glob   : {cfg.data.val_parquet_glob}")
    print(f"  cache_dir          : {cfg.data.cache_dir}")
    print(f"  val_cache_dir      : {cfg.data.val_cache_dir}")
    print(f"  block_size         : {cfg.data.block_size}")
    print(f"  output_dir         : {cfg.train.output_dir}")
    print(f"  state_path         : {cfg.train.state_path}")
    print(f"  per_device_batch   : {cfg.train.per_device_train_batch}  "
          f"x grad_accum {cfg.train.grad_accum_steps} = "
          f"effective batch {cfg.train.per_device_train_batch * cfg.train.grad_accum_steps}")
    print(f"  total_tokens_est.  : {cfg.train.total_tokens_estimate:,}")
    print(f"  hub.repo_id        : {cfg.hub.repo_id}")
    print(f"  hub.push_to_hub    : {cfg.hub.push_to_hub}")
    print("────────────────────────────────────────────────────────────")