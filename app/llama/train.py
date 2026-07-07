"""
app/llama/train.py — Entry point pretrain dùng transformers.Trainer chuẩn
=============================================================================
Thay toàn bộ custom training loop (LlamaPretrainTrainer/TrainLogger tự viết
ở bản trước) bằng transformers.Trainer + TrainingArguments — tận dụng thẳng
mixed precision, gradient checkpointing, gradient accumulation, LR schedule,
logging, checkpointing/resume có sẵn của HF, không tự viết tay nữa.

Usage (Colab, chạy từ trong app/llama/):

    !pip install "transformers>=4.44" datasets accelerate -q
    !python train.py

Hoặc trong notebook cell:

    from config import Config
    from train import main
    main(Config())

Resume — trỏ vào checkpoint do chính Trainer lưu (output_dir/checkpoint-N):

    cfg.train.resume_from = "checkpoints_llama/checkpoint-5000"
    main(cfg)

Đổi nguồn dữ liệu — giống hệt convention cũ:
    cfg.data.source = "wikipedia" | "vtsnlp" | "parquet" | "mix"

Riêng nguồn "parquet"/"mix" TỰ ĐỘNG convert price token cũ ("O_512 H_..")
sang định dạng mới ("<px_O_512>") trước khi tokenize — không cần sửa lại
dữ liệu đã sinh từ app/utils/chart/*.

LƯU Ý dataset streaming (IterableDataset) + resume: Trainer replay-skip batch
đã train khi resume chỉ hoạt động đáng tin cậy với map-style Dataset. Với
IterableDataset, resume sẽ tiếp tục huấn luyện từ optimizer/step state đã lưu
nhưng ĐỌC LẠI STREAM TỪ ĐẦU (không tự động seek đúng vị trí cũ) — chấp nhận
đánh đổi này để đổi lấy RAM-safe streaming trên file parquet lớn. Nếu cần
resume chính xác tuyệt đối, cân nhắc convert dataset sang shard cố định trước.
"""

import json
import os

import torch
from transformers import (
    LlamaForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from app.llama.config import Config
from app.llama.tokenizer import load_llama_tokenizer, convert_legacy_price_tokens
from app.llama.model import build_model, num_params
from app.llama.dataset import build_train_eval_datasets
from app.llama.benchmark import run_all as run_llama_benchmark


# ══════════════════════════════════════════════════════════════════════════
# Callback: lưu vị trí ĐÃ ĐỌC TRONG STREAM — thay vai trò "chunk_idx" cũ
# ══════════════════════════════════════════════════════════════════════════
#
# datasets.IterableDataset (dùng cho train_dataset ở dataset.py) hỗ trợ
# .state_dict()/.load_state_dict() — lưu đúng vị trí (shard_idx +
# shard_example_idx) đã đọc tới trong stream, KỂ CẢ sau các bước
# .map(tokenize)/.map(group_texts)/.shuffle(buffer). Callback này lưu state
# đó vào ĐÚNG bên trong mỗi checkpoint-N/ mà Trainer tự tạo (sau khi
# _save_checkpoint() đã tạo xong thư mục — xem thứ tự gọi trong trainer.py:
# _save_checkpoint() luôn chạy TRƯỚC callback_handler.on_save()).
#
# LƯU Ý về .shuffle(buffer_size=...): resume sẽ tiếp tục ĐỌC ĐÚNG VỊ TRÍ
# trong stream gốc (không đọc lại từ đầu, không lặp lại dữ liệu đã train),
# nhưng buffer xáo trộn cục bộ sẽ được NẠP LẠI TỪ ĐẦU (rỗng → đầy dần) thay
# vì khôi phục đúng nội dung cũ của buffer — đây là hành vi CHÍNH THỨC của
# thư viện `datasets` (in ra cảnh báo "shuffle buffer... will be refilled"),
# không phải bug. Nói cách khác: không mất/lặp document nào, chỉ thứ tự
# xáo trộn cục bộ ngay sau điểm resume là ngẫu nhiên mới — chấp nhận được
# và tốt hơn nhiều so với đọc lại toàn bộ từ đầu.
#
# QUAN TRỌNG — chỉ đúng khi dataloader_num_workers=0: với num_workers>0,
# PyTorch fork tiến trình con, mỗi worker giữ BẢN SAO IterableDataset riêng
# → train_dataset ở tiến trình chính (được callback này giữ tham chiếu)
# KHÔNG cập nhật theo dữ liệu mà worker con đã tiêu thụ, state lưu ra sẽ SAI.
# TrainConfig.dataloader_num_workers mặc định = 0 vì lý do này.

class DatasetStateCallback(TrainerCallback):
    def __init__(self, train_dataset):
        self.train_dataset = train_dataset

    def on_save(self, args, state, control, **kwargs):
        ckpt_dir   = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
        state_path = os.path.join(ckpt_dir, "dataset_state.json")
        try:
            with open(state_path, "w") as f:
                json.dump(self.train_dataset.state_dict(), f)
        except Exception as e:
            print(f"  ⚠ Không lưu được dataset_state.json: {e}")
        return control


def _maybe_resume_dataset_position(train_dataset, resume_from: str):
    """Nếu resume_from có kèm dataset_state.json (do DatasetStateCallback lưu
    ở lần chạy trước) → load lại đúng vị trí đã đọc trong stream. Nếu không
    có (vd checkpoint cũ, hoặc lần đầu chạy) → dataset đọc từ đầu, in cảnh báo
    rõ ràng để không ai ngỡ ngàng vì tưởng đã resume đúng mà thực ra không."""
    state_path = os.path.join(resume_from, "dataset_state.json")
    if os.path.exists(state_path):
        with open(state_path) as f:
            train_dataset.load_state_dict(json.load(f))
        print(f"  ✓ Dataset resume đúng vị trí đã đọc trong stream (từ {state_path})")
    else:
        print(
            f"  ⚠ Không thấy {state_path} — dataset sẽ ĐỌC LẠI TỪ ĐẦU stream "
            f"(document đã train ở lần chạy trước có thể được thấy lại)."
        )


# ══════════════════════════════════════════════════════════════════════════
# Callback: benchmark (semantic/entity/fact/ood/language) + lưu best riêng
# ══════════════════════════════════════════════════════════════════════════

class BenchmarkCallback(TrainerCallback):
    """Chạy run_llama_benchmark() mỗi lần Trainer evaluate (eval_steps), in
    kết quả, và lưu riêng 1 bản save_pretrained() vào best_benchmark_dir mỗi
    khi điểm benchmark tổng ('total') cải thiện — ĐỘC LẬP với cơ chế
    load_best_model_at_end (dựa trên eval_loss) mà TrainingArguments đã lo."""

    def __init__(self, cfg: Config, tokenizer):
        self.cfg          = cfg
        self.tokenizer    = tokenizer
        self.best_total   = float("-inf")

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if not self.cfg.train.run_benchmark_every_eval or model is None:
            return control

        was_training = model.training
        model.eval()
        result = run_llama_benchmark(
            model, self.tokenizer, self.cfg, verbose=False,
            step=state.global_step, n_language_samples=self.cfg.train.n_language_samples,
        )
        if was_training:
            model.train()

        print(
            f"  [Benchmark] step {state.global_step} | total={result['total']:+.3f} | "
            f"sem={result['semantic']:+.2f} ent={result['entity']:+.2f} "
            f"fact={result['fact']:+.2f} lang={result['language']:+.3f} ood={result['ood']:+.2f}"
        )

        if result["total"] > self.best_total:
            self.best_total = result["total"]
            path = self.cfg.train.best_benchmark_dir
            model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
            print(f"  ✓ New best benchmark ({result['total']:+.3f}) → {path}")

        return control


# ══════════════════════════════════════════════════════════════════════════
# TrainingArguments — ánh xạ từ TrainConfig
# ══════════════════════════════════════════════════════════════════════════

def build_training_args(cfg: Config) -> TrainingArguments:
    t = cfg.train
    return TrainingArguments(
        output_dir                  = t.output_dir,

        max_steps                     = t.max_steps,

        per_device_train_batch_size    = t.per_device_train_batch_size,
        per_device_eval_batch_size      = t.per_device_eval_batch_size,
        gradient_accumulation_steps      = t.gradient_accumulation_steps,

        learning_rate                     = t.learning_rate,
        weight_decay                       = t.weight_decay,
        max_grad_norm                       = t.max_grad_norm,
        warmup_ratio                         = t.warmup_ratio,
        lr_scheduler_type                     = t.lr_scheduler_type,

        eval_strategy                          = "steps",
        eval_steps                              = t.eval_steps,
        save_strategy                            = "steps",
        save_steps                                = t.save_steps,
        save_total_limit                           = t.save_total_limit,
        logging_steps                               = t.logging_steps,

        load_best_model_at_end                        = True,
        metric_for_best_model                        = "eval_loss",
        greater_is_better                             = False,

        gradient_checkpointing                         = t.gradient_checkpointing,
        fp16                                             = t.fp16,
        bf16                                               = t.bf16,

        dataloader_num_workers                              = t.dataloader_num_workers,
        # Dataset streaming (IterableDataset) không có __len__ → tắt hẳn việc
        # Trainer tự "replay-skip" batch cũ khi resume (xem docstring đầu file).
        ignore_data_skip                                      = True,

        report_to     = "none",
        push_to_hub    = t.push_to_hub,
        hub_model_id    = t.hf_repo_id,
        hub_token        = t.hf_token,
        seed              = cfg.seed,
    )


def main(cfg: Config = None):
    if cfg is None:
        cfg = Config()

    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU : {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
        print(f"bf16 supported: {torch.cuda.is_bf16_supported()}")
    else:
        # CPU không hỗ trợ bf16/fp16 autocast theo cùng cách — tắt để tránh lỗi
        cfg.train.bf16 = False
        cfg.train.fp16 = False

    # ── Tokenizer ──────────────────────────────────────────────────────────
    print("\n── Loading tokenizer (BPE + price token thật) ──")
    tokenizer = load_llama_tokenizer(cfg)
    print(f"Vocab size: {len(tokenizer):,}")

    # ── Model ────────────────────────────────────────────────────────────
    print("\n── Building model ──")
    if cfg.train.resume_from:
        print(f"Sẽ resume weight/optimizer/step từ checkpoint: {cfg.train.resume_from}")
        model = LlamaForCausalLM.from_pretrained(cfg.train.resume_from)
    else:
        model = build_model(cfg, tokenizer)

    print(f"Total params: {num_params(model)/1e6:.1f}M")

    # ── Dataset (streaming, RAM-safe) ───────────────────────────────────────
    print(f"\n── Dataset streaming: source={cfg.data.source} | block_size={cfg.data.block_size} ──")
    text_transform = convert_legacy_price_tokens if cfg.data.source in ("parquet", "mix") else None
    train_dataset, eval_dataset = build_train_eval_datasets(cfg, tokenizer, text_transform=text_transform)

    if cfg.train.resume_from:
        _maybe_resume_dataset_position(train_dataset, cfg.train.resume_from)

    if cfg.train.dataloader_num_workers > 0:
        print(
            "  ⚠ dataloader_num_workers > 0: vị trí resume của dataset streaming "
            "có thể KHÔNG chính xác (xem docstring DatasetStateCallback). "
            "Đặt = 0 nếu cần resume đúng tuyệt đối."
        )

    # ── Data collator — labels = input_ids, HF tự shift nội bộ khi loss ─────
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # ── Trainer ──────────────────────────────────────────────────────────
    training_args = build_training_args(cfg)
    trainer = Trainer(
        model         = model,
        args          = training_args,
        train_dataset = train_dataset,
        eval_dataset  = eval_dataset,
        data_collator = data_collator,
        callbacks     = [
            BenchmarkCallback(cfg, tokenizer),
            DatasetStateCallback(train_dataset),
        ],
    )

    # ── Train ────────────────────────────────────────────────────────────
    print("\n── Starting pretraining ──")
    trainer.train(resume_from_checkpoint=cfg.train.resume_from)

    trainer.save_model(f"{cfg.train.output_dir}/final")
    tokenizer.save_pretrained(f"{cfg.train.output_dir}/final")
    print(f"\n✓ Pretraining hoàn tất → {cfg.train.output_dir}/final")
    return trainer


if __name__ == "__main__":
    main()
