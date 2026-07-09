"""
train.py — Vòng lặp chính, pretrain LLaMA tiếng Việt nhiều-shard (app/llama)
================================================================================
Bản refactor từ file nháp gốc (1 file ~300 dòng gộp hết) — logic KHÔNG ĐỔI,
chỉ tách theo trách nhiệm ra các module riêng:

    config.py         — Config/ModelConfig/DataConfig/TrainConfig/HubConfig
    tokenizer.py       — load_tokenizer, sync_vocab_size, check_tokenizer_matches_model_config
    model.py            — build_model (LlamaForCausalLM, random init)
    dataset.py            — tokenize_function/group_texts, shard + val dataset (parquet local)
    state.py                — shard_index / resume state (JSON trên Drive)
    hub_utils.py              — login/push/resume qua Hugging Face Hub (backup phụ)
    trainer_utils.py            — TrainingArguments + optimizer/scheduler (1 lần cho toàn bộ 14B token)

File này giờ CHỈ CÒN vòng lặp orchestration: gọi các module trên theo đúng
thứ tự, không còn logic chi tiết nào nằm trực tiếp ở đây.

Chạy:
    from app.llama.config import get_default_config
    from app.llama.train import main
    main(get_default_config())
"""

import torch
from transformers import Trainer
from transformers.trainer_utils import get_last_checkpoint

from app.llama.config import Config, get_default_config, validate_config, print_config_summary
from app.llama.tokenizer import (
    load_tokenizer,
    sync_vocab_size,
    check_tokenizer_matches_model_config,
)
from app.llama.model import build_model, count_params
from app.llama.dataset import (
    list_shard_files,
    get_or_build_val_dataset,
    get_or_build_shard_dataset,
    clear_previous_shard_cache,
    main_process_first_if_distributed
)
from app.llama.state import (
    ensure_dirs,
    load_state,
    mark_shard_completed,
    is_fresh_shard_start,
)
from app.llama.hub_utils import hub_login, push_tokenizer_once
from app.llama.trainer_utils import (
    build_training_args,
    build_optimizer_and_scheduler,
    build_data_collator,
)


def main(cfg: Config = None):
    if cfg is None:
        cfg = get_default_config()

    print_config_summary(cfg)
    validate_config(cfg)

    ensure_dirs(cfg)
    torch.manual_seed(cfg.seed)

    # Device chỉ dùng để LOG — TrainingArguments tự phát hiện GPU/TPU/CPU
    # thật sự dùng lúc build Trainer (accelerate lo phần đó), không đọc lại
    # cfg.train.device. Nhánh hardware="tpu" không gọi torch.cuda.is_available()
    # vì không có ý nghĩa gì trên máy chỉ có TPU.
    if cfg.train.hardware == "tpu":
        device_str = "tpu"
        print(f"Hardware: TPU ({cfg.train.num_tpu_cores} core, qua run())")
    else:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {device_str}")
        if device_str == "cuda":
            print(f"GPU : {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
    cfg.train.device = device_str

    # ── Hugging Face Hub login (tùy chọn, chỉ cảnh báo nếu thiếu) ────────────
    hub_login(cfg)

    # ── Tokenizer ────────────────────────────────────────────────────────────
    tokenizer = load_tokenizer(cfg)
    sync_vocab_size(tokenizer, cfg)       # PHẢI chạy trước build_model()
    check_tokenizer_matches_model_config(tokenizer, cfg)
    print(f"Tokenizer vocab size: {cfg.model.vocab_size}")

    push_tokenizer_once(tokenizer, cfg)   # 1 lần duy nhất, cố định xuyên suốt

    # ── Val set — xử lý 1 lần, cố định xuyên suốt toàn bộ quá trình ─────────
    # ── Val set ──
    with main_process_first_if_distributed(cfg):
        lm_val = get_or_build_val_dataset(cfg, tokenizer)
    # ── ──
    data_collator = build_data_collator(tokenizer)

    # ── Model ────────────────────────────────────────────────────────────────
    model = build_model(cfg)
    params = count_params(model)
    print(f"Model params: {params['total']/1e6:.1f}M "
          f"(embedding: {params['embedding']/1e6:.1f}M, "
          f"non-embedding: {params['non_embedding']/1e6:.1f}M)")
    
    # QUAN TRỌNG: phải chuyển model sang XLA device TRƯỚC khi tạo optimizer —
    # optimizer chụp tham chiếu tensor tham số lúc khởi tạo, nếu model còn ở
    # CPU thì optimizer sẽ lệch device so với model sau khi Trainer/accelerate
    # chuyển model sang TPU, gây lỗi "not on the same device".
    if cfg.train.hardware == "tpu":
        import torch_xla.core.xla_model as xm
        model = model.to(xm.xla_device())
        print(f"Model đã chuyển sang XLA device: {xm.xla_device()}")

    # ── Optimizer + scheduler — tạo 1 LẦN cho toàn bộ hành trình nhiều-shard ─
    optimizer, scheduler, total_steps = build_optimizer_and_scheduler(model, cfg)
    print(f"Tổng step ước tính: {total_steps}")

    # ── Shard list + state (resume) ──────────────────────────────────────────
    shard_files = list_shard_files(cfg)
    print(f"Tổng số shard: {len(shard_files)}")

    state = load_state(cfg)
    training_args = build_training_args(cfg)

    # ── Vòng lặp chính — xử lý tuần tự từng shard, resume đúng theo 2 trường hợp ─
    for i in range(state["shard_index"], len(shard_files)):
        shard_path = shard_files[i]
        with main_process_first_if_distributed(cfg):
            lm_shard = get_or_build_shard_dataset(cfg, tokenizer, i, shard_path)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=lm_shard,
            eval_dataset=lm_val,
            data_collator=data_collator,
            optimizers=(optimizer, scheduler),
        )

        # --- xác định đây là "resume giữa chừng shard này" hay "bắt đầu shard mới" ---
        local_ckpt = get_last_checkpoint(cfg.train.output_dir)

        if local_ckpt is None:
            print(f"[Shard {i}] Không có checkpoint nào, train từ đầu.")
            trainer.train()
        elif is_fresh_shard_start(state, local_ckpt):
            print(f"[Shard {i}] Checkpoint thuộc về shard trước, bắt đầu shard mới (không resume).")
            trainer.train()
        else:
            print(f"[Shard {i}] Resume giữa chừng từ {local_ckpt}")
            trainer.train(resume_from_checkpoint=local_ckpt)

        # --- cập nhật state SAU KHI shard train xong hoàn toàn ---
        final_ckpt = get_last_checkpoint(cfg.train.output_dir)
        state = mark_shard_completed(cfg, i, final_ckpt)
        print(f"✅ Hoàn thành shard {i}.")

        # --- xoá cache shard TRƯỚC (không xoá shard vừa train) ---
        clear_previous_shard_cache(cfg, i)

    print("🎉 Đã train xong toàn bộ shard.")

def _tpu_main_wrapper(index, cfg):
    """torch_xla.launch tự truyền process index (0-7) làm tham số đầu tiên
    khi gọi hàm — main() chỉ nhận cfg nên cần wrapper này để bỏ qua index."""
    main(cfg)


def run(cfg: Config = None):
    """
    Entry point NÊN DÙNG thay vì gọi main() trực tiếp — đây là chỗ DUY NHẤT
    biết cách "launch" khác nhau giữa GPU/CPU (gọi main() bình thường) và
    TPU (phải fork cfg.train.num_tpu_cores process qua torch_xla.launch,
    mỗi process chạy 1 core trong 8 core của TPU v5e-8 trên Kaggle — main()
    KHÔNG tự động chạy song song nếu gọi trực tiếp trên TPU).
    """
    if cfg is None:
        cfg = get_default_config()

    if cfg.train.hardware == "tpu":
        import torch_xla
        torch_xla.launch(_tpu_main_wrapper, args=(cfg,))
    else:
        main(cfg)


if __name__ == "__main__":
    run()