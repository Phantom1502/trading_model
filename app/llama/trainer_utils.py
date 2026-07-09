"""
trainer_utils.py — TrainingArguments + optimizer/scheduler cho app/llama
================================================================================
Tách từ train.py nháp: build_training_args(), và phần tạo optimizer +
get_cosine_schedule_with_warmup(...) đang nằm trực tiếp trong main().

Cũng thêm build_data_collator() — dòng
`DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)` trong nháp
chỉ có 1 dòng nhưng nằm rải trong main(), gom vào đây cho đủ bộ "mọi thứ
Trainer cần" ở 1 chỗ.
"""

from typing import Tuple

import torch
from transformers import (
    DataCollatorForLanguageModeling,
    TrainingArguments,
    get_cosine_schedule_with_warmup,
)

from app.llama.config import Config


def build_training_args(cfg: Config) -> TrainingArguments:
    """
    Dùng CHUNG cho mọi shard (tạo 1 lần, truyền lại cho Trainer của từng shard
    trong vòng lặp main() — xem readme.md mục 5.5).

    LƯU Ý QUAN TRỌNG (giữ nguyên từ nháp): KHÔNG set `lr_scheduler_type`/
    `warmup_steps`/`learning_rate` ở đây dù TrainingArguments có các field này
    — vì optimizer/scheduler được tạo thủ công (build_optimizer_and_scheduler
    bên dưới) và truyền trực tiếp vào Trainer(optimizers=(...)), nên mọi tham
    số LR khai báo trong TrainingArguments sẽ bị Trainer BỎ QUA HOÀN TOÀN
    trong trường hợp đó (readme.md mục 2, "Về LR scheduler trong TrainingArguments").
    """
    return TrainingArguments(
        output_dir=cfg.train.output_dir,
        per_device_train_batch_size=cfg.train.per_device_train_batch,
        per_device_eval_batch_size=cfg.train.per_device_eval_batch,
        gradient_accumulation_steps=cfg.train.grad_accum_steps,
        gradient_checkpointing=cfg.train.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": cfg.train.gradient_checkpointing_use_reentrant},
        fp16=cfg.train.fp16,
        bf16=cfg.train.bf16,
        dataloader_num_workers=cfg.train.dataloader_num_workers,
        logging_steps=cfg.train.logging_steps,
        eval_strategy="steps",
        eval_steps=cfg.train.eval_steps,
        save_strategy="steps",
        save_steps=cfg.train.save_steps,
        save_total_limit=cfg.train.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        num_train_epochs=cfg.train.num_train_epochs,   # chỉ 1 lượt qua SHARD hiện tại rồi dừng
        report_to="none",
        push_to_hub=cfg.hub.push_to_hub,
        hub_model_id=cfg.hub.repo_id,
        hub_private_repo=cfg.hub.private_repo,
        hub_strategy=cfg.hub.hub_strategy,   # push full checkpoint (model+optimizer+scheduler+rng)
    )


def compute_total_steps(cfg: Config) -> Tuple[int, int]:
    """
    Trả về (total_steps, warmup_steps) suy từ total_tokens_estimate — TÁCH
    RIÊNG khỏi build_optimizer_and_scheduler() để có thể log/kiểm tra con số
    này TRƯỚC khi tạo optimizer thật (vd in ra để sanity-check trước khi
    train 2 tháng, tránh phát hiện total_steps sai sau khi đã chạy dở).
    """
    effective_batch = cfg.train.per_device_train_batch * cfg.train.grad_accum_steps
    total_steps = cfg.train.total_tokens_estimate // (cfg.data.block_size * effective_batch)
    warmup_steps = int(cfg.train.warmup_ratio * total_steps)
    return total_steps, warmup_steps


def build_optimizer_and_scheduler(model, cfg: Config):
    """
    Tạo optimizer + scheduler MỘT LẦN DUY NHẤT cho TOÀN BỘ hành trình nhiều-shard
    (readme.md mục 5.2) — PHẢI dùng chung xuyên suốt mọi shard, không tạo lại
    mỗi shard, nếu không cosine LR sẽ decay về 0 rồi reset lại ở shard kế tiếp,
    gãy đường cong learning rate.

    Trả về (optimizer, scheduler, total_steps) — total_steps trả kèm để caller
    (train.py) log ra, không cần gọi compute_total_steps() lại lần 2.
    """
    total_steps, warmup_steps = compute_total_steps(cfg)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        betas=(cfg.train.adam_beta1, cfg.train.adam_beta2),
        weight_decay=cfg.train.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    return optimizer, scheduler, total_steps


def build_data_collator(tokenizer) -> DataCollatorForLanguageModeling:
    """Causal LM: mlm=False -> labels tự sinh từ input_ids (shift bên trong
    model), không cần dataset tự tạo cột labels."""
    return DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)