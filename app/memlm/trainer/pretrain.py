"""
trainer/pretrain.py — Pretraining trên Wikipedia tiếng Việt
================================================================
Dùng BaseTrainer với compute_loss mặc định (cross-entropy next-token).
Vòng lặp ngoài cùng: load từng chunk dữ liệu, train, giải phóng, load tiếp.

M (memory) PERSIST xuyên suốt các chunk theo đúng cấu hình
`persist_memory_across_chunks=True` — chỉ reset M tại document boundary,
không reset khi chuyển chunk dữ liệu.

────────────────────────────────────────────────────────────────────────────
RESUME TỪ CHUNK CỤ THỂ (ví dụ chunk 20 bị corrupt, muốn train tiếp từ đó):

ChunkedWikiLoader.skip() ở TẦNG DATASET (dataset.py) — KHÔNG tokenize các
article thuộc chunk đã bỏ qua. Khác với cách cũ (load + tokenize hết rồi
mới `continue` bỏ qua), cách này nhanh hơn nhiều khi resume ở chunk lớn.
────────────────────────────────────────────────────────────────────────────
"""

import torch
from .base import BaseTrainer
from dataset import ChunkedWikiLoader
from utils import save_checkpoint, load_checkpoint
from utils.checkpoint import hf_upload_latest


class PretrainTrainer(BaseTrainer):
    """Kế thừa nguyên BaseTrainer — pretrain dùng đúng cross-entropy mặc định."""
    pass


# ══════════════════════════════════════════════════════════════════════════
# PATCH 3: pretrain.py — truyền file_order vào/ra checkpoint
# Sửa hàm run_pretrain, 3 chỗ được đánh dấu NEW
# ══════════════════════════════════════════════════════════════════════════

def run_pretrain(cfg, model, tokenizer, data_loader_gen=None, start_chunk: int = 0,
                  reset_lr_for_new_round: bool = False):

    trainer = PretrainTrainer(cfg, model, tokenizer)

    # ── Resume optimizer/scheduler state ─────────────────────────────────────
    file_order = None   # NEW: sẽ đọc từ checkpoint nếu có
    if cfg.train.resume_from:
        state = load_checkpoint(
            cfg.train.resume_from, trainer.model,
            trainer.optimizer, trainer.scheduler, trainer.device,
        )
        trainer.global_step   = state["global_step"]
        trainer.best_val_loss = state["val_loss"] or float("inf")
        file_order            = state["file_order"]   # NEW — None nếu checkpoint cũ
        if start_chunk == 0:
            start_chunk = state["chunk_idx"]
        print(f"Resuming từ step {trainer.global_step}, chunk {start_chunk}")
        if file_order:
            print(f"  file_order loaded từ checkpoint ({len(file_order)} sources)")
        else:
            print(f"  file_order không có trong checkpoint → shuffle mới")

        if reset_lr_for_new_round:
            print(f"reset_lr_for_new_round=True: tạo lại optimizer lr={cfg.train.lr}")
            trainer.optimizer = torch.optim.AdamW(
                trainer.model.parameters(),
                lr=cfg.train.lr,
                weight_decay=cfg.train.weight_decay,
                betas=(0.9, 0.95),
            )
            trainer._setup_scheduler()
            for _ in range(trainer.global_step):
                trainer.scheduler.step()

    # ── Data loader ───────────────────────────────────────────────────────────
    if data_loader_gen is None:
        if cfg.data.source == "mix":
            from dataset import ChunkedMixLoader
            data_loader_gen = ChunkedMixLoader(
                cfg, tokenizer,
                start_chunk=start_chunk,
                file_order=file_order,   # NEW — None = shuffle mới, dict = resume
            )
        else:
            data_loader_gen = ChunkedWikiLoader(cfg, tokenizer, start_chunk=start_chunk)

    hf_repo_id = getattr(cfg.train, "hf_repo_id", None)
    hf_token   = getattr(cfg.train, "hf_token",   None)

    # ── Train loop ────────────────────────────────────────────────────────────
    for train_loader, val_loader in data_loader_gen:
        chunk_idx = data_loader_gen.chunk_count

        print(f"\n{'='*60}")
        print(f"CHUNK {chunk_idx}")
        print(f"{'='*60}")

        val_loss = trainer.train_one_chunk(train_loader, val_loader, chunk_idx)

        chunk_path = f"{cfg.train.save_dir}/chunk_{chunk_idx}.pt"

        # NEW: lấy file_order từ loader nếu là MixLoader
        current_file_order = (
            data_loader_gen.file_order
            if hasattr(data_loader_gen, "file_order")
            else None
        )

        save_checkpoint(
            chunk_path,
            trainer.model, trainer.optimizer, trainer.scheduler,
            trainer.global_step, chunk_idx, val_loss,
            model_cfg=cfg.model,
            file_order=current_file_order,   # NEW
        )

        if hf_repo_id:
            hf_upload_latest(chunk_path, repo_id=hf_repo_id, token=hf_token)

    print("\n✓ Pretraining hoàn tất toàn bộ dataset.")
    return trainer