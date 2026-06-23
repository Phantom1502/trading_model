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


def run_pretrain(cfg, model, tokenizer, data_loader_gen=None, start_chunk: int = 0,
                  reset_lr_for_new_round: bool = False):
    """
    Entry point chạy toàn bộ pretraining.

    Args:
        cfg            : Config object
        model          : MemoryLM
        tokenizer      : VietnameseTokenizer
        data_loader_gen: ChunkedWikiLoader đã khởi tạo sẵn (optional).
        start_chunk    : chunk muốn bắt đầu/tiếp tục train (0 = từ đầu).
        reset_lr_for_new_round: QUAN TRỌNG khi train round mới với lr/lr_decay_cycle_steps
                         khác round trước.

    ────────────────────────────────────────────────────────────────────────
    HuggingFace auto-upload (cfg.train.hf_repo_id):

    Sau mỗi chunk, checkpoint chunk_{idx}.pt sẽ được upload lên HF với tên
    cố định "last_chunk.pt" — overwrite để tiết kiệm storage. Khi resume,
    tải file này về rồi đặt cfg.train.resume_from trỏ vào, tự check chunk_idx
    từ nội dung checkpoint để biết tiếp tục từ chunk nào.

    Để bật:
        cfg.train.hf_repo_id = "username/memlm-checkpoints"

    Auth (1 trong 3 cách):
        1. export HF_TOKEN=hf_xxx...          # Colab: Secrets hoặc os.environ
        2. huggingface-cli login              # dev local
        3. cfg.train.hf_token = "hf_xxx..."  # tường minh (không khuyến nghị hardcode)

    Nếu cfg.train.hf_repo_id bỏ trống hoặc upload lỗi mạng, training tiếp
    tục bình thường — hf_upload_latest() không raise exception.
    ────────────────────────────────────────────────────────────────────────
    BUG ĐÃ PHÁT HIỆN VÀ FIX (verify bằng thực nghiệm):

    Mặc định, optimizer.load_state_dict() và scheduler.load_state_dict()
    PHỤC HỒI NGUYÊN giá trị lr đã lưu trong checkpoint — kể cả khi bạn đặt
    cfg.train.lr khác trong round mới. Cụ thể: scheduler lưu base_lrs (giá
    trị lr tại thời điểm save), load lại sẽ ghi đè lên mọi config mới.

    Ví dụ bug: train round 1 với lr=1e-3, lưu checkpoint ở step 5000.
    Round 2 đặt cfg.train.lr=1e-5 (muốn lr nhỏ hơn), gọi resume_from
    checkpoint đó — nếu KHÔNG có reset_lr_for_new_round=True, optimizer
    THỰC TẾ vẫn chạy với lr≈1e-3 (giá trị cũ), không phải 1e-5 như ý muốn.

    Cách dùng đúng:

        cfg = get_100m_config()
        cfg.train.resume_from = "checkpoints/chunk_33.pt"
        cfg.train.lr = 1e-4                     # lr MỚI cho round này
        cfg.train.lr_decay_cycle_steps = 2000   # chu kỳ MỚI

        main(cfg, start_chunk=34, reset_lr_for_new_round=True)
        #                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^
        #    BẮT BUỘC True nếu muốn lr/cycle MỚI có hiệu lực thật sự

    Nếu reset_lr_for_new_round=False (mặc định) — optimizer/scheduler giữ
    NGUYÊN trạng thái cũ, tức là round mới tiếp tục lr ở đúng điểm round
    trước dừng lại (đúng ý nghĩa "resume" thông thường, KHÔNG đổi gì).
    ────────────────────────────────────────────────────────────────────────
    """
    trainer = PretrainTrainer(cfg, model, tokenizer)

    # ── Resume optimizer/scheduler state từ checkpoint (nếu có) ──────────────
    if cfg.train.resume_from:
        state = load_checkpoint(
            cfg.train.resume_from, trainer.model,
            trainer.optimizer, trainer.scheduler, trainer.device,
        )
        trainer.global_step   = state["global_step"]
        trainer.best_val_loss = state["val_loss"] or float("inf")
        if start_chunk == 0:
            start_chunk = state["chunk_idx"]
        print(f"Resuming optimizer/scheduler từ step {trainer.global_step}, "
              f"sẽ bắt đầu data từ chunk {start_chunk}")

        if reset_lr_for_new_round:
            # Ghi đè lr theo cfg MỚI — không giữ giá trị cũ từ checkpoint.
            # Tạo lại optimizer hoàn toàn mới (giữ nguyên model weight vừa
            # load) để tránh mọi trạng thái Adam (momentum, variance) bị
            # "kế thừa" theo lr cũ một cách không tường minh.
            print(f"reset_lr_for_new_round=True: tạo lại optimizer với "
                  f"lr={cfg.train.lr}, lr_decay_cycle_steps={cfg.train.lr_decay_cycle_steps}")
            trainer.optimizer = torch.optim.AdamW(
                trainer.model.parameters(),
                lr=cfg.train.lr,
                weight_decay=cfg.train.weight_decay,
                betas=(0.9, 0.95),
            )
            trainer._setup_scheduler()   # tạo lại scheduler với cfg.train mới
            # global_step KHÔNG reset — warmup/cycle vẫn tính tiếp từ vị trí
            # hiện tại trong lr_lambda(step), tránh warmup lại từ 0 đột ngột
            for _ in range(trainer.global_step):
                trainer.scheduler.step()

    # ── Tạo data loader với start_chunk (skip thật sự ở tầng dataset) ────────
    if data_loader_gen is None:
        data_loader_gen = ChunkedWikiLoader(cfg, tokenizer, start_chunk=start_chunk)

    # ── HF upload config ──────────────────────────────────────────────────────
    hf_repo_id = getattr(cfg.train, "hf_repo_id", None)
    hf_token   = getattr(cfg.train, "hf_token",   None)   # None = đọc từ env HF_TOKEN

    # ── Train qua từng chunk — KHÔNG còn cần `continue` để skip ──────────────
    for train_loader, val_loader in data_loader_gen:
        chunk_idx = data_loader_gen.chunk_count   # số chunk thực tế (đã tính cả start_chunk)

        print(f"\n{'='*60}")
        print(f"CHUNK {chunk_idx}")
        print(f"{'='*60}")

        val_loss = trainer.train_one_chunk(train_loader, val_loader, chunk_idx)

        # Lưu checkpoint sau mỗi chunk — quan trọng vì RAM thấp,
        # có thể cần restart kernel giữa các chunk
        chunk_path = f"{cfg.train.save_dir}/chunk_{chunk_idx}.pt"
        save_checkpoint(
            chunk_path,
            trainer.model, trainer.optimizer, trainer.scheduler,
            trainer.global_step, chunk_idx, val_loss,
            model_cfg=cfg.model,
        )

        # Upload lên HuggingFace (nếu hf_repo_id được đặt)
        if hf_repo_id:
            hf_upload_latest(chunk_path, repo_id=hf_repo_id, token=hf_token)

    print("\n✓ Pretraining hoàn tất toàn bộ dataset.")
    return trainer