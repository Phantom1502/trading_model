"""
trainer/base.py — Logic chung cho mọi loại training (pretrain/SFT/DPO)
==========================================================================
Class này chứa:
    - optimizer/scheduler setup
    - gradient accumulation
    - evaluate loop
Các subclass (PretrainTrainer, SFTTrainer, DPOTrainer) override compute_loss().

────────────────────────────────────────────────────────────────────────────
GHI CHÚ QUAN TRỌNG về Memory + Gradient:

Model dùng EMA write với alpha CỐ ĐỊNH (xem model/block.py):
    M_new = alpha * M + (1-alpha) * Q

Đã verify bằng thực nghiệm: vì alpha không phải tham số học được,
detach_memory() NGAY SAU MỖI BATCH là an toàn — không làm mất tín hiệu
gradient cho Wq, W_self, W_mread (các trọng số này đều có đường tắt
trực tiếp tới loss trong CHÍNH batch hiện tại).

Vì vậy trainer này quay lại cấu trúc đơn giản: forward → backward → detach
mỗi batch, KHÔNG cần BPTT window phức tạp như khi alpha học được.
────────────────────────────────────────────────────────────────────────────
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import causal_mask
from utils import TrainLogger, log_eval, save_checkpoint


class BaseTrainer:
    def __init__(self, cfg, model, tokenizer):
        self.cfg       = cfg
        self.model     = model
        self.tokenizer = tokenizer

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and cfg.train.device == "cuda" else "cpu"
        )
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
            betas=(0.9, 0.95),
        )

        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=(self.device.type == "cuda" and cfg.train.mixed_precision)
        )

        self.global_step   = 0
        self.chunk_idx      = 0
        self.best_val_loss   = float("inf")

        self.logger = TrainLogger(log_every=cfg.train.log_every)
        self._setup_scheduler()

    def _setup_scheduler(self):
        """
        Warmup tuyến tính rồi cosine decay theo CHU KỲ GIẢ ĐỊNH
        (lr_decay_cycle_steps), lặp lại "warm restart" khi hết chu kỳ.

        Lý do: dataset là streaming, không biết trước total_steps thật.
        Coi mỗi `lr_decay_cycle_steps` step là một "epoch giả định":
            - warmup ở đầu mỗi cycle
            - cosine decay xuống lr_min_ratio * lr ở cuối cycle
            - rồi nhảy về đỉnh, lặp lại (SGDR — Stochastic Gradient Descent
              with Warm Restarts)

        Hiện tượng forget bạn quan sát thường do lr giữ NGUYÊN ở mức cao
        sau warmup mãi mãi — model liên tục "ghi đè mạnh" lên trọng số bằng
        gradient lớn, không bao giờ có giai đoạn lr thấp để ổn định những gì
        đã học. Decay định kỳ giải quyết đúng vấn đề này; warm restart giúp
        model không bị kẹt ở lr quá thấp mãi mãi khi train cực dài.
        """
        warmup       = self.cfg.train.warmup_steps
        cycle_steps  = self.cfg.train.lr_decay_cycle_steps
        min_ratio    = self.cfg.train.lr_min_ratio

        def lr_lambda(step):
            if step < warmup:
                return step / max(warmup, 1)

            # Vị trí trong chu kỳ hiện tại (sau khi trừ warmup ban đầu)
            step_in_cycle = (step - warmup) % cycle_steps
            progress      = step_in_cycle / max(cycle_steps, 1)

            cosine = 0.5 * (1 + math.cos(math.pi * progress))
            return min_ratio + (1 - min_ratio) * cosine

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Override ở subclass. Mặc định: cross-entropy next-token prediction."""
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
        )

    def train_one_batch(self, batch: dict, accum_step: int) -> float:
        """
        Forward + backward cho MỘT batch. Detach M ngay sau backward —
        an toàn vì alpha (EMA decay) cố định, không cần gradient xuyên-batch.
        """
        ids          = batch["input_ids"].to(self.device)
        labels       = batch["labels"].to(self.device)
        is_doc_start = batch["is_doc_start"]

        B, T = ids.shape

        # Reset M nếu đây là đoạn đầu của document mới
        if self.cfg.train.reset_memory_per_document and is_doc_start.any():
            self.model.reset_memory(B, self.device)
        elif not self.model.has_memory_initialized(batch_size=B):
            self.model.reset_memory(B, self.device)

        mask = causal_mask(T, self.device)

        with torch.amp.autocast("cuda", enabled=(self.device.type == "cuda" and self.cfg.train.mixed_precision)):
            logits = self.model(ids, attn_mask=mask)
            loss   = self.compute_loss(logits, labels)

        scaled_loss = loss / self.cfg.train.grad_accum
        self.scaler.scale(scaled_loss).backward()

        

        if (accum_step + 1) % self.cfg.train.grad_accum == 0:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            
            # Detach NGAY — an toàn vì alpha cố định (đã verify thực nghiệm)
            self.model.detach_memory()
        
            self.scheduler.step()
            self.global_step += 1

        return loss.item()

    @torch.no_grad()
    def evaluate(self, val_loader, max_batches: int = 50) -> float:
        self.model.eval()
        total, n = 0.0, 0

        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break

            ids          = batch["input_ids"].to(self.device)
            labels       = batch["labels"].to(self.device)
            is_doc_start = batch["is_doc_start"]
            B, T = ids.shape

            if is_doc_start.any() or not self.model.has_memory_initialized(batch_size=B):
                self.model.reset_memory(B, self.device)

            mask   = causal_mask(T, self.device)
            logits = self.model(ids, attn_mask=mask)
            loss   = self.compute_loss(logits, labels)

            total += loss.item()
            n     += 1
            self.model.detach_memory()

        self.model.train()
        return total / max(n, 1)

    def train_one_chunk(self, train_loader, val_loader, chunk_idx: int):
        """Train trên một chunk data (gọi từ pretrain.py)."""
        self.model.train()
        self.chunk_idx = chunk_idx

        for epoch in range(self.cfg.train.epochs_per_chunk):
            for accum_step, batch in enumerate(train_loader):
                loss = self.train_one_batch(batch, accum_step)
                self.logger.update(loss)

                if self.logger.should_log():
                    lr = self.scheduler.get_last_lr()[0]
                    self.logger.flush(step=self.global_step, lr=lr)

                if self.global_step > 0 and self.global_step % self.cfg.train.eval_every == 0:
                    val_loss = self.evaluate(val_loader)
                    log_eval(val_loss, step=self.global_step)

                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        save_checkpoint(
                            f"{self.cfg.train.save_dir}/best.pt",
                            self.model, self.optimizer, self.scheduler,
                            self.global_step, self.chunk_idx, val_loss,
                            model_cfg=self.cfg.model,
                        )

                if self.global_step > 0 and self.global_step % self.cfg.train.save_every == 0:
                    save_checkpoint(
                        f"{self.cfg.train.save_dir}/step_{self.global_step}.pt",
                        self.model, self.optimizer, self.scheduler,
                        self.global_step, self.chunk_idx,
                        model_cfg=self.cfg.model,
                    )

        val_loss = self.evaluate(val_loader)
        log_eval(val_loss, step=self.global_step, prefix="  [Chunk end] ")
        return val_loss
