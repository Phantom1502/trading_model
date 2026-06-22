"""
trainer/base.py — Logic chung cho mọi loại training (pretrain/SFT/DPO)
==========================================================================
Class này chứa:
    - optimizer/scheduler setup
    - gradient accumulation
    - evaluate loop
Các subclass (PretrainTrainer, SFTTrainer, DPOTrainer) override compute_loss().

════════════════════════════════════════════════════════════════════════════
THAY ĐỔI so với phiên bản cũ:

  [FIX 2] Truncated BPTT — detach_memory() KHÔNG còn gọi sau mỗi batch.
      Thay vào đó trainer đếm số batch kể từ lần detach cuối, và chỉ
      detach sau mỗi `bptt_window` batch HOẶC khi gặp cuối document.

      Ý nghĩa: gradient được phép chảy ngược qua bptt_window batch liên
      tiếp, cho phép write_attn học "memory ghi ở window này có giúp ích
      cho window sau không". Nếu bptt_window=1 thì hành vi giống phiên bản
      cũ (detach mỗi batch).

      Cấu hình: cfg.train.bptt_window (int, mặc định 4)
          bptt_window=1  → detach mỗi batch (hành vi cũ, an toàn nhất về RAM)
          bptt_window=4  → gradient qua 4 window (~2048 token với seg_len=512)
          bptt_window=8  → gradient dài hơn nhưng RAM tăng gấp đôi

      LƯU Ý: bptt_window > 1 chỉ có ý nghĩa khi dùng sequential_mode=True.
      Với TokenChunkDataset (shuffle), các batch liên tiếp không thuộc cùng
      document, giữ gradient qua đó vô nghĩa — nên để bptt_window=1.

  [FIX 4] reset_memory_rows thay reset_memory khi is_doc_start:
      Phiên bản cũ reset toàn bộ batch khi ANY sample là doc_start.
      Phiên bản mới chỉ reset đúng các sample có is_doc_start=True.

════════════════════════════════════════════════════════════════════════════
GHI CHÚ về Memory + Gradient (cập nhật):

Model dùng EMA write với alpha CỐ ĐỊNH (xem model/block.py):
    M_new = alpha * M + (1-alpha) * Q_refined

Gradient chảy trong forward pass hiện tại qua đường:
    loss → read(Q, M_new) → M_new → write_attn → (W_write_q, W_write_k, W_write_v)

Với bptt_window > 1, gradient còn chảy ngược qua M của các batch trước,
cho write_attn biết "memory đã ghi có giúp ích batch tiếp theo không".
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import causal_mask
from utils import TrainLogger, log_eval, log_bench, save_checkpoint
from benchmark import run_all


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

        self.global_step    = 0
        self.chunk_idx      = 0
        self.best_val_loss  = float("inf")

        # [FIX 2] đếm số batch kể từ lần detach memory cuối
        self._batches_since_detach = 0

        self.logger = TrainLogger(log_every=cfg.train.log_every)
        self._setup_scheduler()

    @property
    def bptt_window(self) -> int:
        """Số batch giữ gradient xuyên suốt trước khi detach memory."""
        return getattr(self.cfg.train, "bptt_window", 4)

    def _setup_scheduler(self):
        """
        Warmup tuyến tính rồi cosine decay theo chu kỳ giả định
        (lr_decay_cycle_steps), lặp lại "warm restart" khi hết chu kỳ.
        """
        warmup      = self.cfg.train.warmup_steps
        cycle_steps = self.cfg.train.lr_decay_cycle_steps
        min_ratio   = self.cfg.train.lr_min_ratio

        def lr_lambda(step):
            if step < warmup:
                return step / max(warmup, 1)
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

    def _maybe_detach_memory(self, force: bool = False):
        """
        [FIX 2] Detach memory theo bptt_window.

        Detach khi:
            - force=True  : cuối document, ranh giới tự nhiên
            - _batches_since_detach đã đủ bptt_window

        Không detach khi đang ở giữa window — gradient giữ xuyên suốt.
        """
        self._batches_since_detach += 1
        if force or self._batches_since_detach >= self.bptt_window:
            self.model.detach_memory()
            self._batches_since_detach = 0

    def train_one_batch(self, batch: dict, accum_step: int) -> float:
        """
        Forward + backward cho MỘT batch.

        Thay đổi so với phiên bản cũ:
            - [FIX 4] reset_memory_rows(mask) chỉ reset sample is_doc_start=True
            - [FIX 2] detach theo bptt_window thay vì mỗi batch
        """
        ids          = batch["input_ids"].to(self.device)
        labels       = batch["labels"].to(self.device)
        is_doc_start = batch["is_doc_start"].to(self.device)   # (B,) bool
        is_doc_end   = batch["is_doc_end"].to(self.device)     # (B,) bool

        B, T = ids.shape

        # Khởi tạo memory nếu chưa có (đầu training)
        if not self.model.has_memory_initialized(batch_size=B):
            self.model.reset_memory(B, self.device)
            self._batches_since_detach = 0

        # [FIX 4] Chỉ reset memory của sample nào là doc_start thật sự
        elif self.cfg.train.reset_memory_per_document and is_doc_start.any():
            self.model.reset_memory_rows(is_doc_start, self.device)

        mask = causal_mask(T, self.device)

        with torch.amp.autocast("cuda", enabled=(self.device.type == "cuda" and self.cfg.train.mixed_precision)):
            logits = self.model(ids, attn_mask=mask)
            loss   = self.compute_loss(logits, labels)

        scaled_loss = loss / self.cfg.train.grad_accum
        self.scaler.scale(scaled_loss).backward()

        # [FIX 2] Detach theo bptt_window — force detach khi kết thúc document
        force_detach = bool(is_doc_end.any())
        self._maybe_detach_memory(force=force_detach)

        if (accum_step + 1) % self.cfg.train.grad_accum == 0:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
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
            is_doc_start = batch["is_doc_start"].to(self.device)
            B, T         = ids.shape

            if not self.model.has_memory_initialized(batch_size=B):
                self.model.reset_memory(B, self.device)
            elif is_doc_start.any():
                self.model.reset_memory_rows(is_doc_start, self.device)

            mask   = causal_mask(T, self.device)
            logits = self.model(ids, attn_mask=mask)
            loss   = self.compute_loss(logits, labels)

            total += loss.item()
            n     += 1
            self.model.detach_memory()   # eval luôn detach mỗi batch

        self.model.train()
        return total / max(n, 1)

    def train_one_chunk(self, train_loader, val_loader, chunk_idx: int):
        """Train trên một chunk data."""
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

                    bench = run_all(self.model, self.tokenizer, self.cfg, verbose=False, step=self.global_step)
                    log_bench(bench, step=self.global_step)

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