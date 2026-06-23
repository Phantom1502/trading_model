"""
model/lm.py — MemoryLM hoàn chỉnh (LLaMA-style)
===================================================
Áp dụng đủ 4 kỹ thuật cốt lõi + scaled init:
    - RMSNorm + Pre-Norm  : norm_out dùng RMSNorm thay LayerNorm
    - SwiGLU              : trong mỗi MemoryBlock (xem block.py)
    - No bias             : lm_head và toàn bộ Linear trong block đều bias=False
    - RoPE                : KHÔNG còn pos_emb (absolute position) —
                             vị trí được mã hóa qua RoPE áp trực tiếp lên Q/K
                             trong self-attention của từng block.
    - Scaled init         : truyền n_layers xuống từng MemoryBlock để tự
                             scale init các projection trên đường residual.

════════════════════════════════════════════════════════════════════════════
THAY ĐỔI so với phiên bản cũ:

    reset_memory_rows(mask, device) — reset memory CHỈ cho các sample
    có is_doc_start=True thay vì reset toàn bộ batch [FIX 4].
    Trainer gọi hàm này thay vì reset_memory().
"""

import torch
import torch.nn as nn

from .block import MemoryBlock
from .layers import RMSNorm, precompute_freqs_cis


class MemoryLM(nn.Module):
    def __init__(
        self,
        vocab_size : int,
        d_model    : int = 512,
        n_heads    : int = 8,
        n_layers   : int = 8,
        num_slots  : int = 4,
        half_life  : int = 100,
        max_seq    : int = 512,
        dropout    : float = 0.1,
        use_memory : bool = True,
        rope_base  : float = 10000.0,
    ):
        super().__init__()
        self.d_model    = d_model
        self.n_layers   = n_layers
        self.n_heads    = n_heads
        self.use_memory = use_memory
        self.max_seq    = max_seq

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.drop      = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            MemoryBlock(
                d_model, n_heads, num_slots, dropout, use_memory,
                half_life, n_layers=n_layers,
            )
            for _ in range(n_layers)
        ])

        self.norm_out = RMSNorm(d_model)
        self.lm_head  = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight   # weight tying

        d_head = d_model // n_heads
        freqs_cis = precompute_freqs_cis(d_head, max_seq * 2, base=rope_base)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_emb.weight, std=0.02)

    def num_params(self, trainable_only: bool = False) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    # ── Memory management ────────────────────────────────────────────────

    def reset_memory(self, batch_size: int, device: torch.device):
        """Reset toàn bộ memory — gọi khi bắt đầu document mới (batch_size=1)
        hoặc khi chưa có memory nào."""
        if self.use_memory:
            for block in self.blocks:
                block.reset_memory(batch_size, device)

    def reset_memory_rows(self, mask: torch.Tensor, device: torch.device):
        """
        [FIX 4] Reset memory CHỈ cho các sample có mask=True.

        mask: (B,) bool tensor — thường là batch["is_doc_start"]
        Gọi hàm này thay vì reset_memory() để tránh xóa oan memory
        của các sample không phải doc_start trong cùng batch.
        """
        if self.use_memory:
            for block in self.blocks:
                block.reset_memory_rows(mask, device)

    def detach_memory(self):
        """Cắt gradient của memory — trainer gọi theo bptt_window."""
        if self.use_memory:
            for block in self.blocks:
                block.detach_memory()

    def has_memory_initialized(self, batch_size: int = None) -> bool:
        if not self.use_memory:
            return False
        if any(b.memory is None for b in self.blocks):
            return False
        if batch_size is not None:
            return all(b.memory.size(0) == batch_size for b in self.blocks)
        return True

    # ── Forward ──────────────────────────────────────────────────────────

    def forward(self, input_ids: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        """
        input_ids: (B, T)
        Returns: logits (B, T, vocab_size)
        """
        B, T = input_ids.shape
        device = input_ids.device

        if self.use_memory and not self.has_memory_initialized(batch_size=B):
            self.reset_memory(B, device)

        x = self.drop(self.token_emb(input_ids))

        freqs_cis = self.freqs_cis.to(device)

        for block in self.blocks:
            x = block(x, freqs_cis=freqs_cis, attn_mask=attn_mask)

        x = self.norm_out(x)
        return self.lm_head(x)

    def forward_memory_only(self, input_ids: torch.Tensor, attn_mask: torch.Tensor = None) -> None:
        """
        [FIX 5] Chạy forward CHỈ để cập nhật M — bỏ qua norm_out và lm_head.

        Dùng trong generate.py khi flush token cũ ra khỏi sliding window:
        chỉ cần M được update, không cần logit → tiết kiệm ~30% compute
        (norm_out + lm_head chiếm phần đáng kể với vocab_size lớn).

        Không trả về gì — caller chỉ quan tâm đến side-effect trên self.memory.
        """
        B, T = input_ids.shape
        device = input_ids.device

        if self.use_memory and not self.has_memory_initialized(batch_size=B):
            self.reset_memory(B, device)

        x = self.drop(self.token_emb(input_ids))

        freqs_cis = self.freqs_cis.to(device)

        for block in self.blocks:
            x = block(x, freqs_cis=freqs_cis, attn_mask=attn_mask)
        # Dừng ở đây — bỏ norm_out và lm_head


def causal_mask(T: int, device: torch.device) -> torch.Tensor:
    """Tạo causal mask cho autoregressive attention."""
    mask = torch.triu(torch.ones(T, T, device=device), diagonal=1)
    return mask.masked_fill(mask.bool(), float("-inf"))


def build_model(cfg) -> MemoryLM:
    """Entry point xây model từ ModelConfig."""
    return MemoryLM(
        vocab_size = cfg.model.vocab_size,
        d_model    = cfg.model.d_model,
        n_heads    = cfg.model.n_heads,
        n_layers   = cfg.model.n_layers,
        num_slots  = cfg.model.num_slots,
        half_life  = cfg.model.half_life,
        max_seq    = cfg.model.max_seq,
        dropout    = cfg.model.dropout,
        use_memory = cfg.model.use_memory,
        rope_base  = getattr(cfg.model, "rope_base", 10000.0),
    )