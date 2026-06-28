"""
model/block.py — Transformer Block (LLaMA-style + Context Memory)
==================================================================
Block chỉ chịu trách nhiệm:
    1. Pre-Norm + Self-Attention (RoPE, causal)
    2. Gọi MemoryLayer với x_post (đã causal → không leak)
    3. FFN (SwiGLU, Pre-Norm)
    4. Quản lý vòng đời self.memory (init / reset / detach)

Logic read/write/gate/EMA nằm hoàn toàn trong MemoryLayer (memory.py).

Luồng:
    x_normed = RMSNorm(x)
    Q        = Wq(x_normed)
    attn_out = self_attn(Q, freqs_cis, causal_mask)
    x_post   = x + attn_out                          ← không leak

    if memory:
        gated_out, memory_new = mem_layer(x_post, memory)
        x_new  = x_post + gated_out
        memory = memory_new.detach()
    else:
        x_new = x_post

    out = x_new + FFN(RMSNorm(x_new))
"""

import math
import torch
import torch.nn as nn

from .attention import SelfAttentionRoPE
from .layers import RMSNorm, SwiGLU
from .memory import MemoryLayer


class MemoryBlock(nn.Module):
    def __init__(
        self,
        d_model    : int,
        n_heads    : int,
        num_slots  : int  = 4,
        dropout    : float = 0.1,
        use_memory : bool  = True,
        half_life  : int  = 100,
        n_layers   : int  = 8,
    ):
        super().__init__()
        self.use_memory = use_memory
        self.d_model    = d_model
        self.num_slots  = num_slots

        # ── Self-Attention ────────────────────────────────────────────────────
        self.norm1     = RMSNorm(d_model)
        self.Wq        = nn.Linear(d_model, d_model, bias=False)
        self.self_attn = SelfAttentionRoPE(d_model, n_heads, dropout=dropout)

        # ── Memory Layer (tách hoàn toàn) ─────────────────────────────────────
        if use_memory:
            self.mem_layer = MemoryLayer(
                d_model   = d_model,
                n_heads   = n_heads,
                num_slots = num_slots,
                half_life = half_life,
                n_layers  = n_layers,
            )

        # ── FFN ───────────────────────────────────────────────────────────────
        self.norm2 = RMSNorm(d_model)
        self.ffn   = SwiGLU(d_model, bias=False)

        self.memory: torch.Tensor | None = None

        self._scaled_init(n_layers)

    def _scaled_init(self, n_layers: int):
        scale = 1.0 / math.sqrt(2 * n_layers)
        nn.init.normal_(self.self_attn.Wo.weight, std=0.02 * scale)
        nn.init.normal_(self.ffn.w2.weight,       std=0.02 * scale)
        # MemoryLayer tự _scaled_init trong __init__ của nó

    # ── Memory management ─────────────────────────────────────────────────────
    def init_memory(self, device):
        """Gọi một lần khi bắt đầu, không phụ thuộc batch size."""
        self.memory = torch.zeros(
            1, self.cfg.num_slots, self.cfg.d_model,
            device=device
        )

    def reset_memory(self, batch_size: int, device: torch.device):
        if self.use_memory and self.memory is None:
            self.init_memory(device)

    def reset_memory_rows(self, mask: torch.Tensor, device: torch.device):
        """No-op — memory persist theo document."""
        pass

    def detach_memory(self):
        if self.memory is not None:
            self.memory = self.memory.detach()

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        x        : torch.Tensor,
        freqs_cis: torch.Tensor,
        attn_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        B, T, D = x.shape

        # 1. Self-Attention (causal)
        x_normed = self.norm1(x)
        Q        = self.Wq(x_normed)
        attn_out = self.self_attn(Q, freqs_cis, attn_mask=attn_mask)
        x_post   = x + attn_out   # (B, T, D) — token t chỉ thấy t-1, t-2,...

        # 2. Memory (nếu có)
        if self.use_memory and self.memory is not None:
            B = x_post.size(0)
            # Expand theo batch, nhưng memory gốc vẫn là (1, S, D)
            mem_expanded = self.memory.expand(B, -1, -1)  # không copy data

            gated_out, memory_new = self.mem_layer(x_post, mem_expanded)
            x_new = x_post + gated_out
            
            # Gộp lại về (1, S, D) — average across batch
            self.memory = memory_new.detach().mean(dim=0, keepdim=True)
        else:
            x_new = x_post

        # 3. FFN
        out = x_new + self.ffn(self.norm2(x_new))
        return out