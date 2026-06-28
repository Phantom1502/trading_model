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
        dropout    : float = 0.1,
        n_layers   : int  = 8,
    ):
        super().__init__()
        self.d_model    = d_model

        # ── Self-Attention ────────────────────────────────────────────────────
        self.norm1     = RMSNorm(d_model)
        self.Wq        = nn.Linear(d_model, d_model, bias=False)
        self.self_attn = SelfAttentionRoPE(d_model, n_heads, dropout=dropout)

        # ── FFN ───────────────────────────────────────────────────────────────
        self.norm2 = RMSNorm(d_model)
        self.ffn   = SwiGLU(d_model, bias=False)

        # ── Context Memory ────────────────────────────────────────────────────    
        self._scaled_init(n_layers)

    def _scaled_init(self, n_layers: int):
        scale = 1.0 / math.sqrt(2 * n_layers)
        nn.init.normal_(self.self_attn.Wo.weight, std=0.02 * scale)
        nn.init.normal_(self.ffn.w2.weight,       std=0.02 * scale)

    # ── Forward ───────────────────────────────────────────────────────────────
    def forward(
        self,
        x        : torch.Tensor,
        freqs_cis: torch.Tensor,
        attn_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        B, T, D = x.shape

        # 1. Self-Attention (causal)
        attn_out = self.self_attn(x, freqs_cis, attn_mask=attn_mask)
        x_post   = x + attn_out   # (B, T, D) — token t chỉ thấy t-1, t-2,...

        # 3. FFN
        out = x_post + self.ffn(self.norm2(x_post))
        return out