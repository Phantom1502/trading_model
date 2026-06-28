"""
model/attention.py — Self-Attention với RoPE + Cross-Attention cho Read M
=============================================================================
SelfAttentionRoPE : thay nn.MultiheadAttention, hỗ trợ RoPE và no-bias
CrossAttention    : dùng cho Read M, không cần RoPE (M không có vị trí)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import apply_rope


# ══════════════════════════════════════════════════════════════════════════
# Self-Attention với RoPE
# ══════════════════════════════════════════════════════════════════════════

class SelfAttentionRoPE(nn.Module):
    """
    Self-attention chuẩn, có RoPE áp lên Q/K, không bias trên các Linear.
    Thay thế nn.MultiheadAttention vì PyTorch built-in không hỗ trợ RoPE.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.scale   = math.sqrt(self.d_head)

        # no bias — theo kiến trúc LLaMA-style
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)

        # Init std=0.02 chuẩn. Wo nằm trên đường residual nên sẽ bị
        # MemoryBlock._scaled_init() ghi đè SAU khi block cha khởi tạo xong.
        for layer in [self.Wq, self.Wk, self.Wv, self.Wo]:
            nn.init.normal_(layer.weight, std=0.02)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        B, T, D = x.shape
        h, dh = self.n_heads, self.d_head

        q = self.Wq(x).view(B, T, h, dh).transpose(1, 2)   # (B, h, T, dh)
        k = self.Wk(x).view(B, T, h, dh).transpose(1, 2)
        v = self.Wv(x).view(B, T, h, dh).transpose(1, 2)

        # Áp RoPE lên Q và K — KHÔNG áp lên V
        q = apply_rope(q, freqs_cis)
        k = apply_rope(k, freqs_cis)

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale   # (B, h, T, T)
        if attn_mask is not None:
            scores = scores + attn_mask   # mask dạng additive (-inf ở vị trí cấm)

        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        out     = torch.matmul(weights, v)                            # (B, h, T, dh)

        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.Wo(out)