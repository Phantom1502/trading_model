"""
model/block.py — Transformer Block tích hợp Context Memory (LLaMA-style)
=============================================================================
Áp dụng đầy đủ 4 kỹ thuật cốt lõi + scaled init đã thống nhất:

    1. RMSNorm + Pre-Norm   thay LayerNorm/Post-Norm
    2. SwiGLU FFN           thay Linear-GELU-Linear
    3. No bias              trên toàn bộ Linear layer
    4. RoPE                 cho self-attention (vị trí tương đối)
    5. Scaled init          theo 1/sqrt(2*n_layers) cho các projection residual

Luồng (giữ nguyên logic Read/Write đã thống nhất trước đó):

    Q = Wq(RMSNorm(x))                            trích đặc trưng hiện tại
    attn_out = SelfAttentionRoPE(Q)                attention + RoPE trong sequence

    PHA READ  : Hiện tại hỏi Quá khứ
        m_out = CrossAttn(Q=Q, K=Memory, V=Memory)        → (B, T, D)

    PHA WRITE : Quá khứ tra vấn Hiện tại (Memory làm Query)
        Q_tinh_che = CrossAttn(Q=Memory, K=Q, V=Q)         → (B, num_slots, D)

    x_new = x + attn_out + m_out
    out   = x_new + SwiGLU(RMSNorm(x_new))

    Memory_new = alpha * Memory_old + (1-alpha) * Q_tinh_che   (EMA, alpha cố định)

alpha cố định, detach_memory() an toàn mỗi step (đã verify thực nghiệm).
Memory khởi tạo bằng nhiễu nhỏ (std=0.02), KHÔNG dùng zeros tuyệt đối —
zeros làm Wq/Wk của write_attn không bao giờ nhận gradient (đã verify).
"""

import math
import torch
import torch.nn as nn

from .attention import SelfAttentionRoPE, CrossAttention
from .layers import RMSNorm, SwiGLU, precompute_freqs_cis


def alpha_from_half_life(half_life: int) -> float:
    """Tính alpha (decay rate) từ half-life mong muốn (số step)."""
    return 0.5 ** (1.0 / half_life)


class MemoryBlock(nn.Module):
    def __init__(
        self,
        d_model     : int,
        n_heads     : int,
        num_slots   : int = 4,      # số slot bộ nhớ — khuyên dùng 4 hoặc 8
        dropout     : float = 0.1,
        use_memory  : bool = True,
        half_life   : int = 100,    # nhớ tương đương ~100 token cũ
        n_layers    : int = 8,      # tổng số layer của model — dùng cho scaled init
    ):
        super().__init__()
        self.d_model    = d_model
        self.num_slots  = num_slots
        self.use_memory = use_memory

        # alpha CỐ ĐỊNH — an toàn để detach mỗi step
        self.alpha = alpha_from_half_life(half_life)

        # ── Mạch chính: Pre-Norm + RoPE Self-Attention ──────────────────────
        self.norm1     = RMSNorm(d_model)
        self.Wq        = nn.Linear(d_model, d_model, bias=False)
        self.self_attn = SelfAttentionRoPE(d_model, n_heads, dropout=dropout)

        if use_memory:
            # CỔNG 1: READ — hiện tại hỏi quá khứ
            self.read   = CrossAttention(d_model, n_heads)
            self.norm_m = RMSNorm(d_model)

            # CỔNG 2: WRITE — quá khứ tra vấn hiện tại (Memory làm Query)
            self.write_attn = CrossAttention(d_model, n_heads)
            self.norm_w     = RMSNorm(d_model)

        # ── Mạch reasoning: Pre-Norm + SwiGLU ────────────────────────────────
        self.norm2 = RMSNorm(d_model)
        self.ffn   = SwiGLU(d_model, bias=False)

        # Trạng thái bộ nhớ — runtime state, không phải parameter
        self.memory: torch.Tensor | None = None

        self._scaled_init(n_layers)

    def _scaled_init(self, n_layers: int):
        """
        Scaled init: các projection nằm trên đường residual (Wo của attention,
        w2 của SwiGLU) được scale theo 1/sqrt(2*n_layers).

        Lý do: mỗi layer cộng residual vào x, qua n_layers tầng thì variance
        của activation tăng tuyến tính theo n_layers nếu không scale. Init nhỏ
        dần theo số tầng giữ variance ổn định, tránh NaN khi xếp nhiều layer.
        """
        scale = 1.0 / math.sqrt(2 * n_layers)

        nn.init.normal_(self.self_attn.Wo.weight, std=0.02 * scale)
        nn.init.normal_(self.ffn.w2.weight, std=0.02 * scale)

        if self.use_memory:
            nn.init.normal_(self.read.Wo.weight, std=0.02 * scale)
            nn.init.normal_(self.write_attn.Wo.weight, std=0.02 * scale)

    def init_memory(self, batch_size: int, device: torch.device):
        """
        Khởi tạo bộ nhớ khi vào document/chunk mới.

        Dùng nhiễu nhỏ (std=0.02) thay vì zeros tuyệt đối — zeros làm
        write_attn.Wq(memory)=0 (Linear không bias), khiến Wq/Wk của
        write_attn không bao giờ nhận gradient (verify thực nghiệm).
        """
        if self.use_memory:
            self.memory = torch.zeros(
                batch_size, self.num_slots, self.d_model, device=device
            )
            nn.init.normal_(self.memory, std=0.02)

    def reset_memory(self, batch_size: int, device: torch.device):
        self.init_memory(batch_size, device)

    def detach_memory(self):
        """Chặt đồ thị tính toán nối về step trước. Gọi ở cuối loop train chính."""
        if self.memory is not None:
            self.memory = self.memory.detach()

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        """
        x        : (B, T, D)
        freqs_cis: (max_seq, d_head/2) complex — bảng RoPE tính sẵn từ model
        Returns  : (B, T, D)
        """
        B, T, D = x.shape

        # 1. Trích xuất đặc trưng hiện tại (Pre-Norm)
        x_normed = self.norm1(x)
        Q = self.Wq(x_normed)                                           # (B, T, D)

        # 2. Self-Attention + RoPE trong sequence hiện tại
        attn_out = self.self_attn(Q, freqs_cis, attn_mask=attn_mask)    # (B, T, D)

        if self.use_memory and self.memory is not None:
            # ── PHA READ: hiện tại tra vấn bộ nhớ cũ ──
            m_out = self.norm_m(self.read(Q=Q, K=self.memory, V=self.memory))

            x_new = x + attn_out + m_out

            # ── PHA WRITE: quá khứ tra vấn hiện tại (Memory làm Query) ──
            Q_tinh_che = self.norm_w(self.write_attn(Q=self.memory, K=Q, V=Q))

            # Cập nhật EMA — alpha cố định
            self.memory = self.alpha * self.memory + (1 - self.alpha) * Q_tinh_che
        else:
            x_new = x + attn_out

        # 5. SwiGLU FFN — reasoning (Pre-Norm)
        out = x_new + self.ffn(self.norm2(x_new))
        return out
