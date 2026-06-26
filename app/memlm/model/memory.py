"""
model/memory.py — Context Memory Layer (Read / Write / Gate)
=============================================================
MemoryLayer là một module độc lập, nhận x_post và memory hiện tại,
trả về (gated_output, memory_new).

Thiết kế:
    - Write : memory tra vấn x_post (đã qua causal self-attn → không leak)
    - EMA   : memory_new = alpha * memory + (1-alpha) * delta
    - Read  : x_post tra vấn memory_NEW (còn trong graph → gradient chảy qua write)
    - Gate  : sigmoid(Linear([x_post, m_out])) — 1 lớp, đủ cho nhị phân

Norm tách riêng:
    norm_m  : normalize memory     trước khi làm K/V cho read
    norm_wm : normalize memory     trước khi làm Q   cho write
    norm_wx : normalize x_post     trước khi làm K/V cho write

Gradient path (đầy đủ, khép kín trong 1 forward pass):
    loss → gated_out → m_out → read(Q=x_post, K/V=memory_new)
                                    └─ memory_new → delta → write_attn ✓
                                    └─ x_post còn trong graph ✓
    loss → gated_out → gate → gate_proj([x_post, m_out]) ✓

    Tất cả: read, write_attn, gate_proj đều nhận gradient từ loss.

Caller (block.py) chịu trách nhiệm:
    - Truyền vào memory đã detach (hoặc None nếu chưa khởi tạo)
    - Lưu memory_new.detach() vào self.memory sau mỗi forward
"""

import math
import torch
import torch.nn as nn

from .attention import CrossAttention
from .layers import RMSNorm


class MemoryLayer(nn.Module):
    def __init__(
        self,
        d_model   : int,
        n_heads   : int,
        num_slots : int,
        half_life : int,
        n_layers  : int,
    ):
        super().__init__()
        self.d_model   = d_model
        self.num_slots = num_slots
        self.alpha     = 0.5 ** (1.0 / half_life)

        # Norm tách riêng cho từng vai trò
        self.norm_m  = RMSNorm(d_model)   # memory     → K/V của read
        self.norm_wm = RMSNorm(d_model)   # memory     → Q   của write
        self.norm_wx = RMSNorm(d_model)   # x_post     → K/V của write

        self.write_attn = CrossAttention(d_model, n_heads)
        self.read       = CrossAttention(d_model, n_heads)

        # Gate: 1 lớp Linear + Sigmoid — đủ cho quyết định nhị phân
        self.gate_proj = nn.Linear(d_model * 2, d_model, bias=False)

        self._scaled_init(n_layers)

    def _scaled_init(self, n_layers: int):
        scale = 1.0 / math.sqrt(2 * n_layers)
        nn.init.normal_(self.read.Wo.weight,       std=0.02 * scale)
        nn.init.normal_(self.write_attn.Wo.weight, std=0.02 * scale)
        nn.init.normal_(self.gate_proj.weight,     std=0.02)

    def forward(
        self,
        x_post : torch.Tensor,   # (B, T, D) — đã qua causal self-attn, không leak
        memory : torch.Tensor,   # (B, num_slots, D) — đã detach từ batch trước
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            gated_out  : (B, T, D)         — cộng vào residual stream ở block
            memory_new : (B, num_slots, D) — caller lưu lại sau .detach()
        """
        # ── WRITE: memory tra vấn x_post (đã causal) → EMA ──────────────────
        # x_post an toàn (không leak) → write tham gia gradient bình thường
        delta      = self.write_attn(
            Q = self.norm_wm(memory),    # (B, num_slots, D)
            K = self.norm_wx(x_post),    # (B, T, D)
            V = self.norm_wx(x_post),
        )                                # (B, num_slots, D)

        memory_new = self.alpha * memory + (1 - self.alpha) * delta   # còn trong graph

        # ── READ: x_post tra vấn memory_NEW → gradient chảy qua write ────────
        m_normed = self.norm_m(memory_new)                             # (B, num_slots, D)
        m_out    = self.read(Q=x_post, K=m_normed, V=m_normed)        # (B, T, D)

        # ── GATE: học từ [x_post, m_out] ─────────────────────────────────────
        gate      = torch.sigmoid(
            self.gate_proj(torch.cat([x_post, m_out], dim=-1))
        )                                                               # (B, T, D)
        gated_out = gate * m_out                                        # (B, T, D)

        return gated_out, memory_new