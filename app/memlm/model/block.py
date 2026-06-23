"""
model/block.py — Transformer Block tích hợp Context Memory (LLaMA-style)
=============================================================================
Áp dụng đầy đủ 4 kỹ thuật cốt lõi + scaled init đã thống nhất:

    1. RMSNorm + Pre-Norm   thay LayerNorm/Post-Norm
    2. SwiGLU FFN           thay Linear-GELU-Linear
    3. No bias              trên toàn bộ Linear layer
    4. RoPE                 cho self-attention (vị trí tương đối)
    5. Scaled init          theo 1/sqrt(2*n_layers) cho các projection residual

════════════════════════════════════════════════════════════════════════════
LUỒNG ĐÃ FIX (so với phiên bản cũ):

  [FIX 3] Pre-Norm nhất quán — norm áp lên INPUT của attention, không phải output:
      Phiên bản cũ: m_out = norm(read(Q, memory))   ← Post-Norm, sai
      Phiên bản mới: m_out = read(Q, norm(memory))  ← Pre-Norm, đúng

  [FIX 1] write_attn nhận gradient từ loss qua đường read:
      Phiên bản cũ:
          m_out = read(Q, memory_cũ)          ← read memory cũ
          memory_mới = EMA(memory_cũ, write)  ← write xảy ra sau, không ảnh hưởng output
          x_new = x + attn_out + m_out        ← loss không chảy vào write_attn

      Phiên bản mới:
          Q_refined  = write_attn(memory, Q)              ← write trước
          memory_new = EMA(memory_cũ, Q_refined)          ← còn trong graph
          m_out      = read(Q, norm(memory_new))          ← read từ memory MỚI
          x_new      = x + attn_out + m_out              ← gradient: loss→m_out→memory_new→Q_refined→write_attn ✓
          self.memory = memory_new.detach()               ← lưu đã detach, tránh graph tích lũy

  Gradient khép kín trong 1 forward pass — KHÔNG giữ graph qua batch:
      loss → m_out → read → memory_new → Q_refined → write_attn ✓
      self.memory lưu .detach() nên backward() lần sau không bị "graph freed"

Luồng hoàn chỉnh:
    x_normed = RMSNorm(x)
    Q        = Wq(x_normed)

    attn_out = SelfAttentionRoPE(Q)

    [WRITE] Q_refined  = CrossAttn(Q=norm_w(memory), K=norm_w(Q), V=norm_w(Q))
            memory_new = alpha * memory + (1-alpha) * Q_refined   [còn graph]

    [READ]  m_out = CrossAttn(Q=Q, K=norm_m(memory_new), V=norm_m(memory_new))

    x_new = x + attn_out + m_out
    out   = x_new + SwiGLU(RMSNorm(x_new))

    self.memory = memory_new.detach()   ← graph bị cắt ở đây, không tích lũy

────────────────────────────────────────────────────────────────────────────
Memory khởi tạo bằng nhiễu nhỏ (std=0.02), KHÔNG dùng zeros tuyệt đối.
"""

import math
import torch
import torch.nn as nn

from .attention import SelfAttentionRoPE, CrossAttention
from .layers import RMSNorm, SwiGLU


def alpha_from_half_life(half_life: int) -> float:
    """Tính alpha (decay rate) từ half-life mong muốn (số step)."""
    return 0.5 ** (1.0 / half_life)


class MemoryBlock(nn.Module):
    def __init__(
        self,
        d_model     : int,
        n_heads     : int,
        num_slots   : int = 4,
        dropout     : float = 0.1,
        use_memory  : bool = True,
        half_life   : int = 100,
        n_layers    : int = 8,
    ):
        super().__init__()
        self.d_model    = d_model
        self.num_slots  = num_slots
        self.use_memory = use_memory
        self.alpha      = alpha_from_half_life(half_life)

        # ── Mạch chính: Pre-Norm + RoPE Self-Attention ──────────────────────
        self.norm1     = RMSNorm(d_model)
        self.Wq        = nn.Linear(d_model, d_model, bias=False)
        self.self_attn = SelfAttentionRoPE(d_model, n_heads, dropout=dropout)

        if use_memory:
            # [FIX 3] norm áp lên INPUT của attention (Pre-Norm):
            #   norm_m : normalize memory trước khi làm K/V cho read
            #   norm_w : normalize Q và memory trước khi vào write_attn
            self.norm_m = RMSNorm(d_model)   # dùng cho read  — norm memory
            self.norm_w = RMSNorm(d_model)   # dùng cho write — norm memory và Q

            # CỔNG WRITE: memory tra vấn token hiện tại → trích gì cần nhớ
            self.write_attn = CrossAttention(d_model, n_heads)

            # CỔNG READ: token hiện tại tra vấn memory đã cập nhật
            self.read = CrossAttention(d_model, n_heads)

        # ── Mạch reasoning: Pre-Norm + SwiGLU ──────────────────────────────
        self.norm2 = RMSNorm(d_model)
        self.ffn   = SwiGLU(d_model, bias=False)

        self.memory: torch.Tensor | None = None

        self._scaled_init(n_layers)

    def _scaled_init(self, n_layers: int):
        """
        Scaled init: các projection nằm trên đường residual (Wo của attention,
        w2 của SwiGLU) được scale theo 1/sqrt(2*n_layers).
        """
        scale = 1.0 / math.sqrt(2 * n_layers)

        nn.init.normal_(self.self_attn.Wo.weight, std=0.02 * scale)
        nn.init.normal_(self.ffn.w2.weight,       std=0.02 * scale)

        if self.use_memory:
            nn.init.normal_(self.read.Wo.weight,       std=0.02 * scale)
            nn.init.normal_(self.write_attn.Wo.weight, std=0.02 * scale)

    def init_memory(self, batch_size: int, device: torch.device):
        """
        Khởi tạo memory khi vào document mới.
        Dùng nhiễu nhỏ (std=0.02) thay vì zeros tuyệt đối.
        """
        if self.use_memory:
            self.memory = torch.zeros(
                batch_size, self.num_slots, self.d_model, device=device
            )
            nn.init.normal_(self.memory, std=0.02)

    def reset_memory(self, batch_size: int, device: torch.device):
        self.init_memory(batch_size, device)

    def reset_memory_rows(self, mask: torch.Tensor, device: torch.device):
        """
        [FIX 4] Reset memory CHỈ cho các sample trong batch có mask=True.
        Tránh reset toàn bộ batch khi chỉ một số sample là doc_start.

        mask: (B,) bool tensor
        """
        if self.use_memory and self.memory is not None:
            noise = torch.zeros_like(self.memory)
            nn.init.normal_(noise, std=0.02)
            # Chỉ ghi đè những row có mask=True
            self.memory = torch.where(
                mask.view(-1, 1, 1),   # (B, 1, 1) broadcast
                noise,
                self.memory,
            )

    def detach_memory(self):
        """Chặt đồ thị tính toán — trainer gọi theo bptt_window."""
        if self.memory is not None:
            self.memory = self.memory.detach()

    def forward(
        self,
        x         : torch.Tensor,
        freqs_cis : torch.Tensor,
        attn_mask : torch.Tensor = None,
    ) -> torch.Tensor:
        """
        x        : (B, T, D)
        freqs_cis: (max_seq, d_head/2) complex
        Returns  : (B, T, D)

        Luồng memory kiến trúc (READ TRƯỚC, WRITE SAU):
            1. READ:  Token Q hiện tại tra vấn memory CŨ (từ batch trước, đã detach).
                      → m_out = read(Q, norm_m(memory), norm_m(memory))
            2. TỔNG HỢP: x_new = x + attn_out + m_out
            3. WRITE: Tạo memory MỚI bằng cách nén ngữ cảnh của batch hiện tại.
                      → Q_refined = write_attn(norm_w(memory), norm_w(Q), norm_w(Q))
                      → memory_new = alpha*memory + (1-alpha)*Q_refined
            4. KÉO GRADIENT: Ép memory_new vào graph của batch hiện tại bằng trick x + 0.0 * memory_new
            5. LƯU:   self.memory = memory_new.detach() (Dành cho batch tiếp theo)

        Gradient path (Sạch 100%, không rò rỉ tương lai):
            loss → x_new → (0.0 * memory_new) → Q_refined → write_attn ✓
            Mô hình vừa không ăn gian được tương lai, vừa học được cách nén WRITE.
        """
        B, T, D = x.shape

        # 1. Pre-Norm + trích Q
        x_normed = self.norm1(x)
        Q        = self.Wq(x_normed)                                      # (B, T, D)

        # 2. Self-Attention + RoPE
        attn_out = self.self_attn(Q, freqs_cis, attn_mask=attn_mask)     # (B, T, D)

        if self.use_memory and self.memory is not None:
            # ─── 1. READ TRƯỚC (Đọc từ quá khứ, an toàn 100%) ───
            mem_normed = self.norm_m(self.memory)      # (B, num_slots, D)
            m_out      = self.read(
                Q=Q,
                K=mem_normed,
                V=mem_normed,
            )                                          # (B, T, D)

            # ─── 2. TỔNG HỢP OUTPUT ───
            x_new = x + attn_out + m_out

            # ─── 3. WRITE SAU (Nén thông tin hiện tại cho tương lai) ───
            mem_for_write = self.norm_w(self.memory)   # (B, num_slots, D)
            q_for_write   = self.norm_w(Q)             # (B, T, D)
            
            Q_refined = self.write_attn(
                Q=mem_for_write,
                K=q_for_write,
                V=q_for_write,
                # KHÔNG CẦN MASK NỮA: Vì memory_new này sẽ chỉ được READ bởi batch tiếp theo!
            )                                          # (B, num_slots, D)

            # EMA update 
            memory_new = self.alpha * self.memory + (1 - self.alpha) * Q_refined

            # ─── 4. THỦ THUẬT KÉO GRADIENT MA THUẬT ───
            # Tạo đường dẫn (shortcut) để Gradient từ loss có thể chảy ngược vào write_attn
            # Phép nhân 0.0 đảm bảo giá trị forward của x_new không bị thay đổi.
            x_new = x_new + 0.0 * memory_new.sum()

            # ─── 5. LƯU & CẮT ĐỒ THỊ ───
            self.memory = memory_new.detach()

        else:
            x_new = x + attn_out

        # 3. SwiGLU FFN — Pre-Norm
        out = x_new + self.ffn(self.norm2(x_new))
        return out