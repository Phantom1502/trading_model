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
        # KHÔNG còn pos_emb — RoPE thay thế absolute position embedding
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

        # Precompute bảng RoPE — đăng ký làm buffer để tự chuyển device cùng model
        d_head = d_model // n_heads
        freqs_cis = precompute_freqs_cis(d_head, max_seq * 2, base=rope_base)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_emb.weight, std=0.02)
        # std=0.02 mặc định cho mọi Linear (Wq, Wk, Wv, w1, w3, ...).
        # Các layer trên đường residual (Wo, w2) đã được _scaled_init() trong
        # MemoryBlock.__init__ ghi đè SAU bước này (xem thứ tự gọi: mỗi
        # MemoryBlock tự gọi _scaled_init ở cuối __init__ của chính nó,
        # trước khi MemoryLM._init_weights() chạy) — nên ở đây init lại std=0.02
        # cho TẤT CẢ Linear sẽ vô tình ghi đè luôn cả phần đã scaled.
        # Do đó: chỉ init Linear nằm NGOÀI block (không có ở đây, lm_head dùng
        # weight tying nên không cần). Hàm này chỉ còn init token_emb.
        pass

    def num_params(self, trainable_only: bool = False) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    # ── Memory management ────────────────────────────────────────────────
    def reset_memory(self, batch_size: int, device: torch.device):
        """Reset M về trạng thái khởi tạo — gọi khi bắt đầu document mới."""
        if self.use_memory:
            for block in self.blocks:
                block.reset_memory(batch_size, device)

    def detach_memory(self):
        """Cắt gradient của M giữa các segment — truncated BPTT."""
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

    # ── Forward ───────────────────────────────────────────────────────────
    def forward(self, input_ids: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        """
        input_ids: (B, T)
        Returns: logits (B, T, vocab_size)
        """
        B, T = input_ids.shape
        device = input_ids.device

        if self.use_memory and not self.has_memory_initialized(batch_size=B):
            self.reset_memory(B, device)

        # Không còn cộng pos_emb — vị trí được mã hóa qua RoPE bên trong mỗi block
        x = self.drop(self.token_emb(input_ids))

        freqs_cis = self.freqs_cis.to(device)

        for block in self.blocks:
            x = block(x, freqs_cis=freqs_cis, attn_mask=attn_mask)

        x = self.norm_out(x)
        return self.lm_head(x)


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
    )
