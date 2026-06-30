"""
model/block.py — TransformerBlock + Sequence-level Skip/Run routing
====================================================================
Ý tưởng:
    - Mỗi layer giữa có 1 router nhỏ (Linear d_model → 2)
    - Router nhìn mean pooling của sequence → quyết định Skip hay Run
    - Cân bằng bằng DeepSeek bias correction (không cần aux loss)
    - Layer đầu + cuối luôn Run

Config (thêm vào ModelConfig):
    use_router   : bool  = False
    router_gamma : float = 0.5
"""

import math
import torch
import torch.nn as nn

from .attention import SelfAttentionRoPE
from .layers import RMSNorm, SwiGLU


class SequenceRouter(nn.Module):
    """
    1 quyết định Skip/Run cho cả sequence (không phải per-token).
    Cân bằng tải theo DeepSeek: bias = -gamma * (count / total).
    """

    def __init__(self, d_model: int, gamma: float = 0.5):
        super().__init__()
        self.gamma = gamma
        self.proj  = nn.Linear(d_model, 2, bias=False)
        nn.init.normal_(self.proj.weight, std=0.02)
        self.register_buffer("counts", torch.zeros(2))   # [skip, run]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) → chosen: (B,)  0=skip 1=run"""
        logits = self.proj(x.mean(dim=1))                # (B, 2)

        if self.training:
            bias   = -self.gamma * self.counts / (self.counts.sum() + 1e-5)
            logits = logits + bias
            chosen = logits.argmax(dim=-1)               # (B,)
            with torch.no_grad():
                for i in chosen:
                    self.counts[i] += 1
        else:
            chosen = logits.argmax(dim=-1)

        return chosen

    def reset_counts(self):
        self.counts.zero_()

    @property
    def skip_ratio(self) -> float:
        total = self.counts.sum().item()
        return 0.0 if total == 0 else self.counts[0].item() / total


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model     : int,
        n_heads     : int,
        dropout     : float = 0.1,
        n_layers    : int   = 8,
        use_router  : bool  = False,
        router_gamma: float = 0.5,
    ):
        super().__init__()
        self.use_router = use_router
        self.self_attn  = SelfAttentionRoPE(d_model, n_heads, dropout=dropout)
        self.norm2      = RMSNorm(d_model)
        self.ffn        = SwiGLU(d_model, bias=False)

        if use_router:
            self.router = SequenceRouter(d_model, gamma=router_gamma)

        scale = 1.0 / math.sqrt(2 * n_layers)
        nn.init.normal_(self.self_attn.Wo.weight, std=0.02 * scale)
        nn.init.normal_(self.ffn.w2.weight,       std=0.02 * scale)

    def _run_block(self, x, freqs_cis, attn_mask=None):
        x = x + self.self_attn(x, freqs_cis, attn_mask=attn_mask)
        return x + self.ffn(self.norm2(x))

    def forward(self, x, freqs_cis, attn_mask=None):
        if not self.use_router:
            return self._run_block(x, freqs_cis, attn_mask)

        run_mask = self.router(x).bool()                 # (B,) True=run

        # Chỉ chạy block trên sequence được chọn Run — tiết kiệm compute thật
        out            = x.clone()
        out[run_mask]  = self._run_block(x[run_mask], freqs_cis, attn_mask)
        return out


def build_blocks(n_layers, d_model, n_heads, dropout=0.1,
                 use_router=False, router_gamma=0.5) -> nn.ModuleList:
    """Layer 0 và n-1 luôn Run. Layer giữa có router nếu use_router=True."""
    return nn.ModuleList([
        TransformerBlock(
            d_model      = d_model,
            n_heads      = n_heads,
            dropout      = dropout,
            n_layers     = n_layers,
            use_router   = use_router and 0 < i < n_layers - 1,
            router_gamma = router_gamma,
        )
        for i in range(n_layers)
    ])