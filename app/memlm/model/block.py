# model/block.py
# ============================================================
# TransformerBlock + Sequence-level Dynamic Depth Routing
# ============================================================

import math
import torch
import torch.nn as nn

from .attention import SelfAttentionRoPE
from .layers import RMSNorm, SwiGLU


# ============================================================
# Sequence Router
# ============================================================

class SequenceRouter(nn.Module):
    """
    Router nhìn toàn bộ (T, D) của sequence bằng một CNN nhỏ.

    x:
        (B, T, D)

    quyết định:
        0 = Skip
        1 = Run

    Cân bằng bằng DeepSeek bias correction:
        bias_i = -gamma * count_i / total
    """

    IDX_SKIP = 0
    IDX_RUN = 1

    def __init__(
        self,
        d_model: int,
        gamma: float = 0.5,
    ):
        super().__init__()

        self.gamma = gamma

        self.encoder = nn.Sequential(
            nn.Conv2d(
                1,
                8,
                kernel_size=5,
                padding=2,
            ),
            nn.GELU(),

            nn.Conv2d(
                8,
                16,
                kernel_size=5,
                padding=2,
            ),
            nn.GELU(),

            nn.AdaptiveAvgPool2d(1),
        )

        self.head = nn.Linear(
            16,
            2,
            bias=False,
        )

        nn.init.normal_(
            self.head.weight,
            std=0.02,
        )

        self.register_buffer(
            "counts",
            torch.zeros(2),
        )

    def _bias(self):
        total = self.counts.sum() + 1e-5
        return -self.gamma * self.counts / total

    def forward(
        self,
        x: torch.Tensor,
    ):
        """
        x:
            (B,T,D)

        returns:
            run_mask:
                (B,) bool
        """

        h = x.unsqueeze(1)          # (B,1,T,D)

        h = self.encoder(h)         # (B,16,1,1)
        h = h.flatten(1)            # (B,16)

        logits = self.head(h)       # (B,2)

        if self.training:
            logits = logits + self._bias()

        chosen = logits.argmax(dim=-1)

        if self.training:
            with torch.no_grad():
                self.counts.scatter_add_(
                    0,
                    chosen,
                    torch.ones_like(
                        chosen,
                        dtype=self.counts.dtype,
                    ),
                )

        return chosen == self.IDX_RUN

    def reset_counts(self):
        self.counts.zero_()

    @property
    def skip_ratio(self):
        total = self.counts.sum().item()
        if total == 0:
            return 0.0

        return (
            self.counts[self.IDX_SKIP].item()
            / total
        )


# ============================================================
# Transformer Block
# ============================================================

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        n_layers: int = 8,
        use_router: bool = False,
        router_gamma: float = 0.5,
    ):
        super().__init__()

        self.use_router = use_router

        self.self_attn = SelfAttentionRoPE(
            d_model,
            n_heads,
            dropout=dropout,
        )

        self.norm2 = RMSNorm(d_model)

        self.ffn = SwiGLU(
            d_model,
            bias=False,
        )

        if use_router:
            self.router = SequenceRouter(
                d_model=d_model,
                gamma=router_gamma,
            )

        scale = 1.0 / math.sqrt(
            2 * n_layers
        )

        nn.init.normal_(
            self.self_attn.Wo.weight,
            std=0.02 * scale,
        )

        nn.init.normal_(
            self.ffn.w2.weight,
            std=0.02 * scale,
        )

    def _run_block(
        self,
        x,
        freqs_cis,
        attn_mask=None,
    ):
        x = x + self.self_attn(
            x,
            freqs_cis,
            attn_mask=attn_mask,
        )

        x = x + self.ffn(
            self.norm2(x)
        )

        return x

    def forward(
        self,
        x,
        freqs_cis,
        attn_mask=None,
    ):
        """
        x:
            (B,T,D)
        """

        # layer cố định
        if not self.use_router:
            return self._run_block(
                x,
                freqs_cis,
                attn_mask,
            )

        run_mask = self.router(x)

        # toàn bộ skip
        if not run_mask.any():
            return x

        # toàn bộ run
        if run_mask.all():
            return self._run_block(
                x,
                freqs_cis,
                attn_mask,
            )

        run_idx = run_mask.nonzero(
            as_tuple=True
        )[0]

        x_run = x[run_idx]

        attn_mask_run = None
        if attn_mask is not None:
            attn_mask_run = attn_mask[
                run_idx
            ]

        out = x.clone()

        out_run = self._run_block(
            x_run,
            freqs_cis,
            attn_mask_run,
        )

        out[run_idx] = out_run

        return out


# ============================================================
# Build Blocks
# ============================================================

def build_blocks(
    n_layers: int,
    d_model: int,
    n_heads: int,
    dropout: float = 0.1,
    use_router: bool = False,
    router_gamma: float = 0.5,
):
    blocks = []

    for i in range(n_layers):

        fixed = (
            i == 0
            or i == n_layers - 1
        )

        blocks.append(
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
                n_layers=n_layers,
                use_router=(
                    use_router
                    and not fixed
                ),
                router_gamma=router_gamma,
            )
        )

    return nn.ModuleList(blocks)