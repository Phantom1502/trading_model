"""
model/block.py — Transformer Block (LLaMA-style)
=================================================
Luồng:
    x → SelfAttentionRoPE (Pre-Norm bên trong) → residual → FFN (Pre-Norm) → out
"""

import math
import torch
import torch.nn as nn

from .attention import SelfAttentionRoPE
from .layers import RMSNorm, SwiGLU

class DepthRouter(nn.Module):
    """
    Router quyết định token này có nên đi qua layer hay không.

    output:
        gate.shape = (B, T, 1)
        gate ∈ [0,1]
    """

    def __init__(self, d_model: int):
        super().__init__()

        self.proj = nn.Linear(
            d_model,
            1,
            bias=False
        )

        nn.init.normal_(
            self.proj.weight,
            std=0.02
        )

    def forward(self, x):
        return torch.sigmoid(self.proj(x))
    
class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        n_layers: int = 8,

        # bật/tắt router
        use_router: bool = False,

        # tỷ lệ token mong muốn đi qua layer
        target_usage: float = 0.5,

        # threshold khi inference
        inference_threshold: float = 0.5,
    ):
        super().__init__()

        self.self_attn = SelfAttentionRoPE(
            d_model,
            n_heads,
            dropout=dropout
        )

        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, bias=False)

        self.use_router = use_router
        self.target_usage = target_usage
        self.inference_threshold = inference_threshold

        if use_router:
            self.router = DepthRouter(d_model)

        self._scaled_init(n_layers)

    def _scaled_init(self, n_layers: int):
        """Scale init cho các projection nằm trên đường residual."""
        scale = 1.0 / math.sqrt(2 * n_layers)
        nn.init.normal_(self.self_attn.Wo.weight, std=0.02 * scale)
        nn.init.normal_(self.ffn.w2.weight,       std=0.02 * scale)

    def _run_block(
        self,
        x,
        freqs_cis,
        attn_mask=None,
    ):
        """
        Dense transformer block gốc.
        """

        x = x + self.self_attn(
            x,
            freqs_cis,
            attn_mask=attn_mask
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
        return_aux_loss=False,
    ):
        """
        return:
            out
            balance_loss (optional)
        """

        #
        # Không dùng router
        #
        if not self.use_router:
            out = self._run_block(
                x,
                freqs_cis,
                attn_mask
            )

            if return_aux_loss:
                return out, x.new_zeros(())
            return out

        #
        # gate shape = (B,T,1)
        #
        gate = self.router(x)

        #
        # ==================================================
        # TRAIN
        # ==================================================
        #
        if self.training:

            block_out = self._run_block(
                x,
                freqs_cis,
                attn_mask
            )

            #
            # Identity Expert:
            #     y = x
            #
            # Layer Expert:
            #     y = block_out
            #
            # Soft routing:
            #     y = x + g * (block_out - x)
            #
            delta = block_out - x
            out = x + gate * delta

            #
            # Load balancing.
            #
            # usage = % token đi qua layer.
            #
            usage = gate.mean()

            balance_loss = (
                usage - self.target_usage
            ).pow(2)

            if return_aux_loss:
                return out, balance_loss

            return out

        #
        # ==================================================
        # INFERENCE
        # ==================================================
        #

        #
        # token nào cần compute thêm
        #
        mask = (
            gate.squeeze(-1)
            >= self.inference_threshold
        )

        #
        # mặc định: đi thẳng
        #
        out = x.clone()

        #
        # chỉ chạy block với token được chọn
        #
        if mask.any():

            x_packed = x[mask]

            #
            # xử lý RoPE tương ứng
            #
            if freqs_cis.dim() == 4:
                freqs_cis_packed = freqs_cis[mask]

            elif freqs_cis.dim() == 3:
                freqs_cis_packed = freqs_cis[mask]

            else:
                #
                # prefill thường dùng chung
                #
                freqs_cis_packed = freqs_cis

            attn_mask_packed = None
            if attn_mask is not None:
                attn_mask_packed = attn_mask[mask]

            out[mask] = self._run_block(
                x_packed,
                freqs_cis_packed,
                attn_mask_packed
            )

        if return_aux_loss:
            return out, x.new_zeros(())

        return out