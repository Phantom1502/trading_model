import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .attention import SelfAttentionRoPE
from .layers import RMSNorm, SwiGLU


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, n_layers=8, use_checkpoint=True):
        super().__init__()
        self.self_attn = SelfAttentionRoPE(d_model, n_heads, dropout=dropout)
        self.norm2 = RMSNorm(d_model)
        self.ffn   = SwiGLU(d_model, bias=False)
        self.use_checkpoint = use_checkpoint
        self._scaled_init(n_layers)

    def _scaled_init(self, n_layers):
        scale = 1.0 / math.sqrt(2 * n_layers)
        nn.init.normal_(self.self_attn.Wo.weight, std=0.02 * scale)
        nn.init.normal_(self.ffn.w2.weight,       std=0.02 * scale)

    def _forward_impl(self, x, freqs_cis, position_offset=0, past_kv=None, use_cache=False):
        attn_out, present = self.self_attn(
            x, freqs_cis,
            position_offset=position_offset,
            past_kv=past_kv,
            use_cache=use_cache,
        )
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, present

    def forward(self, x, freqs_cis, attn_mask=None, position_offset=0, past_kv=None, use_cache=False):
        # Checkpoint chỉ dùng lúc TRAIN thuần (không cache) — generate/eval
        # có self.training=False nên tự đi qua nhánh dưới, không cần check
        # use_cache thủ công riêng cho case đó.
        if self.use_checkpoint and self.training and past_kv is None and not use_cache:
            x = checkpoint(
                lambda x_, f_: self._forward_impl(x_, f_)[0],
                x, freqs_cis,
                use_reentrant=False,
            )
            return x, None
        return self._forward_impl(x, freqs_cis, position_offset, past_kv, use_cache)