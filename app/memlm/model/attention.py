import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import RMSNorm, apply_rope


class SelfAttentionRoPE(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.norm = RMSNorm(d_model)

        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)

        for layer in [self.Wq, self.Wk, self.Wv, self.Wo]:
            nn.init.normal_(layer.weight, std=0.02)

        self.dropout = dropout

    def forward(
        self,
        x,
        freqs_cis,
        attn_mask=None,          # không dùng nữa (is_causal thay thế), giữ để không vỡ call site cũ
        position_offset: int = 0,
        past_kv=None,            # (past_k, past_v) hoặc None
        use_cache: bool = False,
    ):
        B, T, D = x.shape
        h, dh   = self.n_heads, self.d_head
        x_normed = self.norm(x)

        q = self.Wq(x_normed).view(B, T, h, dh).transpose(1, 2)
        k = self.Wk(x_normed).view(B, T, h, dh).transpose(1, 2)
        v = self.Wv(x_normed).view(B, T, h, dh).transpose(1, 2)

        q = apply_rope(q, freqs_cis, offset=position_offset)
        k = apply_rope(k, freqs_cis, offset=position_offset)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        present = (k, v) if use_cache else None

        dropout_p = self.dropout if self.training else 0.0

        # is_causal=True chỉ đúng khi Q, K cùng độ dài (prefill — mỗi token
        # còn cần che token TƯƠNG LAI trong cùng lô). Khi decode với cache
        # (K dài hơn Q, thường Q chỉ có 1 token mới), không cần mask: token
        # mới luôn ở vị trí CUỐI, được phép attend hết K/V hiện có.
        is_causal = (k.shape[2] == q.shape[2])

        out = F.scaled_dot_product_attention(
            query=q, key=k, value=v,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )

        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.Wo(out), present