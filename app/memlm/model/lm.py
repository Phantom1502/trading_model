"""
model/lm.py — MemoryLM (LLaMA-style tích hợp MoD Cách 1)
"""

import torch
import torch.nn as nn

from .block import TransformerBlock
from .layers import RMSNorm, precompute_freqs_cis


class MemoryLM(nn.Module):
    def __init__(
        self,
        vocab_size : int,
        d_model    : int   = 512,
        n_heads    : int   = 8,
        n_layers   : int   = 8,
        max_seq    : int   = 512,
        dropout    : float = 0.1,
        rope_base  : float = 10000.0,
        use_mod       : bool  = False,
        mod_capacity  : float = 0.5,    
        mod_interleave: bool  = True,   
    ):
        super().__init__()
        self.d_model  = d_model
        self.n_layers = n_layers
        self.max_seq  = max_seq
        self.use_mod  = use_mod

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.drop      = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model, n_heads, dropout,
                n_layers      = n_layers,
                layer_idx     = i,              
                use_mod       = use_mod,
                mod_capacity  = mod_capacity,
                mod_interleave= mod_interleave,
            )
            for i in range(n_layers)
        ])

        self.norm_out = RMSNorm(d_model)
        self.lm_head  = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight   # weight tying

        d_head    = d_model // n_heads
        freqs_cis = precompute_freqs_cis(d_head, max_seq * 2, base=rope_base)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        nn.init.normal_(self.token_emb.weight, std=0.02)

    def num_params(self, trainable_only: bool = False) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
    
    def forward(
        self,
        input_ids      : torch.Tensor,
        attn_mask      : torch.Tensor = None,
        return_aux_loss: bool         = False,   
    ):
        B, T   = input_ids.shape
        device = input_ids.device

        x         = self.drop(self.token_emb(input_ids))
        freqs_cis = self.freqs_cis.to(device)

        total_aux_loss = torch.tensor(0.0, device=device)
        n_mod_layers   = 0   
        
        for block in self.blocks:
            if self.use_mod and return_aux_loss:
                x, aux_loss = block(
                    x, freqs_cis=freqs_cis, attn_mask=attn_mask,
                    return_aux_loss=True,
                )
                if block.mod_active:
                    total_aux_loss = total_aux_loss + aux_loss
                    n_mod_layers  += 1
            else:
                x = block(
                    x, freqs_cis=freqs_cis, attn_mask=attn_mask,
                    return_aux_loss=False
                )

        x      = self.norm_out(x)
        logits = self.lm_head(x)
 
        if return_aux_loss and self.use_mod:
            norm_aux = total_aux_loss / max(n_mod_layers, 1)
            return logits, norm_aux
 
        return logits


def causal_mask(T: int, device: torch.device) -> torch.Tensor:
    """Causal mask additive cho autoregressive attention."""
    mask = torch.triu(torch.ones(T, T, device=device), diagonal=1)
    return mask.masked_fill(mask.bool(), float("-inf"))


def build_model(cfg) -> MemoryLM:
    """Entry point xây model từ ModelConfig."""
    return MemoryLM(
        vocab_size    = cfg.model.vocab_size,
        d_model       = cfg.model.d_model,
        n_heads       = cfg.model.n_heads,
        n_layers      = cfg.model.n_layers,
        max_seq       = cfg.model.max_seq,
        dropout       = cfg.model.dropout,
        rope_base     = getattr(cfg.model, "rope_base",      10000.0),
        use_mod       = getattr(cfg.model, "use_mod",        False),
        mod_capacity  = getattr(cfg.model, "mod_capacity",   0.5),
        mod_interleave= getattr(cfg.model, "mod_interleave", True),
    )