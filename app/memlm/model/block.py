"""
model/block.py — TransformerBlock với Sequence-level Skip/Run routing
======================================================================
NHÁNH THỬ NGHIỆM

Ý tưởng:
    Router quyết định ở cấp độ SEQUENCE (không phải token):
        - Nhìn vào mean pooling của hidden state → 1 quyết định cho cả sequence
        - Sequence "đơn giản" → skip layer này
        - Sequence "phức tạp" → đi qua full block

    Khác với token-level routing:
        Token-level : mỗi token trong sequence có thể đi khác nhau
                      → attention bị vỡ vì token cùng sequence đi khác nhau
        Sequence-level: cả sequence đi cùng 1 hướng
                      → attention hoàn toàn bình thường
                      → không có vấn đề gì về RoPE hay causal mask

    Layer đầu (idx=0) và cuối (idx=n_layers-1): luôn Run.
    Các layer giữa: router quyết định per-sequence.

Cân bằng — DeepSeek bias correction:
    Đếm số lần Skip vs Run được chọn.
    bias_i = -gamma * (count_i / total)
    Nhánh bị chọn nhiều → bias âm → tự động nhường chỗ cho nhánh kia.
    Không cần loss term thêm vào objective.

Config thêm vào ModelConfig:
    use_router   : bool  = False
    router_gamma : float = 0.5
"""

import math
import torch
import torch.nn as nn

from .attention import SelfAttentionRoPE
from .layers import RMSNorm, SwiGLU


# ══════════════════════════════════════════════════════════════════════
# SequenceRouter — 1 quyết định cho cả sequence
# ══════════════════════════════════════════════════════════════════════

class SequenceRouter(nn.Module):
    """
    Router cấp sequence: nhìn vào mean pooling → Skip hay Run.

    Tại sao mean pooling:
        - Đại diện cho "nội dung tổng thể" của sequence
        - Sequence phức tạp → hidden state đa dạng → mean cao hơn
        - Không cần nhìn từng token → 1 vector D → 1 quyết định

    Bias correction theo DeepSeek:
        expert_counts[0] = số sequence được Skip
        expert_counts[1] = số sequence được Run
        bias = -gamma * (count / total) → tự cân bằng
    """

    IDX_SKIP = 0
    IDX_RUN  = 1

    def __init__(self, d_model: int, gamma: float = 0.5):
        super().__init__()
        self.gamma = gamma

        # d_model → 2 logit (skip vs run) — nhìn vào mean pooling
        self.proj = nn.Linear(d_model, 2, bias=False)
        nn.init.normal_(self.proj.weight, std=0.02)

        # Đếm tần suất — per sequence, không phải per token
        self.register_buffer("expert_counts", torch.zeros(2))

    def _bias(self) -> torch.Tensor:
        total = self.expert_counts.sum() + 1e-5
        return -self.gamma * (self.expert_counts / total)   # [2]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, T, D)
        Returns: chosen (B,) — 0=skip, 1=run, 1 giá trị cho MỖI sequence
        """
        # Mean pooling qua chiều T → đại diện toàn sequence
        x_mean = x.mean(dim=1)                             # (B, D)

        logits = self.proj(x_mean)                         # (B, 2)

        if self.training:
            bias   = self._bias()                          # [2]
            logits = logits + bias.unsqueeze(0)            # (B, 2)

        # Argmax → 1 quyết định cho mỗi sequence
        chosen = logits.argmax(dim=-1)                     # (B,)

        # Cập nhật counts theo số sequence, không phải số token
        if self.training:
            with torch.no_grad():
                for idx in chosen:
                    self.expert_counts[idx] += 1

        return chosen

    def reset_counts(self):
        self.expert_counts.zero_()

    @property
    def skip_ratio(self) -> float:
        """Tỉ lệ sequence được skip — dùng để monitor."""
        total = self.expert_counts.sum().item()
        if total == 0:
            return 0.0
        return self.expert_counts[self.IDX_SKIP].item() / total


# ══════════════════════════════════════════════════════════════════════
# TransformerBlock
# ══════════════════════════════════════════════════════════════════════

class TransformerBlock(nn.Module):
    """
    use_router=False : luôn chạy full block (layer đầu & cuối)
    use_router=True  : sequence router quyết định skip hay run
    """

    def __init__(
        self,
        d_model     : int,
        n_heads     : int,
        dropout     : float = 0.1,
        n_layers    : int   = 8,
        layer_idx   : int   = 0,
        use_router  : bool  = False,
        router_gamma: float = 0.5,
    ):
        super().__init__()
        self.use_router = use_router

        self.self_attn = SelfAttentionRoPE(d_model, n_heads, dropout=dropout)
        self.norm2     = RMSNorm(d_model)
        self.ffn       = SwiGLU(d_model, bias=False)

        if use_router:
            self.router = SequenceRouter(d_model, gamma=router_gamma)

        self._scaled_init(n_layers)

    def _scaled_init(self, n_layers: int):
        scale = 1.0 / math.sqrt(2 * n_layers)
        nn.init.normal_(self.self_attn.Wo.weight, std=0.02 * scale)
        nn.init.normal_(self.ffn.w2.weight,       std=0.02 * scale)

    def _run_block(self, x, freqs_cis, attn_mask=None):
        attn_out = self.self_attn(x, freqs_cis, attn_mask=attn_mask)
        x_post   = x + attn_out
        return x_post + self.ffn(self.norm2(x_post))

    def forward(
        self,
        x        : torch.Tensor,   # (B, T, D)
        freqs_cis: torch.Tensor,
        attn_mask: torch.Tensor = None,
    ) -> torch.Tensor:

        # ── Layer đầu & cuối: luôn run ───────────────────────────────
        if not self.use_router:
            return self._run_block(x, freqs_cis, attn_mask)

        # ── Router: 1 quyết định cho mỗi sequence ────────────────────
        chosen   = self.router(x)                          # (B,)
        run_mask = (chosen == SequenceRouter.IDX_RUN)      # (B,) bool

        # Tất cả sequence skip → trả nguyên x
        if not run_mask.any():
            return x

        # Tất cả sequence run → full block
        if run_mask.all():
            return self._run_block(x, freqs_cis, attn_mask)

        # Mixed: một số sequence run, một số skip
        # Tách batch → chạy block chỉ trên sequence được chọn Run
        # → tiết kiệm compute thật sự vì không chạy attention trên sequence skip
        x_run     = x[run_mask]                            # (B_run, T, D)
        out       = x.clone()                              # (B, T, D) — default skip

        # freqs_cis không có batch dim → dùng chung được
        block_out          = self._run_block(x_run, freqs_cis, attn_mask)
        out[run_mask]      = block_out                     # ghi kết quả vào đúng vị trí

        return out


# ══════════════════════════════════════════════════════════════════════
# Build helper
# ══════════════════════════════════════════════════════════════════════

def build_blocks(
    n_layers    : int,
    d_model     : int,
    n_heads     : int,
    dropout     : float = 0.1,
    use_router  : bool  = False,
    router_gamma: float = 0.5,
) -> nn.ModuleList:
    """
    Layer 0 và layer n-1: luôn run (use_router=False).
    Layer 1 ~ n-2: có sequence router nếu use_router=True.
    """
    blocks = []
    for i in range(n_layers):
        is_fixed = (i == 0) or (i == n_layers - 1)
        blocks.append(
            TransformerBlock(
                d_model      = d_model,
                n_heads      = n_heads,
                dropout      = dropout,
                n_layers     = n_layers,
                layer_idx    = i,
                use_router   = use_router and not is_fixed,
                router_gamma = router_gamma,
            )
        )
    return nn.ModuleList(blocks)