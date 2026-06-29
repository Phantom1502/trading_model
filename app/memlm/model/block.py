"""
model/block.py — TransformerBlock với Skip/Run routing + DeepSeek bias correction
===================================================================================
NHÁNH THỬ NGHIỆM

Ý tưởng:
    Mỗi token tại mỗi layer được router quyết định:
        Expert 0 = Identity (skip) — x đi thẳng, không tốn compute
        Expert 1 = Run            — x đi qua full TransformerBlock

    Layer đầu (idx=0) và layer cuối (idx=n_layers-1): luôn Run, không có router.
    Các layer giữa: có router, top-1 chọn Skip hoặc Run.

Cân bằng tải — DeepSeek bias correction (thay cho aux/balance loss):
    Router cho ra 2 logit [logit_skip, logit_run].
    Trước khi argmax, cộng thêm bias:
        bias_i = -gamma * (count_i / total_count)
    Expert nào được chọn nhiều → count cao → bias âm lớn → bị trừ điểm.
    Tự động cân bằng mà không cần thêm loss term vào objective.

    Ưu điểm so với balance loss:
        - Không cần tune thêm loss coefficient
        - Không can thiệp vào gradient của lm_loss
        - Đơn giản, ổn định hơn khi train

Config thêm vào ModelConfig (config.py):
    use_router  : bool  = False  # bật/tắt routing cho layer giữa
    router_gamma: float = 0.5    # DeepSeek bias correction strength
                                  # tune trong [0.1, 1.0]
                                  # cao → cân bằng mạnh hơn
                                  # thấp → router tự do hơn
"""

import math
import torch
import torch.nn as nn

from .attention import SelfAttentionRoPE
from .layers import RMSNorm, SwiGLU


# ══════════════════════════════════════════════════════════════════════
# DepthRouter — Skip hay Run, cân bằng theo DeepSeek
# ══════════════════════════════════════════════════════════════════════

class DepthRouter(nn.Module):
    """
    Router 2 expert: Skip (0) vs Run (1).

    Cơ chế cân bằng DeepSeek:
        - Buffer `expert_counts[2]` đếm số lần mỗi expert được chọn
        - bias = -gamma * (count / total) cộng vào logits trước argmax
        - Expert bị chọn quá nhiều → bias âm lớn → tự động nhường chỗ
        - Không cần loss term, không cần tune coefficient riêng
    """

    IDX_SKIP = 0
    IDX_RUN  = 1

    def __init__(self, d_model: int, gamma: float = 0.5):
        super().__init__()
        self.gamma = gamma

        # Linear nhỏ: d_model → 2 logit (skip vs run)
        self.proj = nn.Linear(d_model, 2, bias=False)
        nn.init.normal_(self.proj.weight, std=0.02)

        # Đếm tần suất chọn — buffer, không train được
        self.register_buffer("expert_counts", torch.zeros(2))

    def _bias(self) -> torch.Tensor:
        """
        DeepSeek bias correction:
            bias_i = -gamma * (count_i / total_count)
        Expert nhiều → count cao → bias âm → bị trừ điểm → tự cân bằng.
        """
        total = self.expert_counts.sum() + 1e-5
        return -self.gamma * (self.expert_counts / total)   # [2]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, T, D)
        Returns: chosen (B, T) — 0=skip, 1=run
        """
        B, T, _ = x.shape

        logits = self.proj(x)                              # (B, T, 2)

        if self.training:
            # Cộng bias correction vào logits trước khi chọn
            bias   = self._bias()                          # [2]
            logits = logits + bias.unsqueeze(0).unsqueeze(0)

        # Top-1: chọn expert có logit cao nhất
        chosen = logits.argmax(dim=-1)                     # (B, T)

        # Cập nhật counts — no_grad, chỉ là thống kê
        if self.training:
            with torch.no_grad():
                for idx in chosen.view(-1):
                    self.expert_counts[idx] += 1

        return chosen

    def reset_counts(self):
        """Reset counts mỗi chunk để bias không bị stale."""
        self.expert_counts.zero_()

    @property
    def skip_ratio(self) -> float:
        """Tỉ lệ token skip từ đầu epoch — dùng để monitor."""
        total = self.expert_counts.sum().item()
        if total == 0:
            return 0.0
        return self.expert_counts[self.IDX_SKIP].item() / total


# ══════════════════════════════════════════════════════════════════════
# TransformerBlock
# ══════════════════════════════════════════════════════════════════════

class TransformerBlock(nn.Module):
    """
    LLaMA-style block với tùy chọn Skip/Run routing.

    use_router=False (layer đầu & cuối):
        Luôn chạy full block — không có router, không có skip.

    use_router=True (layer giữa):
        Router chọn Skip hoặc Run cho từng token.
        Train: chạy full block, sau đó select theo chosen mask.
        Infer: giống train — chạy full block, select theo mask.
        (Không cần soft gate — bias correction đã đảm bảo cân bằng)
    """

    def __init__(
        self,
        d_model    : int,
        n_heads    : int,
        dropout    : float = 0.1,
        n_layers   : int   = 8,
        layer_idx  : int   = 0,
        # ── Routing config ───────────────────────────────────────────
        use_router  : bool  = False,
        router_gamma: float = 0.5,
    ):
        super().__init__()
        self.layer_idx  = layer_idx
        self.use_router = use_router

        # ── Attention + FFN — giống block gốc hoàn toàn ──────────────
        self.self_attn = SelfAttentionRoPE(d_model, n_heads, dropout=dropout)
        self.norm2     = RMSNorm(d_model)
        self.ffn       = SwiGLU(d_model, bias=False)

        # ── Router — chỉ tạo cho layer giữa ──────────────────────────
        if use_router:
            self.router = DepthRouter(d_model, gamma=router_gamma)

        self._scaled_init(n_layers)

    def _scaled_init(self, n_layers: int):
        scale = 1.0 / math.sqrt(2 * n_layers)
        nn.init.normal_(self.self_attn.Wo.weight, std=0.02 * scale)
        nn.init.normal_(self.ffn.w2.weight,       std=0.02 * scale)

    def _run_block(self, x, freqs_cis, attn_mask=None):
        """Full block: Attention + FFN."""
        attn_out = self.self_attn(x, freqs_cis, attn_mask=attn_mask)
        x_post   = x + attn_out
        return x_post + self.ffn(self.norm2(x_post))

    def forward(
        self,
        x        : torch.Tensor,   # (B, T, D)
        freqs_cis: torch.Tensor,
        attn_mask: torch.Tensor = None,
    ) -> torch.Tensor:

        # ── Layer đầu & cuối: luôn chạy full block ───────────────────
        if not self.use_router:
            return self._run_block(x, freqs_cis, attn_mask)

        # ── Layer giữa: router quyết định skip hay run ────────────────
        chosen   = self.router(x)                          # (B, T)
        run_mask = (chosen == DepthRouter.IDX_RUN)         # (B, T) bool

        # Tất cả skip → trả nguyên x, tiết kiệm toàn bộ compute
        if not run_mask.any():
            return x

        # Tất cả run → chạy full block, không cần masking
        if run_mask.all():
            return self._run_block(x, freqs_cis, attn_mask)

        # Mixed: chạy full block, sau đó select theo mask
        # Không thể pack token vì attention cần (B, T, D) liên tục
        # để RoPE và causal mask đúng vị trí
        block_out = self._run_block(x, freqs_cis, attn_mask)   # (B, T, D)

        # Token được chọn Run → block_out, token Skip → x gốc
        mask = run_mask.unsqueeze(-1).expand_as(x)         # (B, T, D)
        return torch.where(mask, block_out, x)


# ══════════════════════════════════════════════════════════════════════
# Build helper — dùng trong lm.py
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
    Tạo danh sách block theo quy tắc:
        Layer 0            : luôn run (use_router=False)
        Layer 1 ~ n-2      : có router (use_router=True) nếu use_router=True
        Layer n_layers-1   : luôn run (use_router=False)

    Đảm bảo layer đầu và cuối không bao giờ skip.
    """
    blocks = []
    for i in range(n_layers):
        is_fixed = (i == 0) or (i == n_layers - 1)
        blocks.append(
            TransformerBlock(
                d_model     = d_model,
                n_heads     = n_heads,
                dropout     = dropout,
                n_layers    = n_layers,
                layer_idx   = i,
                use_router  = use_router and not is_fixed,
                router_gamma= router_gamma,
            )
        )
    return nn.ModuleList(blocks)