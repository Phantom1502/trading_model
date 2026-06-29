"""
model/block.py — Transformer Block (LLaMA-style) with MoE-style Depth Routing
===============================================================================
Luồng tổng quát:
    x → [Layer 0: bắt buộc chạy] → [Layer giữa: top-1 chọn Skip hoặc Run] → [Layer cuối: bắt buộc chạy]

Với các layer giữa (use_router=True):
    - Có 2 "expert": Expert 0 = Identity (skip), Expert 1 = TransformerBlock thật sự
    - Router cho ra 2 logits → top-1 chọn 1 trong 2
    - Bias correction: track tần suất mỗi lựa chọn → trừ điểm lựa chọn bị dùng quá nhiều
    - Train : soft blend  →  out = w_skip * x  +  w_run * block_out   (differentiable)
    - Infer : hard route  →  chạy đúng nhánh được chọn, nhánh kia bỏ qua hoàn toàn
"""

import math
import torch
import torch.nn as nn

from .attention import SelfAttentionRoPE
from .layers import RMSNorm, SwiGLU


# ---------------------------------------------------------------------------
# DepthRouter — chọn Skip hay Run cho từng token
# ---------------------------------------------------------------------------

class DepthRouter(nn.Module):
    """
    Cho mỗi token, tạo ra 2 logits tương ứng với:
        logit[0] → Expert 0: Skip (identity, x đi thẳng)
        logit[1] → Expert 1: Run  (chạy full transformer block)

    Có bias correction để cân bằng tải giữa skip và run.

    Buffers (không phải parameter, không train được):
        route_counts : [2,]  — tổng số lần mỗi nhánh được chọn (cộng dồn qua các batch)
    """

    NUM_EXPERTS = 2   # 0 = skip, 1 = run
    IDX_SKIP    = 0
    IDX_RUN     = 1

    def __init__(self, d_model: int, gamma: float = 0.5):
        """
        Args:
            d_model : chiều hidden của model
            gamma   : hệ số phạt bias correction (càng cao → cân bằng càng mạnh)
        """
        super().__init__()

        self.gamma = gamma

        self.proj = nn.Linear(d_model, self.NUM_EXPERTS, bias=False)
        nn.init.normal_(self.proj.weight, std=0.02)

        # Đếm tần suất từng nhánh được chọn — giống expert_counts trong MoE
        self.register_buffer(
            "route_counts",
            torch.zeros(self.NUM_EXPERTS)
        )

    # ------------------------------------------------------------------
    def _bias(self) -> torch.Tensor:
        """
        Tính bias correction theo công thức giống DeepSeek:
            bias = -gamma * (count_i / total_count)

        Nhánh nào bị chọn nhiều hơn → bị trừ điểm nhiều hơn
        → router tự động trao cơ hội cho nhánh còn lại.
        """
        total = self.route_counts.sum() + 1e-5
        load_ratio = self.route_counts / total          # [2,]
        return -self.gamma * load_ratio                 # [2,]

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,                               # (B, T, d_model)
        training: bool = True,
    ):
        """
        Returns:
            weights  : (B, T, 2)  — softmax weights cho 2 nhánh (dùng khi train)
            chosen   : (B, T)     — index nhánh được chọn  (0=skip, 1=run)
        """
        B, T, _ = x.shape

        logits = self.proj(x)                          # (B, T, 2)

        # Áp bias correction vào logits trước khi chọn
        if training:
            bias = self._bias()                        # [2,]
            logits = logits + bias.unsqueeze(0).unsqueeze(0)

        # Top-1: chọn nhánh có logit cao nhất cho từng token
        chosen = logits.argmax(dim=-1)                 # (B, T)

        # Cập nhật route_counts (no_grad vì đây chỉ là thống kê)
        if training:
            with torch.no_grad():
                flat_chosen = chosen.view(-1)          # (B*T,)
                for idx in flat_chosen:
                    self.route_counts[idx] += 1

        return chosen


# ---------------------------------------------------------------------------
# TransformerBlock
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """
    use_router=False  →  layer đầu / cuối, luôn chạy full block.
    use_router=True   →  layer giữa, top-1 routing giữa Skip và Run.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        n_layers: int = 8,
        use_router: bool = False,
        # chỉ dùng khi use_router=True
        gamma: float = 0.5,
        inference_threshold: float = 0.5,
    ):
        """
        Args:
            d_model             : chiều hidden
            n_heads             : số attention head
            dropout             : dropout rate
            n_layers            : tổng số layer (dùng để scale init)
            use_router          : False → layer cố định; True → layer có routing
            gamma               : hệ số bias correction (chỉ dùng khi use_router=True)
            inference_threshold : ngưỡng để chạy block khi inference
                                  (nếu weight_run >= threshold → chạy block)
        """
        super().__init__()

        self.self_attn = SelfAttentionRoPE(d_model, n_heads, dropout=dropout)
        self.norm2     = RMSNorm(d_model)
        self.ffn       = SwiGLU(d_model, bias=False)

        self.use_router          = use_router
        self.inference_threshold = inference_threshold

        if use_router:
            self.router = DepthRouter(d_model, gamma=gamma)

        self._scaled_init(n_layers)

    # ------------------------------------------------------------------
    def _scaled_init(self, n_layers: int):
        scale = 1.0 / math.sqrt(2 * n_layers)
        nn.init.normal_(self.self_attn.Wo.weight, std=0.02 * scale)
        nn.init.normal_(self.ffn.w2.weight,       std=0.02 * scale)

    # ------------------------------------------------------------------
    def _run_block(self, x, freqs_cis, attn_mask=None):
        """Full transformer block: Attention + FFN."""
        x = x + self.self_attn(x, freqs_cis, attn_mask=attn_mask)
        x = x + self.ffn(self.norm2(x))
        return x

    # ------------------------------------------------------------------
    def forward(
        self,
        x,
        freqs_cis,
        attn_mask=None,
        return_aux_loss=False,
    ):
        """
        Args:
            x               : (B, T, d_model)
            freqs_cis       : RoPE frequencies
            attn_mask       : attention mask (optional)
            return_aux_loss : nếu True, trả thêm scalar aux_loss (luôn = 0 vì
                              cân bằng tải đã được xử lý bởi bias correction,
                              không cần loss phụ)

        Returns:
            out             : (B, T, d_model)
            aux_loss        : scalar 0 (chỉ trả khi return_aux_loss=True)
        """

        # ==============================================================
        # LAYER CỐ ĐỊNH (layer đầu & cuối): luôn chạy full block
        # ==============================================================
        if not self.use_router:
            out = self._run_block(x, freqs_cis, attn_mask)
            if return_aux_loss:
                return out, x.new_zeros(())
            return out

        # ==============================================================
        # LAYER CÓ ROUTING
        # ==============================================================

        # Router trả về:
        #   weights : (B, T, 2)  — softmax([logit_skip, logit_run])
        #   chosen  : (B, T)     — 0=skip, 1=run
        chosen = self.router(x, training=self.training)

        # ----------------------------------------------------------
        # TRAIN & INFERENCE — Hard routing (giống nhau hoàn toàn)
        # ----------------------------------------------------------
        #
        # chosen=0 → skip: out = x          (không tốn compute)
        # chosen=1 → run : out = block_out  (chạy full block)
        #
        # Bias correction trong router tự ép cân bằng khi một nhánh
        # bị chọn quá nhiều — không cần soft blend, không cần aux_loss.
        #
        out = x.clone()                               # mặc định: skip

        run_mask = (chosen == DepthRouter.IDX_RUN)   # (B, T) bool

        if run_mask.any():
            x_packed = x[run_mask]                   # (N_run, d_model)

            # Xử lý RoPE frequencies cho đúng subset token
            if freqs_cis.dim() >= 2:
                freqs_cis_packed = freqs_cis[run_mask]
            else:
                freqs_cis_packed = freqs_cis

            attn_mask_packed = None
            if attn_mask is not None:
                attn_mask_packed = attn_mask[run_mask]

            out[run_mask] = self._run_block(
                x_packed,
                freqs_cis_packed,
                attn_mask_packed,
            )

        if return_aux_loss:
            return out, x.new_zeros(())
        return out