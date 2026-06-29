"""
model/block.py — Transformer Block (LLaMA-style)
=================================================
Luồng:
    x → SelfAttentionRoPE (Pre-Norm bên trong) → residual → FFN (Pre-Norm) → out
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import SelfAttentionRoPE
from .layers import RMSNorm, SwiGLU

class MoDRouter(nn.Module):
    """
    Router cho Mixture of Depths — gồm 2 thành phần:
 
    1. main_router  : Linear d_model → 1, dùng khi TRAIN (top-k trên sequence)
    2. aux_router   : MLP nhỏ d_model → d_model//4 → 1, dùng khi INFERENCE
                      (predict per-token không cần future context)
 
    Tại sao cần 2 router:
        Top-k khi train nhìn toàn sequence (không causal) → không dùng được
        khi generate autoregressive từng token.
        Auxiliary router được train để predict decision của main router,
        chỉ dùng hidden state tại vị trí hiện tại (causal).
 
    Auxiliary router loss (binary cross-entropy):
        label = 1 nếu token được main router chọn (top-k), 0 nếu không
        loss  = BCE(aux_logit, label)
        → aux_router học "token nào thường được chọn" mà không cần future
    """
 
    def __init__(self, d_model: int):
        super().__init__()
 
        # Main router: scalar score mỗi token — đơn giản, theo paper
        self.main = nn.Linear(d_model, 1, bias=False)
 
        # Auxiliary router: MLP nhỏ — cần đủ capacity để predict decision
        # hidden = d_model // 4 theo paper (lightweight)
        hidden = max(d_model // 4, 16)
        self.aux = nn.Sequential(
            nn.Linear(d_model, hidden, bias=False),
            nn.SiLU(),
            nn.Linear(hidden, 1, bias=False),
        )
 
        # Init nhỏ để không át signal ban đầu
        nn.init.normal_(self.main.weight, std=0.02)
        for m in self.aux.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
 
    def forward_train(self, x: torch.Tensor, capacity: int):
        """
        Train path — top-k routing trên toàn sequence.
 
        Args:
            x        : (B, T, D) hidden states
            capacity : k = số token được phép đi qua block
 
        Returns:
            selected_mask : (B, T) bool — True = token này qua block
            gate_scores   : (B, T) float — weight nhân vào output (STE)
            aux_loss      : scalar — BCE loss của auxiliary router
        """
        B, T, D = x.shape
 
        # Main router scores — (B, T)
        main_scores = self.main(x).squeeze(-1)                 # (B, T)
 
        # Top-k: chọn capacity token có score cao nhất mỗi sequence
        # topk trả về (values, indices) — values là scores của top-k token
        topk_scores, topk_idx = main_scores.topk(
            k=min(capacity, T), dim=-1, sorted=False
        )                                                       # (B, k)
 
        # Tạo mask từ top-k indices
        selected_mask = torch.zeros(B, T, dtype=torch.bool, device=x.device)
        selected_mask.scatter_(1, topk_idx, True)              # (B, T) bool
 
        # Gate scores cho STE (Straight-Through Estimator):
        # Token được chọn → gate = sigmoid(score) ∈ (0,1)
        # Token không chọn → gate = 0
        # Gradient chảy qua gate → main router được train
        gate_scores = torch.zeros(B, T, device=x.device, dtype=torch.float32)
        gate_scores[selected_mask] = torch.sigmoid(
            main_scores[selected_mask].float()
        )
 
        # Auxiliary router loss — train aux để predict main's decision
        # Detach x để aux loss không ảnh hưởng feature learning
        aux_logits = self.aux(x.detach()).squeeze(-1)          # (B, T)
        aux_labels = selected_mask.float()                     # 1 nếu được chọn
        aux_loss   = F.binary_cross_entropy_with_logits(
            aux_logits, aux_labels,
        )
 
        return selected_mask, gate_scores, aux_loss
 
    def forward_infer(self, x: torch.Tensor, capacity: int):
        """
        Inference path — dùng auxiliary router (causal, per-token).
 
        Không cần future tokens → an toàn cho autoregressive generate.
        Threshold = 0.5 (sigmoid > 0.5 → predict "sẽ được chọn")
 
        Returns:
            selected_mask : (B, T) bool
            gate_scores   : (B, T) float — sigmoid của aux score
        """
        aux_logits    = self.aux(x).squeeze(-1)                # (B, T)
        aux_probs     = torch.sigmoid(aux_logits)              # (B, T)
 
        # Hard threshold — token nào aux_prob > 0.5 thì cho qua
        # Có thể tune threshold tại inference mà không cần retrain
        selected_mask = aux_probs > 0.5                        # (B, T) bool
 
        # Đảm bảo không vượt capacity (optional safety cap)
        # Nếu quá nhiều token được chọn, giữ top-capacity theo prob
        n_selected = selected_mask.sum(dim=-1)                 # (B,)
        for b in range(x.size(0)):
            if n_selected[b] > capacity:
                # Giữ top-capacity theo aux probability
                _, topk_idx = aux_probs[b].topk(capacity)
                selected_mask[b] = torch.zeros(
                    x.size(1), dtype=torch.bool, device=x.device
                )
                selected_mask[b].scatter_(0, topk_idx, True)
 
        gate_scores = torch.zeros_like(aux_probs)
        gate_scores[selected_mask] = aux_probs[selected_mask]
 
        return selected_mask, gate_scores

class TransformerBlock(nn.Module):
    """
    Transformer Block với Mixture of Depths (MoD) tùy chọn.
 
    use_mod=False : block gốc, không thay đổi gì (backward compatible)
    use_mod=True  : thêm MoDRouter, chỉ top-k token đi qua attn+ffn
 
    mod_interleave=True (khuyến nghị):
        Block chỉ thực sự apply MoD nếu layer_idx là số lẻ.
        Layer chẵn (0, 2, 4...) luôn chạy bình thường.
        Truyền layer_idx khi khởi tạo.
    """
    
    def __init__(
        self,
        d_model  : int,
        n_heads  : int,
        dropout  : float = 0.1,
        n_layers : int   = 8,
        layer_idx  : int   = 0,       # index của layer này trong model
        # ── MoD config ───────────────────────────────────────────────────
        use_mod       : bool  = False,
        mod_capacity  : float = 0.5,  # k = int(capacity * T), paper default 0.5
        mod_interleave: bool  = True,  # chỉ apply MoD ở layer lẻ
    ):
        super().__init__()
        self.d_model      = d_model
        self.layer_idx    = layer_idx
        self.mod_capacity = mod_capacity
 
        # MoD active = use_mod=True VÀ (interleave=False HOẶC layer lẻ)
        # Layer 0 luôn bình thường để giữ low-level features ổn định
        self.mod_active = (
            use_mod and
            (not mod_interleave or layer_idx % 2 == 1)
        )

        # Self-attention
        self.self_attn = SelfAttentionRoPE(d_model, n_heads, dropout=dropout)

        # FFN
        self.norm2 = RMSNorm(d_model)
        self.ffn   = SwiGLU(d_model, bias=False)
        
        # ── MoD Router — chỉ tạo khi layer này thực sự dùng MoD ─────────
        if self.mod_active:
            self.router = MoDRouter(d_model)

        self._scaled_init(n_layers)

    def _scaled_init(self, n_layers: int):
        """Scale init cho các projection nằm trên đường residual."""
        scale = 1.0 / math.sqrt(2 * n_layers)
        nn.init.normal_(self.self_attn.Wo.weight, std=0.02 * scale)
        nn.init.normal_(self.ffn.w2.weight,       std=0.02 * scale)

    def _run_block(
        self,
        x        : torch.Tensor,
        freqs_cis: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Chạy full block (attn + ffn) trên x — dùng nội bộ."""
        attn_out = self.self_attn(x, freqs_cis, attn_mask=attn_mask)
        x_post   = x + attn_out
        return x_post + self.ffn(self.norm2(x_post))
    
    def forward(
        self,
        x        : torch.Tensor,
        freqs_cis: torch.Tensor,
        attn_mask: torch.Tensor = None,
        return_aux_loss: bool         = False,   # True khi train với MoD
    ) -> torch.Tensor:
        """
        Returns:
            mod_active=False hoặc layer không có MoD:
                torch.Tensor (B, T, D)
 
            mod_active=True, return_aux_loss=True (train):
                (out, aux_loss)  — aux_loss scalar BCE
 
            mod_active=True, return_aux_loss=False (infer):
                torch.Tensor (B, T, D)
        """
        if not self.mod_active:
            # ── Path gốc: mọi token qua block bình thường ─────────────
            out = self._run_block(x, freqs_cis, attn_mask)
            if return_aux_loss:
                return out, torch.tensor(0.0, device=x.device)
            return out
        
        # ── MoD path ──────────────────────────────────────────────────
        B, T, D   = x.shape
        capacity  = max(1, int(self.mod_capacity * T))   # k token được chọn
 
        if self.training or return_aux_loss:
            # ── TRAIN: top-k routing trên toàn sequence ───────────────
            selected_mask, gate_scores, aux_loss = self.router.forward_train(
                x, capacity
            )
 
            # Chạy full block trên toàn bộ x — output có đủ (B, T, D)
            # Lý do không mask trước attn: causal attn cần full sequence
            # để position encoding (RoPE) đúng vị trí
            block_out = self._run_block(x, freqs_cis, attn_mask)   # (B, T, D)
 
            # Ghép output:
            # - Token được chọn: gate_score * block_out  (STE: gradient qua gate)
            # - Token không chọn: x nguyên (residual)
            gate = gate_scores.unsqueeze(-1).to(x.dtype)  # match dtype với x (float16/32)
            mask = selected_mask.unsqueeze(-1)         # (B, T, 1)
            out  = torch.where(mask, gate * block_out + (1 - gate) * x, x)
 
            if return_aux_loss:
                return out, aux_loss
            return out
 
        else:
            # ── INFERENCE: auxiliary router (causal, per-token) ───────
            selected_mask, gate_scores = self.router.forward_infer(x, capacity)
 
            if selected_mask.all():
                # Tất cả token qua block — không cần masking
                return self._run_block(x, freqs_cis, attn_mask)
 
            elif not selected_mask.any():
                # Không token nào qua block — trả nguyên x
                return x
 
            else:
                # Mixed: chạy block trên toàn bộ, sau đó select
                block_out = self._run_block(x, freqs_cis, attn_mask)
                gate = gate_scores.unsqueeze(-1).to(x.dtype)
                mask = selected_mask.unsqueeze(-1)
                return torch.where(mask, gate * block_out + (1 - gate) * x, x)