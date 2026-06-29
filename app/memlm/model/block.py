import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import SelfAttentionRoPE
from .layers import RMSNorm, SwiGLU

class MoDRouter(nn.Module):
    """
    Router chuẩn hóa theo Paper Mixture of Depths (MoD) của Google DeepMind.
    Tích hợp chế độ Top-k khi Train và Ngưỡng tĩnh (Threshold) khi Infer.
    """
    def __init__(self, d_model: int, inference_threshold: float = 0.5):
        super().__init__()
        self.router_weights = nn.Linear(d_model, 1, bias=False)
        nn.init.normal_(self.router_weights.weight, std=0.02)
        self.inference_threshold = inference_threshold

    def forward(self, x: torch.Tensor):
        # Tính toán điểm số định tuyến thô cho từng token: (B, T)
        router_logits = self.router_weights(x).squeeze(-1)
        # Sử dụng Sigmoid theo đúng chuẩn paper MoD (không dùng Softmax xuyên token)
        gate_scores = torch.sigmoid(router_logits) 
        return gate_scores


class TransformerBlock(nn.Module):
    """
    Transformer Block tích hợp Mixture of Depths (MoD) chuẩn Paper.
    Hỗ trợ tăng tốc thực tế cho cả giai đoạn Train (Nén Sequence) và Infer (Nén Batch).
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        n_layers: int = 8,
        layer_idx: int = 0,
        use_mod: bool = False,
        mod_capacity: float = 0.5,
        mod_interleave: bool = True,
        inference_threshold: float = 0.5
    ):
        super().__init__()
        self.d_model = d_model
        self.layer_idx = layer_idx
        self.mod_capacity = mod_capacity
        self.mod_active = use_mod and (not mod_interleave or layer_idx % 2 == 1)

        self.self_attn = SelfAttentionRoPE(d_model, n_heads, dropout=dropout)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, bias=False)

        if self.mod_active:
            self.router = MoDRouter(d_model, inference_threshold=inference_threshold)

        self._scaled_init(n_layers)

    def _scaled_init(self, n_layers: int):
        scale = 1.0 / math.sqrt(2 * n_layers)
        nn.init.normal_(self.self_attn.Wo.weight, std=0.02 * scale)
        nn.init.normal_(self.ffn.w2.weight, std=0.02 * scale)

    def _run_block(self, x: torch.Tensor, freqs_cis: torch.Tensor, attn_mask: torch.Tensor, passed_mask: torch.Tensor = None) -> torch.Tensor:
        """Thực thi lõi tính toán Attention và FeedForward."""
        # LƯU Ý: Nếu self_attn có KV Cache nội bộ, bạn cần chuyển passed_mask vào trong 
        # để lớp Attention biết cách cập nhật chính xác các slot cache tương ứng trong Batch gốc.
        attn_out = self.self_attn(x, freqs_cis, attn_mask=attn_mask)
        x_post = x + attn_out
        return x_post + self.ffn(self.norm2(x_post))

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attn_mask: torch.Tensor = None,
        return_aux_loss: bool = False,
    ) -> torch.Tensor:
        
        # Nếu tầng này không kích hoạt MoD, chạy bypass thông thường
        if not self.mod_active:
            out = self._run_block(x, freqs_cis, attn_mask)
            return (out, torch.tensor(0.0, device=x.device, dtype=x.dtype)) if return_aux_loss else out

        B, T, D = x.shape
        gate_scores = self.router(x) # (B, T)

        # 🚀 KỊCH BẢN 1: INFERENCE TRÊN TOKEN ĐƠN (Autoregressive Generation, T = 1)
        if not self.training and T == 1:
            passed_mask = (gate_scores.squeeze(-1) >= self.router.inference_threshold) # (B,)
            
            out = x.clone()
            if passed_mask.any():
                # Co cụm Batch (Batch Collapsing): Nén từ (B, 1, D) -> (N, 1, D) với N <= B
                x_packed = x[passed_mask].unsqueeze(1)
                
                # Trích xuất chính xác RoPE tương ứng cho các phần tử được chọn
                if freqs_cis.dim() == 4:     # (B, 1, 1, H) -> (N, 1, 1, H)
                    freqs_cis_packed = freqs_cis[passed_mask]
                elif freqs_cis.dim() == 3:   # (B, 1, H) -> (N, 1, H)
                    freqs_cis_packed = freqs_cis[passed_mask]
                else:                        # (1, H) độc lập batch -> (1, H)
                    freqs_cis_packed = freqs_cis
                
                attn_mask_packed = attn_mask[passed_mask] if attn_mask is not None else None
                
                # Tính toán trên N phần tử đạt ngưỡng
                block_out_packed = self._run_block(x_packed, freqs_cis_packed, attn_mask_packed, passed_mask=passed_mask)
                
                # Áp dụng trọng số Gate chuẩn công thức paper: y = x + s * f(x)
                gate_packed = gate_scores[passed_mask].unsqueeze(-1).unsqueeze(-1) # (N, 1, 1)
                delta_packed = (block_out_packed - x_packed) * gate_packed # Shape: (N, 1, D)
                
                # 🔥 SỬA LỖI TẠI ĐÂY: Không sử dụng .squeeze(1) để tránh làm lệch luật Broadcasting của PyTorch
                out[passed_mask] += delta_packed
                
            aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
            return (out, aux_loss) if return_aux_loss else out

        # 🚀 KỊCH BẢN 2: TRAINING HOẶC PREFILL PROMPT (T > 1)
        capacity = max(1, int(self.mod_capacity * T))
        k = min(capacity, T)
        
        # Bước A: Lấy ra Top-k token có điểm cao nhất
        _, topk_idx = gate_scores.topk(k=k, dim=-1, sorted=False) # (B, K)
        
        # Sắp xếp lại index theo chiều tăng dần của thời gian bảo toàn tính Causal và RoPE
        topk_idx, _ = torch.sort(topk_idx, dim=-1)
        
        # ── 1. TIẾN HÀNH PACKING TENSORS (NÉN THEO SEQUENCE) ───────────────────
        gather_idx = topk_idx.unsqueeze(-1).expand(-1, -1, D)
        x_packed = torch.gather(x, 1, gather_idx) # (B, K, D)

        # Trích xuất chính xác tọa độ RoPE tương ứng
        if freqs_cis.dim() == 4:     # Định dạng (B, 1, T, H)
            gather_idx_rope = topk_idx.unsqueeze(1).unsqueeze(-1).expand(-1, 1, -1, freqs_cis.size(-1))
            freqs_cis_packed = torch.gather(freqs_cis, 2, gather_idx_rope)
        elif freqs_cis.dim() == 3:   # Định dạng (B, T, H)
            gather_idx_rope = topk_idx.unsqueeze(-1).expand(-1, -1, freqs_cis.size(-1))
            freqs_cis_packed = torch.gather(freqs_cis, 1, gather_idx_rope)
        else:                        # Định dạng độc lập batch (T, H)
            freqs_cis_packed = freqs_cis[topk_idx]

        # ── 2. NÉN ATTENTION MASK VÀ BẢO TOÀN CAUSAL STRUCTURE ─────────────────
        attn_mask_packed = None
        if attn_mask is not None:
            # 🔥 SỬA LỖI TẠI ĐÂY: Mở rộng mask nếu nó ở dạng broadcast dạng (1, 1, T, T) để tránh crash với torch.gather
            if attn_mask.size(0) == 1 and B > 1:
                attn_mask = attn_mask.expand(B, -1, -1, -1)

            # Lọc theo hàng (Dim 2) -> (B, 1, K, T)
            row_idx = topk_idx.unsqueeze(1).unsqueeze(-1).expand(-1, 1, -1, T)
            mask_rows = torch.gather(attn_mask, 2, row_idx)
            
            # Lọc theo cột (Dim 3) từ kết quả trên -> (B, 1, K, K)
            col_idx = topk_idx.unsqueeze(1).unsqueeze(2).expand(-1, 1, k, -1)
            attn_mask_packed = torch.gather(mask_rows, 3, col_idx)

        # ── 3. THỰC THI CORE BLOCK TIẾT KIỆM TÀI NGUYÊN ──────────────────────
        block_out_packed = self._run_block(x_packed, freqs_cis_packed, attn_mask_packed)

        # ── 4. GATING & UNPACKING (GIẢI NÉN VỀ TENSOR GỐC) ────────────────────
        gate_packed = torch.gather(gate_scores, 1, topk_idx).unsqueeze(-1).to(x.dtype) # (B, K, 1)
        
        layer_delta_packed = block_out_packed - x_packed
        gated_delta_packed = gate_packed * layer_delta_packed # (B, K, D)

        # Rải ngược (Scatter) dữ liệu biến đổi về đúng vị trí ban đầu trên Tensor (B, T, D)
        delta_full = torch.zeros_like(x)
        delta_full.scatter_(1, gather_idx, gated_delta_packed)
        
        out = x + delta_full 

        # Trả về kèm một Aux Loss bằng 0 theo đúng tinh thần tinh giản của MoD Paper
        aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        return (out, aux_loss) if return_aux_loss else out