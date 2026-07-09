"""
model/lm.py — MemoryLM (LLaMA-style, không có Context Memory)
===============================================================
Kỹ thuật cốt lõi:
    - RMSNorm + Pre-Norm  : bên trong SelfAttentionRoPE và trước FFN
    - SwiGLU              : trong mỗi TransformerBlock
    - No bias             : toàn bộ Linear đều bias=False
    - RoPE                : áp lên Q/K trong self-attention, không có pos_emb tuyệt đối
    - Scaled init         : 1/sqrt(2*n_layers) cho projection trên đường residual
    - Weight tying        : lm_head.weight = token_emb.weight
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
    ):
        super().__init__()
        self.d_model  = d_model
        self.n_layers = n_layers
        self.max_seq  = max_seq

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.drop      = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout, n_layers=n_layers)
            for _ in range(n_layers)
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

    def forward(self, input_ids: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        """
        input_ids : (B, T)
        Returns   : logits (B, T, vocab_size)
        """
        B, T   = input_ids.shape
        device = input_ids.device

        x         = self.drop(self.token_emb(input_ids))
        freqs_cis = self.freqs_cis.to(device)

        for block in self.blocks:
            x = block(x, freqs_cis=freqs_cis, attn_mask=attn_mask)

        return self.lm_head(self.norm_out(x))


def causal_mask(T: int, device: torch.device) -> torch.Tensor:
    """Causal mask additive cho autoregressive attention."""
    mask = torch.triu(torch.ones(T, T, device=device), diagonal=1)
    return mask.masked_fill(mask.bool(), float("-inf"))

def make_span_noise_mask(
    T: int,
    device: torch.device,
    noise_ratio: float = 0.05,
    batch_size: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Sinh ma trận nhiễu ngẫu nhiên [T, T] hoặc [batch_size, T, T] (bool).
    True = vị trí bị che khỏi attention.
 
    - Mỗi ô được chọn độc lập với xác suất `noise_ratio`.
    - Đường chéo luôn được loại trừ (không tự che chính mình) ngay tại bước
      sinh mask này (dù get_combined_mask cũng sẽ tự cưỡng chế lại lần nữa,
      đây là lớp bảo vệ đầu tiên).
    - Phần j >= i (tương lai) có thể bị chọn ngẫu nhiên nhưng KHÔNG SAO,
      vì causal mask ở get_combined_mask() sẽ tự động che phần đó rồi,
      OR hai mask không bị ảnh hưởng bởi phần dư thừa này.
 
    Args:
        T: độ dài chuỗi.
        device: thiết bị tensor (cpu/cuda).
        noise_ratio: tỉ lệ (xác suất) mỗi vị trí quá khứ bị che.
                     Khuyến nghị: < 0.05 (dưới 5%).
        batch_size: nếu muốn mỗi sample trong batch có nhiễu độc lập khác
                    nhau -> trả về [B, T, T]. None -> trả về [T, T] dùng
                    chung cho cả batch.
        generator: torch.Generator để cố định seed (tái lập kết quả khi cần).
 
    Returns:
        bool tensor, shape [T, T] hoặc [batch_size, T, T].
    """
    if not (0.0 <= noise_ratio <= 1.0):
        raise ValueError(f"noise_ratio phải trong khoảng [0, 1], nhận: {noise_ratio}")
    if noise_ratio > 0.05:
        import warnings
        warnings.warn(
            f"noise_ratio={noise_ratio} vượt quá khuyến nghị 5%. "
            f"Tỉ lệ càng cao càng dễ làm mất ngữ cảnh quan trọng, "
            f"đặc biệt với các chuỗi ngắn.",
            stacklevel=2,
        )
 
    shape = (batch_size, T, T) if batch_size is not None else (T, T)
    rand_matrix = torch.rand(shape, device=device, generator=generator)
    span_mask = rand_matrix < noise_ratio
 
    diag_idx = torch.arange(T, device=device)
    span_mask[..., diag_idx, diag_idx] = False  # không tự che chính mình
 
    return span_mask
 
 
def get_combined_mask(
    T: int,
    span_mask_matrix: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    strict_check: bool = False,
) -> torch.Tensor:
    """
    Kết hợp causal mask + span noise mask thành additive attention mask
    (dùng bằng cách: attn_scores + mask, rồi softmax).
 
    Args:
        T: độ dài chuỗi.
        span_mask_matrix: bool tensor [T, T] hoặc [B, T, T]. True = bị che.
                          Có thể lấy từ make_span_noise_mask(), hoặc tự tạo.
        device: thiết bị tensor.
        dtype: dtype của mask trả về, nên khớp dtype của model
               (fp32/fp16/bf16) để tránh sai lệch số học khi cộng vào scores.
        strict_check: nếu True, kiểm tra thêm "có hàng nào bị che 100%
                      không" và raise lỗi nếu có (tốn thêm 1 lượt tính
                      O(T^2) mỗi lần gọi). Nên BẬT khi debug/viết unit test,
                      TẮT trong training loop thật (hot path) một khi đã
                      tin tưởng qua test — vì bước cưỡng chế mở đường chéo
                      bên dưới đã đảm bảo nhánh lỗi này không bao giờ có thể
                      kích hoạt được nữa trong điều kiện bình thường.
 
    Returns:
        additive mask, cùng shape với span_mask_matrix (broadcast causal vào),
        giá trị 0 (được attend) hoặc rất âm (bị che).
    """
    if span_mask_matrix.dtype != torch.bool:
        raise TypeError(
            f"span_mask_matrix phải là bool tensor, nhận: {span_mask_matrix.dtype}"
        )
    if span_mask_matrix.shape[-2:] != (T, T):
        raise ValueError(
            f"span_mask_matrix phải có 2 chiều cuối [T,T]=({T},{T}), "
            f"nhận: {span_mask_matrix.shape}"
        )
 
    causal_bool = torch.triu(
        torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1
    )
    final_mask_bool = causal_bool | span_mask_matrix.to(device)
 
    # Chốt an toàn thực sự: mọi token luôn tự attend được vào chính nó.
    # Đây là lý do NaN không thể xảy ra, bất kể span_mask_matrix có che gì.
    diag_idx = torch.arange(T, device=device)
    final_mask_bool[..., diag_idx, diag_idx] = False
 
    if strict_check:
        if final_mask_bool.all(dim=-1).any():
            raise RuntimeError(
                "Phát hiện hàng bị che 100% (sẽ gây NaN sau softmax). "
                "Kiểm tra lại logic sinh span_mask_matrix."
            )
 
    neg_val = torch.finfo(dtype).min  # an toàn hơn -inf tuyệt đối với fp16/bf16
    out = torch.zeros(final_mask_bool.shape, device=device, dtype=dtype)
    return out.masked_fill(final_mask_bool, neg_val)

def build_model(cfg) -> MemoryLM:
    """Entry point xây model từ ModelConfig."""
    return MemoryLM(
        vocab_size = cfg.model.vocab_size,
        d_model    = cfg.model.d_model,
        n_heads    = cfg.model.n_heads,
        n_layers   = cfg.model.n_layers,
        max_seq    = cfg.model.max_seq,
        dropout    = cfg.model.dropout,
        rope_base  = getattr(cfg.model, "rope_base", 10000.0),
    )