"""
model.py — Xây LlamaForCausalLM cho pipeline pretrain LLaMA tiếng Việt (app/llama)
====================================================================================
Tách từ hàm build_model() trong bản nháp train.py. Logic kiến trúc giữ NGUYÊN
(SmolLM-135M style) — chỉ đổi 2 chỗ hardcode trong nháp thành đọc từ config:

    1. attn_implementation="sdpa" hardcode -> cfg.model.attn_implementation
       (đã benchmark sdpa nhanh hơn eager ~3 lần, xem readme.md mục 1, nhưng
       giờ có thể đổi thử "eager"/"flash_attention_2" qua config mà không sửa code)
    2. pad_token_id=3, bos_token_id=1, eos_token_id=2 hardcode -> cfg.model.*
       (xem tokenizer.py::check_tokenizer_matches_model_config — nên gọi hàm
       đó TRƯỚC build_model() để chắc chắn 3 giá trị này khớp tokenizer thật)
"""

from transformers import LlamaConfig, LlamaForCausalLM

from app.llama.config import Config


def build_llama_config(cfg: Config) -> LlamaConfig:
    """Tách riêng việc build LlamaConfig (thuần data, không tốn compute) khỏi
    việc khởi tạo trọng số model — hữu ích khi chỉ cần xem/log kiến trúc mà
    chưa muốn cấp phát tensor (vd in số param ước tính trước khi build thật)."""
    m = cfg.model
    return LlamaConfig(
        vocab_size=m.vocab_size,
        hidden_size=m.hidden_size,
        intermediate_size=m.intermediate_size,
        num_hidden_layers=m.num_hidden_layers,
        num_attention_heads=m.num_attention_heads,
        num_key_value_heads=m.num_key_value_heads,
        max_position_embeddings=m.max_position_embeddings,
        pad_token_id=m.pad_token_id,
        bos_token_id=m.bos_token_id,
        eos_token_id=m.eos_token_id,
        tie_word_embeddings=m.tie_word_embeddings,
    )


def build_model(cfg: Config) -> LlamaForCausalLM:
    """
    Entry point chính — build model TỪ ĐẦU (random init), KHÔNG load checkpoint
    (đúng hành vi bản nháp: `LlamaForCausalLM._from_config(...)`, không phải
    `.from_pretrained(...)`).

    LƯU Ý THỨ TỰ GỌI: cfg.model.vocab_size PHẢI được set đúng (qua
    tokenizer.sync_vocab_size()) TRƯỚC khi gọi hàm này — build_model() không
    tự kiểm tra khớp tokenizer, dùng thẳng cfg.model.vocab_size hiện có.
    """
    llama_cfg = build_llama_config(cfg)
    model = LlamaForCausalLM._from_config(
        llama_cfg,
        attn_implementation=cfg.model.attn_implementation,
    )
    return model


def count_params(model: LlamaForCausalLM) -> dict:
    """Tiện ích nhỏ hay cần khi log/benchmark — tổng param + phần embedding
    (thường chiếm tỉ lệ lớn ở model nhỏ do tie_word_embeddings)."""
    total = sum(p.numel() for p in model.parameters())
    embedding = model.get_input_embeddings().weight.numel()
    return {
        "total": total,
        "embedding": embedding,
        "non_embedding": total - embedding,
    }