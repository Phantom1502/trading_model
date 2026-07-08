from dataclasses import dataclass, field
from typing import Optional

@dataclass
class TokenizerConfig:
    """Cấu hình tokenizer."""
    pretrained_name  : str  = "custom_tokenizer_llama"
    use_fast         : bool = True
    
    
@dataclass
class ModelConfig:
    """Cấu hình kiến trúc model."""
    vocab_size : int = 24000
    hidden_size : int = 576
    intermediate_size : int = 1536
    num_hidden_layers : int = 30
    num_attention_heads : int = 9
    num_key_value_heads : int = 3
    max_position_embeddings : int = 1024
    tie_word_embeddings : bool = True
    
@dataclass
class TrainConfig:
    """Cấu hình quá trình train."""
    
    device          : str  = "cuda"    
    
@dataclass
class DataConfig:
    """Cấu hình dữ liệu."""
    val_cache_path : Optional[str] = None  
    
@dataclass
class HubConfig:
    """Cấu hình huggingface_hub."""
    repo_id : Optional[str] = None
    hf_token: Optional[str] = None
    
@dataclass
class Config:
    """Gộp tất cả config con."""
    tokenizer : TokenizerConfig = field(default_factory=TokenizerConfig)
    model     : ModelConfig     = field(default_factory=ModelConfig)
    train     : TrainConfig = field(default_factory=TrainConfig)
    data      : DataConfig      = field(default_factory=DataConfig)
    hub       : HubConfig   = field(default_factory=HubConfig)
    seed      : int             = 42
    