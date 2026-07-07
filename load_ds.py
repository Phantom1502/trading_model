from app.llama import tokenizer as llama_tokenizer
from app.llama import config as llama_config
from app.llama import model as llama_model
from app.llama import train as llama_train

model_cfg = llama_config.get_small_llama_config()

cfg = llama_config.get_default_config()
cfg.llama = model_cfg
cfg.tokenizer.base_tokenizer_dir = "custom_tokenizer"
cfg.tokenizer.output_dir = "custom_tokenizer_llama"
cfg.tokenizer.n_price_bins = 1024

cfg.data.source = "parquet"
cfg.data.parquet_path = r"E:\LLM Dataset\final\mix_suffle_1.parquet"
cfg.data.parquet_text_col = "text"

llama_train.main(cfg)