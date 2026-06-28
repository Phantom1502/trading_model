from app.memlm.train import main
from app.memlm.config import get_small_config

cfg = get_small_config()


# Config Data
cfg.data.source = "parquet"
cfg.data.parquet_path = "data/processed/mix_part_0001.parquet"
cfg.data.parquet_text_col = "text"
cfg.data.sequential_mode = True
cfg.data.window_stride   = cfg.data.seg_len  # No Overlap
cfg.data.chunk_size = 20_000

# Config logs
cfg.train.eval_every = 99_999_999
cfg.train.save_every = 99_999_999

# Config Save Dir
cfg.train.save_dir = "data/check_point"

main(cfg)
#main(cfg, start_chunk=start_chunk)
#main(cfg, start_chunk=0, reset_lr_for_new_round=True)