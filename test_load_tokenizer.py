from app.memlm.tokenizer import VietnameseTokenizer


import os

base_dir = os.path.dirname(os.path.abspath(__file__))
tok_path = os.path.join(base_dir, "app", "memlm", "custom_tokenizer")
print(f"Testing tokenizer: {tok_path}\n")

tok = VietnameseTokenizer(pretrained_name=tok_path)
print(f"Vocab size (BPE base + price): {tok.vocab_size:,}")
print(f"  BPE base  : {len(tok.tokenizer):,}")
print(f"  Price vocab: {len(tok.price_vocab):,}")
print()
    
