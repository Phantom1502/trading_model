"""
train.py — Entry point chạy pretrain
========================================
Usage trên Colab:

    !pip install transformers datasets accelerate -q
    !python train.py

Hoặc trong notebook cell:

    from config import get_100m_config
    from train import main
    main(get_100m_config())
"""

import torch

from config import get_100m_config, get_small_config
from tokenizer import load_tokenizer
from dataset import ChunkedWikiLoader, ChunkedVTSNLPLoader
from model import build_model
from trainer import run_pretrain


def main(cfg=None, start_chunk: int = 0, reset_lr_for_new_round: bool = False):
    """
    Args:
        cfg        : Config object — LUÔN dùng config NÀY để build model và
                     train, KHÔNG tự động lấy lại config cũ từ checkpoint dù
                     có resume_from. Đây là điểm quan trọng cần hiểu rõ:

    ────────────────────────────────────────────────────────────────────────
    TRAIN ROUND MỚI VỚI CONFIG KHÁC — 2 trường hợp:

    Trường hợp 1 — chỉ đổi HYPERPARAMETER TRAIN (lr, warmup, chunk_size,
    lr_decay_cycle_steps, batch_size...), giữ NGUYÊN kiến trúc model
    (d_model, n_layers, num_slots, use_memory...):

        cfg = get_100m_config()
        cfg.train.resume_from = "checkpoints/chunk_33.pt"
        cfg.train.lr = 1e-4                    # đổi lr cho round mới
        cfg.train.lr_decay_cycle_steps = 2000  # đổi chu kỳ decay
        # cfg.model giữ NGUYÊN giống lúc train checkpoint cũ
        main(cfg, start_chunk=34, reset_lr_for_new_round=True)
        #                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^
        #  BẮT BUỘC True — nếu không, optimizer.load_state_dict() sẽ phục
        #  hồi NGUYÊN lr cũ từ checkpoint, cfg.train.lr mới sẽ KHÔNG có
        #  hiệu lực (đã verify bug này bằng thực nghiệm — xem chi tiết
        #  trong docstring của run_pretrain() ở trainer/pretrain.py).

    Trường hợp 2 — đổi KIẾN TRÚC MODEL (num_slots, d_model, use_memory...):

        KHÔNG THỂ resume_from checkpoint cũ — state_dict sẽ lỗi
        "size mismatch" hoặc "Missing/Unexpected key(s)" vì shape các layer
        không khớp. Phải bắt đầu round mới HOÀN TOÀN TỪ ĐẦU:

        cfg = get_100m_config()
        cfg.model.num_slots = 8        # đổi kiến trúc
        cfg.train.resume_from = None   # BẮT BUỘC None — không thể resume
        cfg.train.save_dir = "checkpoints_round2"  # khuyên đổi thư mục
                                                      # để không ghi đè round 1
        main(cfg, start_chunk=0)       # train lại từ chunk 0

    ────────────────────────────────────────────────────────────────────────

    ────────────────────────────────────────────────────────────────────────
    ĐỔI NGUỒN DỮ LIỆU (Wikipedia → VTSNLP curated dataset):

        cfg = get_100m_config()
        cfg.data.source = "vtsnlp"
        # tùy chọn: chỉ lấy một số domain thay vì toàn bộ 25 domain
        cfg.data.vtsnlp_domains = ["Science", "Books_and_Literature"]
        main(cfg)

    Lưu ý: đổi source giữa các round train là đổi PHÂN PHỐI DỮ LIỆU, không
    phải đổi kiến trúc — vẫn RESUME ĐƯỢC bình thường từ checkpoint cũ nếu
    muốn (model không quan tâm dữ liệu đến từ nguồn nào, chỉ cần cùng
    tokenizer/vocab). Nhưng nên cân nhắc: trộn dữ liệu khác phân phối giữa
    các round có thể ảnh hưởng đến tính nhất quán của quá trình học.
    ────────────────────────────────────────────────────────────────────────

        start_chunk: chunk muốn bắt đầu/tiếp tục train (0 = từ đầu dataset).
    """
    if cfg is None:
        cfg = get_100m_config()

    torch.manual_seed(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.train.device = device
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU : {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

    # ── Tokenizer ──────────────────────────────────────────────────────────
    print("\n── Loading tokenizer (PhoBERT) ──")
    tokenizer = load_tokenizer(cfg)
    cfg.model.vocab_size = tokenizer.vocab_size
    print(f"Vocab size: {tokenizer.vocab_size}")

    # ── Model ──────────────────────────────────────────────────────────────
    print("\n── Building model ──")
    model = build_model(cfg)
    print(f"Total params     : {model.num_params()/1e6:.1f}M")
    print(f"Trainable params : {model.num_params(trainable_only=True)/1e6:.1f}M")

    # ── Data loader (incremental, RAM-safe) ──────────────────────────────────
    print("\n── Setting up incremental data loader ──")
    print(f"Source: {cfg.data.source} | Chunk size: {cfg.data.chunk_size} sample/chunk")
    if start_chunk > 0:
        print(f"Bắt đầu từ chunk {start_chunk} (skip {start_chunk * cfg.data.chunk_size:,} sample đầu)")

    if cfg.data.source == "vtsnlp":
        if cfg.data.vtsnlp_domains:
            print(f"Lọc domain: {cfg.data.vtsnlp_domains}")
        data_loader_gen = ChunkedVTSNLPLoader(
            cfg, tokenizer, start_chunk=start_chunk, domains=cfg.data.vtsnlp_domains
        )
    elif cfg.data.source == "wikipedia":
        data_loader_gen = ChunkedWikiLoader(cfg, tokenizer, start_chunk=start_chunk)
    else:
        raise ValueError(
            f"cfg.data.source='{cfg.data.source}' không hợp lệ — "
            f"chỉ hỗ trợ 'wikipedia' hoặc 'vtsnlp'"
        )

    # ── Train ──────────────────────────────────────────────────────────────
    print("\n── Starting pretraining ──")
    trainer = run_pretrain(cfg, model, tokenizer, data_loader_gen,
                            start_chunk=start_chunk,
                            reset_lr_for_new_round=reset_lr_for_new_round)

    return trainer


if __name__ == "__main__":
    main()
