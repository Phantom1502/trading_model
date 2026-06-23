"""
utils/checkpoint.py — Save/Load/Resume checkpoint
=====================================================
"""

import os
import torch


def save_checkpoint(
    path        : str,
    model       : torch.nn.Module,
    optimizer   : torch.optim.Optimizer = None,
    scheduler                            = None,
    global_step  : int = 0,
    chunk_idx     : int = 0,
    val_loss       : float = None,
    extra           : dict = None,
    model_cfg        = None,   # ModelConfig dùng để build lại đúng kiến trúc khi load
):
    """
    Lưu toàn bộ state cần thiết để resume training HOẶC load lại đúng
    kiến trúc cho inference.

    QUAN TRỌNG: model_cfg PHẢI được truyền vào nếu kiến trúc model có thể
    thay đổi giữa các lần train (ví dụ use_memory=False, num_slots khác,
    d_model khác...). Thiếu model_cfg sẽ khiến load_model_for_inference()
    build model với config mặc định, gây lỗi "Missing key(s) in state_dict"
    hoặc "Unexpected key(s)" khi kiến trúc lúc train và lúc load khác nhau.
    """
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    ckpt = {
        "model"      : model.state_dict(),
        "global_step": global_step,
        "chunk_idx"  : chunk_idx,
        "val_loss"   : val_loss,
    }
    if optimizer is not None:
        ckpt["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        ckpt["scheduler"] = scheduler.state_dict()
    if extra:
        ckpt["extra"] = extra
    if model_cfg is not None:
        # Lưu dạng dict (dataclasses.asdict) để không phụ thuộc việc
        # import lại đúng class ModelConfig khi load ở môi trường khác
        from dataclasses import asdict
        ckpt["model_cfg"] = asdict(model_cfg)

    torch.save(ckpt, path)
    print(f"  ✓ Saved checkpoint → {path}")


def load_checkpoint(
    path        : str,
    model       : torch.nn.Module,
    optimizer   : torch.optim.Optimizer = None,
    scheduler                            = None,
    device                                = "cpu",
) -> dict:
    """
    Load checkpoint, trả về dict chứa global_step, chunk_idx, val_loss
    để caller tiếp tục training đúng chỗ.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])

    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])

    print(f"  ✓ Loaded checkpoint ← {path}")
    print(f"    global_step={ckpt.get('global_step', 0)} | "
          f"chunk_idx={ckpt.get('chunk_idx', 0)} | "
          f"val_loss={ckpt.get('val_loss')}")

    return {
        "global_step": ckpt.get("global_step", 0),
        "chunk_idx"  : ckpt.get("chunk_idx", 0),
        "val_loss"   : ckpt.get("val_loss"),
        "extra"      : ckpt.get("extra"),
        "model_cfg"  : ckpt.get("model_cfg"),   # dict hoặc None nếu checkpoint cũ không có
    }