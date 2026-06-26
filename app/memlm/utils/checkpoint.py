# ══════════════════════════════════════════════════════════════════════════
# PATCH 2: checkpoint.py — thêm file_order vào save/load
# Chỉ sửa 2 hàm save_checkpoint và load_checkpoint
# ══════════════════════════════════════════════════════════════════════════
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
    model_cfg        = None,
    file_order       : dict = None,   # NEW: file order của ChunkedMixLoader
):
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
        from dataclasses import asdict
        ckpt["model_cfg"] = asdict(model_cfg)
    if file_order is not None:
        ckpt["file_order"] = file_order   # NEW

    torch.save(ckpt, path)
    print(f"  ✓ Saved checkpoint → {path}")


def load_checkpoint(
    path        : str,
    model       : torch.nn.Module,
    optimizer   : torch.optim.Optimizer = None,
    scheduler                            = None,
    device                                = "cpu",
) -> dict:
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
        "model_cfg"  : ckpt.get("model_cfg"),
        "file_order" : ckpt.get("file_order"),   # NEW — None nếu checkpoint cũ
    }