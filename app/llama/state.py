"""
state.py — Tiến độ shard (resume giữa các session Colab) cho app/llama
================================================================================
Tách từ train.py nháp: load_state(), save_state(), ensure_dirs().

State chỉ lưu ĐÚNG 2 giá trị (giữ nguyên từ nháp):
    "shard_index"                 : shard tiếp theo cần xử lý
    "last_completed_shard_ckpt_step": step của checkpoint được lưu NGAY SAU
                                      KHI shard trước đó hoàn thành — dùng để
                                      phân biệt "resume giữa chừng 1 shard"
                                      với "bắt đầu shard mới" (xem train.py
                                      nháp, mục 6 — logic is_fresh_shard_start).

QUAN TRỌNG: chỉ cập nhật state SAU KHI 1 shard train xong HOÀN TOÀN (readme.md
mục 5.4) — hàm save_state() ở đây không tự enforce điều đó, caller (train.py)
phải tự gọi đúng thời điểm.
"""

import json
import os
from typing import Optional

from accelerate import PartialState

from app.llama.config import Config


def ensure_dirs(cfg: Config) -> None:
    """Tạo trước các thư mục output/cache cần thiết — tránh lỗi
    FileNotFoundError khi save_checkpoint/save_to_disk lần đầu."""
    os.makedirs(cfg.train.output_dir, exist_ok=True)
    os.makedirs(cfg.data.cache_dir, exist_ok=True)
    os.makedirs(cfg.data.val_cache_dir, exist_ok=True)


def load_state(cfg: Config) -> dict:
    """Đọc state từ cfg.train.state_path. Chưa có file (lần chạy đầu tiên)
    -> trả về state mặc định {"shard_index": 0}."""
    if os.path.exists(cfg.train.state_path):
        return json.load(open(cfg.train.state_path))
    return {"shard_index": 0}


def save_state(cfg: Config, state: dict) -> None:
    """
    QUAN TRỌNG (bug thật đã lường trước, chưa gặp nhưng SẼ gặp trên TPU):
    khi chạy multi-process (vd 8 core TPU qua accelerate.notebook_launcher),
    hàm main() được TÁM process cùng gọi song song — nếu save_state() ghi
    file vô điều kiện, cả 8 process sẽ cùng ghi đè shard_state.json cùng
    lúc, dễ tạo file JSON hỏng (ghi dở dang, xen kẽ). CHỈ process chính
    (rank 0) mới được ghi; các process còn lại no-op — chúng vẫn cần giá trị
    trả về của mark_shard_completed() để tiếp tục vòng lặp đúng, chỉ không
    tự ghi ra đĩa (xem mark_shard_completed bên dưới)."""
    if not PartialState().is_main_process:
        return
    json.dump(state, open(cfg.train.state_path, "w"))


def mark_shard_completed(cfg: Config, shard_index: int, final_ckpt_path: Optional[str]) -> dict:
    """
    Cập nhật + lưu state SAU KHI shard `shard_index` train xong hoàn toàn.

    `final_ckpt_path`: kết quả của `get_last_checkpoint(cfg.train.output_dir)`
    NGAY SAU khi trainer.train() của shard này kết thúc — dùng để trích step,
    lưu lại làm mốc "last_completed_shard_ckpt_step" cho lần chạy tiếp theo
    phân biệt đúng resume-giữa-chừng vs shard-mới (xem docstring module).
    """
    state = load_state(cfg)
    state["shard_index"] = shard_index + 1
    state["last_completed_shard_ckpt_step"] = (
        int(final_ckpt_path.split("-")[-1]) if final_ckpt_path else None
    )
    save_state(cfg, state)
    return state


def is_fresh_shard_start(state: dict, local_ckpt: Optional[str]) -> bool:
    """
    True nếu checkpoint tìm được (`local_ckpt`) chính là checkpoint đã lưu
    NGAY SAU KHI shard TRƯỚC hoàn thành — nghĩa là shard hiện tại chưa có
    step riêng nào, KHÔNG nên resume_from_checkpoint (Trainer sẽ hiểu nhầm
    global_step đã vượt max_steps của "shard mới" rồi dừng ngay lập tức).

    False (nhưng local_ckpt vẫn tồn tại) nghĩa là checkpoint MỚI HƠN mốc đã
    lưu — tức đã train dở shard hiện tại rồi bị ngắt -> nên resume bình thường.

    Giữ nguyên đúng logic so sánh trong train.py nháp (mục 6), chỉ tách hàm
    riêng để test độc lập được (logic này dễ sai lệch, nên đáng có unit test).
    """
    if local_ckpt is None:
        return False
    ckpt_step = int(local_ckpt.split("-")[-1])
    return state.get("last_completed_shard_ckpt_step") == ckpt_step