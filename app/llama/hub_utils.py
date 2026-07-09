"""
hub_utils.py — Backup & resume qua Hugging Face Hub cho app/llama
================================================================================
Tách từ train.py nháp: phần login trong main(), tokenizer.push_to_hub(...),
và resume_checkpoint_from_hub_if_needed().

Vai trò: lớp backup thứ 2 ngoài Google Drive (readme.md mục 6) — hoàn toàn
TÙY CHỌN, không bắt buộc phải dùng. Mọi hàm ở đây tự kiểm tra cfg.hub có được
cấu hình đủ không trước khi làm gì, không raise nếu thiếu (chỉ in cảnh báo)
— vì đây là lớp AN TOÀN BỔ SUNG, không nên làm training chính bị crash chỉ vì
thiếu Hub token.
"""

import os
import shutil
from typing import Optional

from huggingface_hub import login, snapshot_download
from transformers.trainer_utils import get_last_checkpoint
from accelerate import PartialState

from app.llama.config import Config


def hub_login(cfg: Config) -> bool:
    """Đăng nhập HF Hub nếu cfg.hub.hf_token có sẵn. Trả về True nếu đã login
    thành công, False nếu thiếu token (caller tự quyết định có cần Hub hay
    không — không raise, vì Hub chỉ là backup phụ)."""
    if cfg.hub.hf_token:
        login(token=cfg.hub.hf_token)
        return True
    print("  ⚠ cfg.hub.hf_token trống — nhớ gọi huggingface_hub.login() hoặc "
          "notebook_login() thủ công trước khi train nếu muốn push lên Hub.")
    return False


def push_tokenizer_once(tokenizer, cfg: Config) -> None:
    """Push tokenizer lên Hub — CHỈ CẦN GỌI 1 LẦN trước khi vào vòng lặp train
    (tokenizer cố định xuyên suốt toàn bộ quá trình, không đổi theo shard/step).
    Nằm trên branch "main" của repo, không bị các lần push checkpoint (branch
    "last-checkpoint") ghi đè — xem readme.md mục 6.2b.

    QUAN TRỌNG: guard bằng is_main_process — khi chạy multi-process (8 core
    TPU qua notebook_launcher), main() được TÁM process gọi song song; không
    guard sẽ khiến cả 8 process cùng push lên Hub 1 lúc (lãng phí băng thông,
    có thể race ghi đè lẫn nhau giữa các lần push)."""
    if not PartialState().is_main_process:
        return
    if not cfg.hub.repo_id:
        print("  ⚠ cfg.hub.repo_id trống — bỏ qua push tokenizer lên Hub.")
        return
    tokenizer.push_to_hub(cfg.hub.repo_id, private=cfg.hub.private_repo)


def resume_checkpoint_from_hub_if_needed(cfg: Config) -> Optional[str]:
    """
    Gọi hàm này TRƯỚC khi chạy main() nếu nghi ngờ Google Drive đã mất
    checkpoint cục bộ (Drive lỗi/hết quota giữa hành trình train dài).

    Ưu tiên checkpoint local trước (nhanh hơn, không tốn băng thông) — chỉ
    pull từ Hub khi local KHÔNG có gì.
    """
    local_ckpt = get_last_checkpoint(cfg.train.output_dir)
    if local_ckpt:
        print(f"Đã có checkpoint cục bộ: {local_ckpt}, không cần pull từ Hub.")
        return local_ckpt

    if not cfg.hub.repo_id:
        print("Không tìm thấy checkpoint cục bộ, và cfg.hub.repo_id trống — không thể pull từ Hub.")
        return None

    print("Không tìm thấy checkpoint cục bộ, thử pull từ Hub...")
    try:
        hub_ckpt_dir = snapshot_download(repo_id=cfg.hub.repo_id, revision="last-checkpoint")
        # Copy về đúng vị trí output_dir để get_last_checkpoint() ở các bước
        # sau (train.py::main) nhận diện được như checkpoint local bình thường.
        dest = os.path.join(cfg.train.output_dir, "checkpoint-from-hub")
        shutil.copytree(hub_ckpt_dir, dest, dirs_exist_ok=True)
        print(f"Đã khôi phục checkpoint từ Hub về {dest}")
        return dest
    except Exception as e:
        print(f"Không có checkpoint trên Hub hoặc lỗi khi tải: {e}")
        return None