# Convention: Chạy mọi thao tác từ thư mục gốc (repo root)

Kể từ bản cập nhật này, **mọi** thao tác (train, generate, benchmark, sinh
dataset...) đều giả định được chạy/import từ **thư mục gốc project**
(thư mục chứa `app/`, ví dụ `trading_model/`) — không còn `cd` vào
`app/memlm/` hay bất kỳ thư mục con nào trước khi chạy.

## 3 cách hợp lệ để chạy 1 entry-point (vd `train.py`)

```python
# 1. Import như package trong notebook (khuyến nghị cho Colab)
from app.memlm.config import get_110m_config
from app.memlm.train import main
main(get_110m_config())
```

```bash
# 2. Chạy trực tiếp bằng đường dẫn file, từ thư mục gốc
python app/memlm/train.py
```

```bash
# 3. Chạy như module (cách "chuẩn" nhất về mặt Python packaging)
python -m app.memlm.train
```

Cả 3 cách đều cho kết quả giống nhau. Cách (3) là cách Python-idiomatic
nhất và **không bao giờ cần tới bootstrap sys.path** (xem bên dưới) vì
`python -m` tự thêm thư mục hiện tại vào `sys.path`. Cách (1)/(2) cũng
hoạt động vì các file entry-point đã được chỉnh sửa để tự lo việc này.

## Vì sao trước đây bị lỗi khi đổi sang chạy từ gốc

Trước đây, các file trong `app/memlm/` dùng import kiểu "phẳng"
(giả định `cwd`/`sys.path` trỏ vào chính `app/memlm/`):

```python
# train.py (CŨ — chỉ đúng khi cwd = app/memlm/)
from config import get_100m_config
from model import build_model
from trainer import run_pretrain
```

Cách này **chỉ** hoạt động nếu bạn `cd app/memlm` rồi `python train.py`
— vì khi đó Python tự thêm thư mục chứa script (`app/memlm/`) vào
`sys.path`. Khi đổi sang `from app.memlm.train import main` (import như 1
package con của `app`), Python **không** tự thêm `app/memlm/` vào
`sys.path` nữa — chỉ dùng cơ chế package thông thường — nên `from config
import ...` bị `ModuleNotFoundError`.

## Quy tắc import mới (đã áp dụng)

1. **Import xuyên module trong `app/memlm/` → luôn dùng đường dẫn tuyệt
   đối `app.memlm.xxx`**, không dùng tên phẳng:

   ```python
   # ĐÚNG
   from app.memlm.model import causal_mask
   from app.memlm.utils import save_checkpoint, load_checkpoint
   from app.memlm.benchmark import run_all

   # SAI (chỉ đúng khi cwd = app/memlm/)
   from model import causal_mask
   from utils import save_checkpoint, load_checkpoint
   from benchmark import run_all
   ```

2. **Import giữa các file CÙNG 1 package con** (vd `block.py` →
   `attention.py` cùng trong `app/memlm/model/`) vẫn dùng **relative
   import** (`from .attention import SelfAttentionRoPE`) — không đổi,
   loại này luôn hoạt động đúng bất kể cách file cha được import.

3. **Các entry-point có thể chạy trực tiếp** (`train.py`, `generate.py`,
   `benchmark.py`, `benchmark_hellaswag.py`, và vài script trong
   `app/utils/`) có thêm 1 đoạn **bootstrap** ở đầu file:

   ```python
   import os, sys
   _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
   _REPO_ROOT = _THIS_DIR
   while not os.path.isdir(os.path.join(_REPO_ROOT, "app")):
       _parent = os.path.dirname(_REPO_ROOT)
       if _parent == _REPO_ROOT:
           raise RuntimeError("Không tìm thấy thư mục gốc project (chứa 'app/').")
       _REPO_ROOT = _parent
   if _REPO_ROOT not in sys.path:
       sys.path.insert(0, _REPO_ROOT)
   ```

   Đoạn này tự đi ngược lên từ vị trí file hiện tại cho tới khi tìm thấy
   thư mục chứa `app/` (chính là thư mục gốc project), rồi thêm nó vào
   `sys.path`. Nhờ vậy `python app/memlm/train.py` (cách 2) vẫn chạy đúng
   dù không dùng `-m`. Đoạn bootstrap này **vô hại** khi file được import
   theo cách (1)/(3) — lúc đó thư mục gốc đã có sẵn trong `sys.path` nên
   `if _REPO_ROOT not in sys.path` sẽ là no-op.

4. **File không phải entry-point** (không có khối `if __name__ ==
   "__main__":` thao tác chính, chỉ được import — vd `trainer/base.py`,
   `trainer/pretrain.py`, `model/*.py`, `utils/*.py`) — **không cần**
   bootstrap, vì chúng luôn được import gián tiếp qua 1 entry-point đã lo
   việc thêm root vào `sys.path` từ trước.

## Đường dẫn file phụ thuộc `cwd` — đã rà soát và sửa

Một số script trước đây tính đường dẫn phụ (vd tới `custom_tokenizer/`)
dựa trên `cwd` hoặc hard-code sai cấp thư mục. Đã sửa toàn bộ để tính theo
`os.path.dirname(os.path.abspath(__file__))` (vị trí file, không phải
`cwd`) hoặc theo `_REPO_ROOT` (từ bootstrap) — **không phụ thuộc bạn đang
đứng ở thư mục nào khi gọi script**:

| File | Trước | Sau |
|---|---|---|
| `app/memlm/tokenizer.py` (`__main__`) | default path `"custom_tokenizer"` (tương đối theo cwd) | tính theo vị trí file → luôn ra `app/memlm/custom_tokenizer` |
| `app/memlm/scripts/train_tokenizer.py` | default `--output-dir` = `"custom_tokenizer"` (tương đối cwd) | tính theo vị trí file → `app/memlm/custom_tokenizer` |
| `app/memlm/scripts/add_custom_tokens.py` | tương tự | tương tự |
| `app/utils/build_dataset_to_parquet.py` | **bug thật**: tính `tok_path = dirname(__file__)/app/memlm/custom_tokenizer`, nhưng `dirname(__file__)` đã là `.../app/utils` → ra đường dẫn sai `.../app/utils/app/memlm/custom_tokenizer` | dùng `_REPO_ROOT` từ bootstrap → luôn đúng `app/memlm/custom_tokenizer` bất kể vị trí file |
| `app/utils/chart/chart_action_gen.py` | hard-code path Windows tương đối (`data\XAUUSD_1Min.csv`) | tính từ `_REPO_ROOT` → `data/XAUUSD_1Min.csv`, hoạt động cả Windows/Linux/Colab |

## Checklist khi thêm file mới vào `app/memlm/` hoặc `app/utils/`

- [ ] Import module khác trong cùng cây `app/...` → dùng đường dẫn tuyệt
      đối `app.xxx.yyy`, không dùng tên phẳng.
- [ ] Import module cùng package con (anh em trực tiếp) → dùng relative
      import (`.module`).
- [ ] Nếu file có `if __name__ == "__main__":` và dự định chạy trực tiếp
      bằng `python path/to/file.py` (không phải luôn qua `-m`) → thêm
      đoạn bootstrap ở đầu file.
- [ ] Mọi đường dẫn tới file/thư mục phụ (checkpoint, tokenizer, data...)
      → tính từ `os.path.dirname(os.path.abspath(__file__))` hoặc
      `_REPO_ROOT`, **không** dùng chuỗi tương đối trần (`"data/x.csv"`)
      trừ khi cố ý muốn nó phụ thuộc `cwd` (hiếm khi đúng).