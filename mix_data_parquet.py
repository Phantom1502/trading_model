import os, json, glob
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

_PARQUET_SCHEMA = pa.schema([
    ('text', pa.string()),
    ('source', pa.string()),
    ('token_length', pa.int64()),
    ('meta', pa.string()),
])

# ====== CẤU HÌNH ======
CHUNK_TOKEN_LIMIT = 1_000_000_000   # 1B token / file gộp
OUTPUT_DIR        = r"E:\LLM Dataset\Mix\merged_output"
CHECKPOINT_PATH   = os.path.join(OUTPUT_DIR, "checkpoint.json")
INTERLEAVE_TOKENS = 1_000_000     # "nếm" tối đa bao nhiêu token từ 1 category rồi mới đổi cat khác
PART_FLUSH_TOKENS = 50_000_000    # gom nhiều đợt interleave lại, đủ ngưỡng này mới ghi 1 part-file

CATEGORY_CONFIG = {
    "vi": {"files": sorted(glob.glob("E:/LLM Dataset/Mix/vi/*.parquet")), "ratio": 0.2}, # 63GB
    "wiki_en": {"files": sorted(glob.glob("E:/LLM Dataset/Mix/wiki_en/*.parquet")), "ratio": 0.1}, # 6.7GB
    "python": {"files": sorted(glob.glob("E:/LLM Dataset/Mix/python/*.parquet")), "ratio": 0.25}, # 4GB
    "math": {"files": sorted(glob.glob("E:/LLM Dataset/Mix/math/*.parquet")), "ratio": 0.15}, # 2.8GB
    "social_en": {"files": sorted(glob.glob("E:/LLM Dataset/Mix/social_en/*.parquet")), "ratio": 0.1}, # 200GB
    "trading": {"files": sorted(glob.glob("E:/LLM Dataset/Mix/trading/*.parquet")), "ratio": 0.2}, # 1.5GB
}
'''

CATEGORY_CONFIG = {
    "basic": {"files": sorted(glob.glob(r"E:\LLM Dataset\Mix\trading\basic_generator.parquet")), "ratio": 0.01}, # 63GB
    "candle": {"files": sorted(glob.glob(r"E:\LLM Dataset\Mix\trading\candle_generator.parquet")), "ratio": 0.01}, # 6.7GB
    "gap": {"files": sorted(glob.glob(r"E:\LLM Dataset\Mix\trading\gap_generator.parquet")), "ratio": 0.01}, # 4GB
    "math": {"files": sorted(glob.glob(r"E:\LLM Dataset\Mix\trading\math_generator.parquet")), "ratio": 0.01}, # 2.8GB
    "book": {"files": sorted(glob.glob(r"E:\LLM Dataset\Mix\trading\trading_book.parquet")), "ratio": 0.01}, # 200GB
    "trading1": {"files": sorted(glob.glob(r"E:\LLM Dataset\Mix\trading\XAUUSD_M1_base.parquet")), "ratio": 0.8}, # 1.5GB
    "trading2": {"files": sorted(glob.glob(r"E:\LLM Dataset\Mix\trading\XAUUSD_M5_base.parquet")), "ratio": 0.15}, # 1.5GB
}'''
assert abs(sum(c["ratio"] for c in CATEGORY_CONFIG.values()) - 1.0) < 1e-6

# ====== CHECKPOINT STATE ======
def _new_category_state(cat):
    cfg = CATEGORY_CONFIG[cat]
    return {
        "ratio": cfg["ratio"],
        "file_idx": 0,        # file đang/sắp đọc
        "row_group_idx": 0,   # row-group kế tiếp cần đọc trong file đó
        "tokens_in_chunk": 0,
        "done": len(cfg["files"]) == 0,
        "files_status": {f: "pending" for f in cfg["files"]},
    }

def new_state():
    return {
        "chunk_index": 0,
        "chunk_tokens": 0,
        "part_counter": 0,
        "categories": {cat: _new_category_state(cat) for cat in CATEGORY_CONFIG},
    }

def load_state():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return new_state()

def save_state(state):
    tmp = CHECKPOINT_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp, CHECKPOINT_PATH)  # ghi atomic, tránh hỏng checkpoint khi crash
    
# ====== ĐỌC DỮ LIỆU NGUỒN ======
def _read_unit(cat, cstate):
    """Đọc 1 row-group kế tiếp của category. Trả (table, tokens) hoặc (None, 0) nếu hết dữ liệu."""
    files = CATEGORY_CONFIG[cat]["files"]
    while cstate["file_idx"] < len(files):
        path = files[cstate["file_idx"]]
        pf = pq.ParquetFile(path)
        if cstate["row_group_idx"] >= pf.num_row_groups:
            cstate["files_status"][path] = "done"
            cstate["file_idx"] += 1
            cstate["row_group_idx"] = 0
            continue
        cstate["files_status"][path] = "in_progress"
        table = pf.read_row_group(cstate["row_group_idx"])
        cstate["row_group_idx"] += 1
        tokens = pc.sum(table.column("token_length")).as_py() or 0
        return table, tokens
    cstate["done"] = True
    return None, 0

def _pull_mixed_batch(state, budget_tokens):
    """
    Liên tục: chọn category đang thiếu tỉ lệ nhất -> đọc tối đa INTERLEAVE_TOKENS từ nó
    -> chọn lại category -> ... cho tới khi đạt budget_tokens (PART_FLUSH_TOKENS hoặc
    phần còn thiếu của chunk, cái nào nhỏ hơn).
    Nhờ vậy 1 part-file 50M token có thể chứa ~20-25 đoạn nhỏ xen kẽ nhiều category,
    thay vì 1 cục thuần 1 category.
    """
    limit = min(PART_FLUSH_TOKENS, budget_tokens) if budget_tokens else PART_FLUSH_TOKENS
    tables, total, used = [], 0, {}

    while total < limit:
        cat = _pick_category(state)
        if cat is None:
            break
        cstate = state["categories"][cat]
        sub_budget = min(INTERLEAVE_TOKENS, limit - total)
        table, tokens = _pull_batch(cat, cstate, sub_budget)
        if table is None:
            continue  # category này vừa hết -> done=True đã set trong _read_unit, vòng sau pick cat khác

        tables.append(table)
        total += tokens
        used[cat] = used.get(cat, 0) + tokens

        # cộng dồn NGAY để _pick_category() ở vòng kế thấy deficit cập nhật real-time
        cstate["tokens_in_chunk"] += tokens
        state["chunk_tokens"] += tokens

    if not tables:
        return None, 0, {}
    return pa.concat_tables(tables), total, used

def _pull_batch(cat, cstate, max_tokens):
    """Gom vài row-group tới khi đạt PART_FLUSH_TOKENS hoặc max_tokens (phần còn thiếu của chunk)."""
    limit = min(PART_FLUSH_TOKENS, max_tokens) if max_tokens else PART_FLUSH_TOKENS
    tables, total = [], 0
    while total < limit:
        table, tokens = _read_unit(cat, cstate)
        if table is None:
            break
        tables.append(table)
        total += tokens
    if not tables:
        return None, 0
    return pa.concat_tables(tables), total

# ====== SCHEDULER theo tỉ lệ + MAIN LOOP ======
def _pick_category(state):
    candidates = [(c, s) for c, s in state["categories"].items() if not s["done"]]
    if not candidates:
        return None
    total = state["chunk_tokens"] or 1
    def deficit(s):
        return s["tokens_in_chunk"] / total - s["ratio"]  # âm nhất = đang thiếu nhiều nhất
    candidates.sort(key=lambda kv: deficit(kv[1]))
    return candidates[0][0]

def _close_chunk(state):
    print(f"[DONE] chunk_{state['chunk_index']:05d} = {state['chunk_tokens']:,} token")
    state["chunk_index"] += 1
    state["chunk_tokens"] = 0
    state["part_counter"] = 0
    for s in state["categories"].values():
        s["tokens_in_chunk"] = 0
    save_state(state)

def merge():
    state = load_state()
    while True:
        if all(s["done"] for s in state["categories"].values()):
            if state["chunk_tokens"] > 0:
                _close_chunk(state)
            print("Đã merge xong toàn bộ dữ liệu nguồn.")
            break

        remain = CHUNK_TOKEN_LIMIT - state["chunk_tokens"]
        table, tokens, used = _pull_mixed_batch(state, remain)

        if table is None:
            continue  # chưa hẳn done (cờ done set lệch 1 nhịp), vòng sau check lại "all done"

        chunk_dir = os.path.join(OUTPUT_DIR, f"chunk_{state['chunk_index']:05d}")
        os.makedirs(chunk_dir, exist_ok=True)
        part_path = os.path.join(chunk_dir, f"part_{state['part_counter']:06d}.parquet")
        pq.write_table(table, part_path)   # ghi xong rồi mới commit checkpoint

        state["part_counter"] += 1
        save_state(state)   # tokens_in_chunk / chunk_tokens đã được cộng dồn trong _pull_mixed_batch

        mix_str = ", ".join(f"{c}:{t:,}" for c, t in used.items())
        print(f"chunk={state['chunk_index']} part={state['part_counter']-1} "
              f"+{tokens:,} ({state['chunk_tokens']:,}/{CHUNK_TOKEN_LIMIT:,}) mix=[{mix_str}]")

        if state["chunk_tokens"] >= CHUNK_TOKEN_LIMIT:
            _close_chunk(state)

def print_status():
    state = load_state()
    print(f"Đang ở chunk {state['chunk_index']} ({state['chunk_tokens']:,} token)")
    for cat, cs in state["categories"].items():
        print(f"\n[{cat}] ratio={cs['ratio']} done={cs['done']}")
        for f, st in cs["files_status"].items():
            print(f"  {st:12s} {f}")
            
if __name__ == "__main__":
    merge()