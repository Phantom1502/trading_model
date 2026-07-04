"""
stats_validate.py — Chạy validate trên dataset ICT thật đã gen
========================================================================
Đọc parquet output của pipeline render (cột text/meta theo đúng schema
build_dataset_to_parquet.py: text, source, token_length, meta), tách
ngược "text" (đã ghép 4 phần: chart\nrequest\nexplanation\neval) thành
dict sample, rồi chạy:

    1. validate_no_leakage — TỪNG DÒNG độc lập, không cần group.
    2. validate_cross_consistency — GROUP theo (chart_index, sub_range)
       trong meta (các dòng cùng group = cùng render từ 1 fact JSON gốc,
       tối đa 4 dòng: Swept/FVG/Shift/Tổng hợp).

Kèm 2 thống kê phụ hữu ích trước khi đưa data vào train:
    - Phân phối token_length (so với ngân sách ~400 token phần text đã
      tính trong spec mục 5).
    - Tỷ lệ 4 dạng mẫu tin (Swept/FVG/Shift/Tổng hợp) — suy loại mẫu từ
      field đặc trưng trong eval (DEPTH=/GAP_SIZE=/BROKEN=/SEQUENCE=),
      KHÔNG cần cột riêng lưu sample_type trong parquet.

Cách dùng:
    python stats_validate.py --parquet data/ict_dataset.parquet
    python stats_validate.py --parquet ... --sample 5000   # chạy nhanh thử trước
"""

import argparse
import json
from collections import defaultdict
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from app.ict.validate import validate_cross_consistency, validate_no_leakage


# ══════════════════════════════════════════════════════════════════════
# Tách ngược "text" đã ghép 4 phần -> dict sample
# ══════════════════════════════════════════════════════════════════════

def _parse_text_to_sample(text: str) -> Optional[Dict[str, str]]:
    """
    render.py ghép: f"{chart}\\n{request}\\n{explanation}\\n{eval_block}"
    Tách ngược theo đúng thứ tự — dòng đầu là chart, dòng cuối là eval
    (bắt đầu "<eval>", kết thúc "</eval>"), phần giữa gộp lại làm explanation
    (phòng trường hợp explanation có xuống dòng, dù hiện tại template
    không tạo newline trong explanation).
    """
    if not isinstance(text, str):
        return None
    parts = text.split("\n")
    if len(parts) < 4:
        return None

    chart, request = parts[0], parts[1]
    eval_block = parts[-1].strip()
    explanation = "\n".join(parts[2:-1])

    if not (eval_block.startswith("<eval>") and eval_block.endswith("</eval>")):
        return None
    if not chart.startswith("<chart>"):
        return None

    return {"chart": chart, "request": request, "explanation": explanation, "eval": eval_block}


def _infer_sample_type(eval_block: str) -> str:
    """Suy loại mẫu từ field đặc trưng trong eval — không cần cột riêng."""
    if "SEQUENCE=" in eval_block:
        return "synthesis"
    if "GAP_SIZE=" in eval_block:
        return "fvg"
    if "BROKEN=" in eval_block:
        return "shift"
    if "DEPTH=" in eval_block:
        return "swept"
    return "unknown"


def _parse_meta(raw_meta: Any) -> Dict[str, Any]:
    if isinstance(raw_meta, dict):
        return raw_meta
    if isinstance(raw_meta, str):
        try:
            return json.loads(raw_meta)
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}


def _group_key(meta: Dict[str, Any]) -> tuple:
    """Định danh 1 chart gốc — các mẫu cùng key này render từ CÙNG 1 fact JSON."""
    sub_range = meta.get("sub_range")
    return (meta.get("chart_index"), tuple(sub_range) if sub_range else None)


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Validate dataset ICT thật đã gen")
    parser.add_argument("--parquet", type=str, required=True)
    parser.add_argument("--text-col", type=str, default="text")
    parser.add_argument("--meta-col", type=str, default="meta")
    parser.add_argument("--token-length-col", type=str, default="token_length")
    parser.add_argument("--sample", type=int, default=None, help="Chỉ lấy N dòng đầu để chạy nhanh thử")
    parser.add_argument("--batch-size", type=int, default=20_000,
                         help="Số dòng đọc mỗi batch — RAM tỉ lệ với số này, KHÔNG tỉ lệ với tổng file")
    args = parser.parse_args()

    pf = pq.ParquetFile(args.parquet)
    total_rows = pf.metadata.num_rows
    schema_names = pf.schema_arrow.names

    needed_cols = [args.text_col, args.meta_col]
    has_token_len = args.token_length_col in schema_names
    if has_token_len:
        needed_cols.append(args.token_length_col)

    n_target = min(args.sample, total_rows) if args.sample else total_rows
    print(f"Đang xử lý {n_target:,} / {total_rows:,} dòng (streaming, batch={args.batch_size:,})...")

    # ── Accumulators — KHÔNG giữ df gốc, KHÔNG giữ list toàn bộ row đã parse ──
    n_seen        = 0
    parse_fail    = 0
    leak_pass     = leak_fail = 0
    leak_fail_by_type   = defaultdict(int)
    leak_fail_examples  = []            # tối đa 5 phần tử — không đáng kể
    type_counts   = defaultdict(int)
    type_lengths  = defaultdict(list)   # chỉ float, nhẹ hơn nhiều so với giữ cả sample
    all_lengths   = []

    # groups: key -> list sample. Đây là phần DUY NHẤT buộc phải giữ tới cuối
    # (cần so sánh các mẫu CÙNG 1 chart gốc, có thể rơi vào batch khác nhau).
    groups = defaultdict(list)

    stop = False
    for batch in pf.iter_batches(batch_size=args.batch_size, columns=needed_cols):
        bd = batch.to_pydict()   # list Python thuần — không tạo Series/Index per-row như iterrows()
        n_batch = len(bd[args.text_col])
        tl_col  = bd[args.token_length_col] if has_token_len else [None] * n_batch

        for i in range(n_batch):
            if args.sample and n_seen >= args.sample:
                stop = True
                break
            n_seen += 1

            sample = _parse_text_to_sample(bd[args.text_col][i])
            if sample is None:
                parse_fail += 1
                continue

            meta = _parse_meta(bd[args.meta_col][i])
            key  = _group_key(meta)
            typ  = _infer_sample_type(sample["eval"])
            tl   = tl_col[i]

            type_counts[typ] += 1
            if tl is not None and not pd.isna(tl):
                tl_f = float(tl)
                type_lengths[typ].append(tl_f)
                all_lengths.append(tl_f)

            if validate_no_leakage(sample):
                leak_pass += 1
            else:
                leak_fail += 1
                leak_fail_by_type[typ] += 1
                if len(leak_fail_examples) < 5:
                    leak_fail_examples.append({"type": typ, "sample": sample})

            groups[key].append(sample)

        del bd, batch   # giải phóng ngay, không đợi tới cuối hàm
        if stop:
            break

    total_parsed = n_seen - parse_fail

    print(f"\n{'='*60}\n  PARSE\n{'='*60}")
    print(f"  Tổng dòng     : {n_seen:,}")
    print(f"  Parse OK      : {total_parsed:,} ({total_parsed/n_seen*100:.2f}%)")
    print(f"  Parse FAIL    : {parse_fail:,} ({parse_fail/n_seen*100:.2f}%)")
    if total_parsed == 0:
        print("\n⚠ Không có dòng nào parse được — kiểm tra lại --text-col hoặc format text.")
        return

    print(f"\n{'='*60}\n  VALIDATE_NO_LEAKAGE\n{'='*60}")
    print(f"  Pass: {leak_pass:,} ({leak_pass/total_parsed*100:.2f}%)")
    print(f"  Fail: {leak_fail:,} ({leak_fail/total_parsed*100:.2f}%)")
    if leak_fail_by_type:
        print(f"\n  Fail breakdown theo loại mẫu:")
        for t, c in sorted(leak_fail_by_type.items(), key=lambda x: -x[1]):
            print(f"    {t:<12} {c:,}")
        print(f"\n  {min(5, len(leak_fail_examples))} ví dụ FAIL đầu tiên (để debug):")
        for r in leak_fail_examples:
            print(f"    [{r['type']}] explanation: {r['sample']['explanation'][:120]}...")
            print(f"             eval: {r['sample']['eval'][:150]}...")

    # ── validate_cross_consistency — group theo chart gốc ──────────────
    cross_pass = cross_fail = 0
    cross_fail_keys = []
    multi_sample_groups = 0

    for key, samples in groups.items():
        if len(samples) < 2:
            continue
        multi_sample_groups += 1
        if validate_cross_consistency(samples):
            cross_pass += 1
        else:
            cross_fail += 1
            if len(cross_fail_keys) < 5:
                cross_fail_keys.append(key)

    print(f"\n{'='*60}\n  VALIDATE_CROSS_CONSISTENCY\n{'='*60}")
    print(f"  Tổng chart gốc (group)          : {len(groups):,}")
    print(f"  Chart có >=2 mẫu (đủ so sánh)    : {multi_sample_groups:,}")
    if multi_sample_groups:
        print(f"  Pass: {cross_pass:,} ({cross_pass/multi_sample_groups*100:.2f}%)")
        print(f"  Fail: {cross_fail:,} ({cross_fail/multi_sample_groups*100:.2f}%)")
        if cross_fail_keys:
            print(f"\n  Ví dụ group FAIL (chart_index, sub_range): {cross_fail_keys}")

    del groups  # xong phần cần groups, giải phóng trước khi in phần còn lại

    if all_lengths:
        lengths = np.array(all_lengths)
        print(f"\n{'='*60}\n  PHÂN PHỐI TOKEN_LENGTH (TỔNG THỂ)\n{'='*60}")
        for p in [50, 75, 90, 95, 99]:
            print(f"    p{p}: {np.percentile(lengths, p):.0f}")
        print(f"    max: {lengths.max():.0f}")
        over_budget = (lengths > 512).sum()
        print(f"    Số mẫu > 512 token (vượt max_seq): {over_budget:,} ({over_budget/len(lengths)*100:.2f}%)")

    if type_lengths:
        print(f"\n{'='*60}\n  PHÂN PHỐI TOKEN_LENGTH THEO LOẠI MẪU\n{'='*60}")
        for t, vals in sorted(type_lengths.items(), key=lambda x: -np.median(x[1])):
            arr = np.array(vals)
            over = (arr > 512).sum()
            print(f"\n  [{t}] N={len(arr):,}")
            print(f"    p50: {np.percentile(arr,50):.0f}  p75: {np.percentile(arr,75):.0f}  "
                  f"p90: {np.percentile(arr,90):.0f}  p99: {np.percentile(arr,99):.0f}  max: {arr.max():.0f}")
            print(f"    Vượt 512 token: {over:,} ({over/len(arr)*100:.2f}%)")

    print(f"\n{'='*60}\n  TỶ LỆ 4 DẠNG MẪU TIN\n{'='*60}")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"    {t:<12} {c:,} ({c/total_parsed*100:.1f}%)")


if __name__ == "__main__":
    main()