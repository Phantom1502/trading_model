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
    args = parser.parse_args()

    df = pd.read_parquet(args.parquet)
    if args.sample:
        df = df.head(args.sample)

    print(f"Đang xử lý {len(df):,} dòng...")

    # ── Parse toàn bộ, tách sample + suy loại + group key ──────────────
    parsed_rows = []
    parse_fail = 0
    for _, row in df.iterrows():
        sample = _parse_text_to_sample(row[args.text_col])
        if sample is None:
            parse_fail += 1
            continue
        meta = _parse_meta(row.get(args.meta_col))
        parsed_rows.append({
            "key"         : _group_key(meta),
            "type"        : _infer_sample_type(sample["eval"]),
            "sample"      : sample,
            "token_length": row.get(args.token_length_col),
        })

    print(f"\n{'='*60}")
    print(f"  PARSE")
    print(f"{'='*60}")
    print(f"  Tổng dòng     : {len(df):,}")
    print(f"  Parse OK      : {len(parsed_rows):,} ({len(parsed_rows)/len(df)*100:.2f}%)")
    print(f"  Parse FAIL    : {parse_fail:,} ({parse_fail/len(df)*100:.2f}%)")
    if not parsed_rows:
        print("\n⚠ Không có dòng nào parse được — kiểm tra lại --text-col hoặc format text.")
        return

    # ── validate_no_leakage — từng dòng độc lập ────────────────────────
    leak_pass = leak_fail = 0
    leak_fail_by_type = defaultdict(int)
    leak_fail_examples = []

    for r in parsed_rows:
        if validate_no_leakage(r["sample"]):
            leak_pass += 1
        else:
            leak_fail += 1
            leak_fail_by_type[r["type"]] += 1
            if len(leak_fail_examples) < 5:
                leak_fail_examples.append(r)

    total_parsed = len(parsed_rows)
    print(f"\n{'='*60}")
    print(f"  VALIDATE_NO_LEAKAGE")
    print(f"{'='*60}")
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
    groups = defaultdict(list)
    for r in parsed_rows:
        groups[r["key"]].append(r["sample"])

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

    print(f"\n{'='*60}")
    print(f"  VALIDATE_CROSS_CONSISTENCY")
    print(f"{'='*60}")
    print(f"  Tổng chart gốc (group)          : {len(groups):,}")
    print(f"  Chart có >=2 mẫu (đủ so sánh)    : {multi_sample_groups:,}")
    if multi_sample_groups:
        print(f"  Pass: {cross_pass:,} ({cross_pass/multi_sample_groups*100:.2f}%)")
        print(f"  Fail: {cross_fail:,} ({cross_fail/multi_sample_groups*100:.2f}%)")
        if cross_fail_keys:
            print(f"\n  Ví dụ group FAIL (chart_index, sub_range): {cross_fail_keys}")

    # ── Thống kê phụ: phân phối token_length ───────────────────────────
    if args.token_length_col in df.columns:
        lengths = pd.to_numeric(df[args.token_length_col], errors="coerce").dropna().values
        if len(lengths) > 0:
            print(f"\n{'='*60}")
            print(f"  PHÂN PHỐI TOKEN_LENGTH (TỔNG THỂ)")
            print(f"{'='*60}")
            for p in [50, 75, 90, 95, 99]:
                print(f"    p{p}: {np.percentile(lengths, p):.0f}")
            print(f"    max: {lengths.max():.0f}")
            over_budget = (lengths > 512).sum()
            print(f"    Số mẫu > 512 token (vượt max_seq): {over_budget:,} ({over_budget/len(lengths)*100:.2f}%)")

    # ── Breakdown token_length THEO TỪNG LOẠI MẪU — xác định thủ phạm ──
    type_lengths = defaultdict(list)
    for r in parsed_rows:
        tl = r["token_length"]
        if tl is not None and not pd.isna(tl):
            type_lengths[r["type"]].append(float(tl))

    if type_lengths:
        print(f"\n{'='*60}")
        print(f"  PHÂN PHỐI TOKEN_LENGTH THEO LOẠI MẪU")
        print(f"{'='*60}")
        for t, vals in sorted(type_lengths.items(), key=lambda x: -np.median(x[1])):
            arr = np.array(vals)
            over = (arr > 512).sum()
            print(f"\n  [{t}] N={len(arr):,}")
            print(f"    p50: {np.percentile(arr,50):.0f}  p75: {np.percentile(arr,75):.0f}  "
                  f"p90: {np.percentile(arr,90):.0f}  p99: {np.percentile(arr,99):.0f}  max: {arr.max():.0f}")
            print(f"    Vượt 512 token: {over:,} ({over/len(arr)*100:.2f}%)")

    # ── Thống kê phụ: tỷ lệ 4 dạng mẫu tin ──────────────────────────────
    type_counts = defaultdict(int)
    for r in parsed_rows:
        type_counts[r["type"]] += 1

    print(f"\n{'='*60}")
    print(f"  TỶ LỆ 4 DẠNG MẪU TIN")
    print(f"{'='*60}")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"    {t:<12} {c:,} ({c/total_parsed*100:.1f}%)")


if __name__ == "__main__":
    main()