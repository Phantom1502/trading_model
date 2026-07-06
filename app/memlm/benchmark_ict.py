"""
benchmark_ict.py — Đánh giá MemoryLM trên hiểu biết ICT (Swept/FVG/Shift)
================================================================================
Dùng CÙNG phương pháp với benchmark.py (BenchItem + avg_logprob_per_token):
đo xem model có gán log-prob TRUNG BÌNH/TOKEN cao hơn cho completion ĐÚNG
(khớp fact thật từ app.ict detector) so với completion SAI (perturbed) hay
không — KHÔNG yêu cầu model tự generate đúng format từ đầu, phù hợp giai
đoạn pretrain (chưa SFT).

THIẾT KẾ QUAN TRỌNG — prompt dừng ngay tại "<eval>":
    prompt = chart + request + explanation + "<eval>"
    completion (positive/negative) = phần NỘI DUNG bên trong <eval>...</eval>

Cố tình dừng prompt ngay sau khi model đã thấy phần Lý giải (giống hệt lúc
train) rồi mới đo completion phần Eval — đây LÀ pattern thật model đã học
trong pretrain (Lý giải luôn đứng trước Eval trong 1 mẫu). Test này KHÔNG
nhằm cô lập "chart-only" (việc đó cần thiết kế khác, xem ghi chú cuối file)
mà nhằm đo: sau khi thấy đủ ngữ cảnh y hệt lúc train, model có phân biệt
được Eval ĐÚNG khớp chart/Lý giải hay không, so với 1 Eval SAI (lệch nến,
lệch loại pattern, lệch mức giá).

GROUND TRUTH luôn lấy từ chính app.ict detector (build_facts) qua
render_*_sample() thật — KHÔNG viết tay bất kỳ giá trị đúng/sai nào, giữ
đúng nguyên tắc golden test đã dùng xuyên suốt package app/ict/.
"""

import random
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from app.ict.candle import Candle
from app.ict.parser import CandleParser
from app.ict.facts import build_facts
from app.ict.render import (
    render_swept_sample, render_fvg_sample, render_shift_sample,
)

try:
    from .benchmark import BenchItem, avg_logprob_per_token
except ImportError:
    from benchmark import BenchItem, avg_logprob_per_token


# ══════════════════════════════════════════════════════════════════════
# Perturbation — sửa 1 field trong chuỗi eval (string-level, an toàn với
# cả case có tiền tố E1_/E2_ lẫn không tiền tố N==1)
# ══════════════════════════════════════════════════════════════════════

def _replace_first_field(eval_inner: str, key: str, new_value: Any) -> Optional[str]:
    """
    Thay giá trị field `key` ĐẦU TIÊN xuất hiện trong `eval_inner` (không
    phân biệt có tiền tố E1_/E2_ hay không). Field hợp lệ khi đứng ngay
    sau: đầu chuỗi, dấu '_' (tiền tố EVENTk_), hoặc khoảng trắng (field
    kế tiếp không tiền tố khi N==1) — dùng lookbehind để tránh match nhầm
    "C" bên trong "SC" (SWING_CANDLE cũng chứa chữ C).

    Trả về None nếu không tìm thấy field này (caller nên bỏ qua case đó
    thay vì tạo negative sai/rỗng).
    """
    pattern = re.compile(rf"(?:^|(?<=[_\s])){re.escape(key)}=([^\s]+)")
    match = pattern.search(eval_inner)
    if match is None:
        return None
    return eval_inner[:match.start()] + f"{key}={new_value}" + eval_inner[match.end():]


def _extract_field(eval_inner: str, key: str) -> Optional[str]:
    """Lấy giá trị field `key` đầu tiên (chuỗi thô, chưa ép kiểu). Cùng
    quy tắc boundary với _replace_first_field() — xem docstring ở đó."""
    pattern = re.compile(rf"(?:^|(?<=[_\s])){re.escape(key)}=([^\s]+)")
    match = pattern.search(eval_inner)
    return match.group(1) if match else None


# ══════════════════════════════════════════════════════════════════════
# Xây BenchItem từ 1 sample thật (render_*_sample output)
# ══════════════════════════════════════════════════════════════════════

def _make_prompt(sample: Dict[str, Any]) -> str:
    """Prompt dừng NGAY sau khi mở tag <eval> — completion là phần bên trong."""
    return f"{sample['chart']}\n{sample['request']}\n{sample['explanation']}\n<eval>"


def _inner_eval(sample: Dict[str, Any]) -> str:
    """Trích phần bên trong <eval>...</eval>, bỏ 2 tag."""
    e = sample["eval"]
    return e[len("<eval>"):-len("</eval>")]


def _build_swept_items(sample: Dict[str, Any], rng: random.Random) -> List[BenchItem]:
    inner = _inner_eval(sample)
    prompt = _make_prompt(sample)
    negatives = []

    # Perturb 1: sai loại (SWEEP_HIGH <-> SWEEP_LOW)
    t = _extract_field(inner, "T")
    if t in ("SWEEP_HIGH", "SWEEP_LOW"):
        wrong_t = "SWEEP_LOW" if t == "SWEEP_HIGH" else "SWEEP_HIGH"
        neg = _replace_first_field(inner, "T", wrong_t)
        if neg:
            negatives.append(neg)

    # Perturb 2: sai nến (lệch 2-5 nến so với thật)
    c = _extract_field(inner, "C")
    if c is not None:
        wrong_c = int(c) + rng.choice([-5, -3, 3, 5])
        neg = _replace_first_field(inner, "C", wrong_c)
        if neg:
            negatives.append(neg)

    # Perturb 3: sai mức swing level (lệch đáng kể, không phải do quantize)
    sl = _extract_field(inner, "SL")
    if sl is not None:
        wrong_sl = int(sl) + rng.choice([-30, -20, 20, 30])
        neg = _replace_first_field(inner, "SL", wrong_sl)
        if neg:
            negatives.append(neg)

    if not negatives:
        return []
    return [BenchItem(prompt=prompt, positive=[inner], negative=negatives, note="ict_swept")]


def _build_fvg_items(sample: Dict[str, Any], rng: random.Random) -> List[BenchItem]:
    inner = _inner_eval(sample)
    prompt = _make_prompt(sample)
    negatives = []

    t = _extract_field(inner, "T")
    if t in ("BULL", "BEAR"):
        wrong_t = "BEAR" if t == "BULL" else "BULL"
        neg = _replace_first_field(inner, "T", wrong_t)
        if neg:
            negatives.append(neg)

    c = _extract_field(inner, "C")
    if c is not None:
        wrong_c = int(c) + rng.choice([-5, -3, 3, 5])
        neg = _replace_first_field(inner, "C", wrong_c)
        if neg:
            negatives.append(neg)

    gs = _extract_field(inner, "GS")
    if gs is not None:
        wrong_gs = max(1, int(gs) + rng.choice([-15, -10, 10, 15]))
        neg = _replace_first_field(inner, "GS", wrong_gs)
        if neg:
            negatives.append(neg)

    if not negatives:
        return []
    return [BenchItem(prompt=prompt, positive=[inner], negative=negatives, note="ict_fvg")]


def _build_shift_items(sample: Dict[str, Any], rng: random.Random) -> List[BenchItem]:
    inner = _inner_eval(sample)
    prompt = _make_prompt(sample)
    negatives = []

    # Perturb: đảo cả loại lẫn hướng cùng lúc (BOS+BULL <-> CHoCH+BEAR là
    # cặp hợp lệ đối xứng — đảo cả 2 để negative vẫn "trông hợp lệ về mặt
    # cấu trúc", không phải lỗi rõ ràng dễ đoán mà không cần hiểu chart)
    t   = _extract_field(inner, "T")
    dr  = _extract_field(inner, "DIR")
    if t in ("BOS", "CHoCH") and dr in ("BULL", "BEAR"):
        wrong_t  = "CHoCH" if t == "BOS" else "BOS"
        wrong_dr = "BEAR" if dr == "BULL" else "BULL"
        neg = _replace_first_field(inner, "T", wrong_t)
        if neg:
            neg = _replace_first_field(neg, "DIR", wrong_dr)
        if neg:
            negatives.append(neg)

    c = _extract_field(inner, "C")
    if c is not None:
        wrong_c = int(c) + rng.choice([-5, -3, 3, 5])
        neg = _replace_first_field(inner, "C", wrong_c)
        if neg:
            negatives.append(neg)

    sl = _extract_field(inner, "SL")
    if sl is not None:
        wrong_sl = int(sl) + rng.choice([-30, -20, 20, 30])
        neg = _replace_first_field(inner, "SL", wrong_sl)
        if neg:
            negatives.append(neg)

    if not negatives:
        return []
    return [BenchItem(prompt=prompt, positive=[inner], negative=negatives, note="ict_shift")]


# ══════════════════════════════════════════════════════════════════════
# Sinh toàn bộ BenchItem từ 1 tập chart thật (list[list[Candle]])
# ══════════════════════════════════════════════════════════════════════

def build_ict_bench_items(
    charts: List[List[Candle]],
    seed: int = 42,
) -> Dict[str, List[BenchItem]]:
    """
    charts: list các cửa sổ nến thật (mỗi phần tử là list[Candle], khuyến
    nghị đúng 20 nến khớp cấu hình dataset đã dùng để pretrain).

    Trả về dict {"swept": [...], "fvg": [...], "shift": [...]} — mỗi
    value là list BenchItem, dùng trực tiếp với run_logprob_benchmark()
    có sẵn trong benchmark.py.

    CHỈ những chart THẬT SỰ có event mới sinh ra BenchItem (giống nguyên
    tắc SKIP của render.py khi 0 event) — số lượng BenchItem mỗi loại phụ
    thuộc bao nhiêu chart trong `charts` có đúng loại event đó.
    """
    rng = random.Random(seed)
    items: Dict[str, List[BenchItem]] = {"swept": [], "fvg": [], "shift": []}

    for candles in charts:
        parser = CandleParser.from_candles(candles, swing_window=2)
        raw = parser.raw_text
        initial_trend = rng.choice(["BULL", "BEAR"])
        facts = build_facts(parser, initial_trend=initial_trend, lookback=10)

        swept_sample = render_swept_sample(facts, raw, rng=rng)
        if swept_sample:
            items["swept"].extend(_build_swept_items(swept_sample, rng))

        fvg_sample = render_fvg_sample(facts, raw, rng=rng)
        if fvg_sample:
            items["fvg"].extend(_build_fvg_items(fvg_sample, rng))

        shift_sample = render_shift_sample(facts, raw, rng=rng)
        if shift_sample:
            items["shift"].extend(_build_shift_items(shift_sample, rng))

    return items


# ══════════════════════════════════════════════════════════════════════
# Runner — tích hợp cùng pattern run_logprob_benchmark() của benchmark.py
# ══════════════════════════════════════════════════════════════════════

def run_ict_benchmark(
    model,
    tokenizer,
    cfg,
    charts: List[List[Candle]],
    device=None,
    verbose: bool = True,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Chạy benchmark ICT đầy đủ (Swept/FVG/Shift), trả về dict điểm trung
    bình mỗi loại + "ict_total" (trung bình 3 loại) — cùng format trả về
    style run_all() để dễ ghép vào log_bench()/checkpoint tracking.
    """
    from benchmark import run_logprob_benchmark   # import trễ, tránh vòng lặp import nếu cần

    device = device or next(model.parameters()).device
    max_seq = cfg.model.max_seq

    items = build_ict_bench_items(charts, seed=seed)

    scores = {}
    for kind in ("swept", "fvg", "shift"):
        bench = items[kind]
        if not bench:
            if verbose:
                print(f"  ⚠ Không có BenchItem nào cho '{kind}' — kiểm tra lại `charts` đầu vào.")
            scores[f"ict_{kind}"] = 0.0
            continue
        scores[f"ict_{kind}"] = run_logprob_benchmark(
            model, tokenizer, bench, f"ict_{kind}", device, max_seq, verbose
        )

    present = [v for k, v in scores.items() if items[k.replace("ict_", "")]]
    scores["ict_total"] = sum(present) / len(present) if present else 0.0

    return scores

CANDLES = [
    "<chart> O_493 H_502 L_490 C_500 O_511 H_512 L_507 C_510 O_510 H_523 L_490 C_498 O_498 H_508 L_492 C_500 O_500 H_517 L_497 C_517 O_517 H_523 L_507 C_516 O_532 H_538 L_531 C_536 O_536 H_571 L_533 C_566 O_566 H_579 L_559 C_575 O_576 H_585 L_523 C_548 O_548 H_552 L_530 C_540 O_540 H_540 L_446 C_469 O_475 H_475 L_461 C_464 O_464 H_475 L_435 C_467 O_467 H_484 L_455 C_479 O_479 H_483 L_461 C_468 O_468 H_477 L_449 C_466 O_466 H_477 L_441 C_445 O_446 H_448 L_419 C_428 O_428 H_460 L_428 C_456 </chart>",
    "<chart> O_500 H_501 L_471 C_497 O_497 H_510 L_477 C_500 O_501 H_505 L_500 C_503 O_503 H_523 L_501 C_521 O_521 H_531 L_511 C_527 O_526 H_541 L_524 C_536 O_536 H_550 L_520 C_528 O_528 H_585 L_519 C_575 O_571 H_579 L_570 C_576 O_576 H_598 L_563 C_593 O_593 H_606 L_554 C_567 O_567 H_596 L_563 C_592 O_592 H_638 L_589 C_636 O_636 H_640 L_616 C_633 O_641 H_646 L_637 C_643 O_643 H_657 L_618 C_650 O_650 H_657 L_623 C_634 O_634 H_653 L_621 C_652 O_652 H_746 L_604 C_646 O_646 H_668 L_611 C_620 </chart>",
    "<chart> O_429 H_430 L_393 C_401 O_401 H_411 L_388 C_409 O_406 H_408 L_405 C_406 O_406 H_418 L_399 C_415 O_415 H_448 L_415 C_444 O_444 H_455 L_435 C_451 O_451 H_455 L_432 C_444 O_444 H_456 L_435 C_440 O_440 H_440 L_437 C_438 O_438 H_448 L_433 C_442 O_442 H_447 L_422 C_434 O_435 H_477 L_428 C_470 O_470 H_484 L_466 C_471 O_471 H_477 L_455 C_465 O_462 H_468 L_462 C_465 O_465 H_479 L_457 C_478 O_478 H_493 L_465 C_486 O_486 H_486 L_435 C_446 O_446 H_446 L_401 C_409 O_409 H_413 L_384 C_401 </chart>",
    "<chart> O_577 H_580 L_577 C_578 O_578 H_587 L_572 C_586 O_586 H_597 L_575 C_577 O_577 H_579 L_552 C_574 O_574 H_593 L_562 C_591 O_591 H_609 L_589 C_608 O_612 H_614 L_608 C_609 O_609 H_613 L_565 C_574 O_574 H_592 L_566 C_588 O_588 H_611 L_580 C_583 O_583 H_584 L_548 C_555 O_555 H_566 L_543 C_563 O_561 H_562 L_560 C_561 O_561 H_572 L_554 C_570 O_570 H_601 L_570 C_598 O_598 H_608 L_589 C_604 O_604 H_608 L_586 C_598 O_598 H_610 L_589 C_593 O_593 H_594 L_591 C_592 O_592 H_601 L_587 C_595 </chart>",
    "<chart> O_787 H_789 L_781 C_786 O_786 H_786 L_764 C_779 O_779 H_837 L_771 C_837 O_837 H_863 L_815 C_843 O_843 H_882 L_841 C_877 O_877 H_880 L_821 C_841 O_833 H_837 L_825 C_833 O_833 H_857 L_825 C_834 O_834 H_834 L_702 C_710 O_710 H_752 L_663 C_723 O_723 H_770 L_721 C_760 O_760 H_765 L_734 C_747 O_747 H_754 L_734 C_735 O_735 H_795 L_731 C_788 O_788 H_821 L_780 C_810 O_810 H_811 L_726 C_730 O_729 H_758 L_726 C_756 O_756 H_759 L_713 C_742 O_739 H_740 L_732 C_737 O_737 H_765 L_725 C_732 </chart>",
    "<chart> O_497 H_499 L_466 C_494 O_494 H_509 L_472 C_497 O_498 H_503 L_497 C_501 O_501 H_523 L_499 C_521 O_521 H_532 L_510 C_528 O_527 H_543 L_525 C_538 O_538 H_554 L_520 C_529 O_529 H_592 L_519 C_582 O_576 H_586 L_576 C_583 O_583 H_607 L_568 C_602 O_602 H_615 L_558 C_572 O_572 H_604 L_567 C_600 O_600 H_651 L_596 C_649 O_649 H_653 L_627 C_646 O_654 H_660 L_650 C_657 O_657 H_673 L_629 C_665 O_665 H_672 L_635 C_646 O_646 H_668 L_632 C_667 O_667 H_772 L_613 C_660 O_660 H_685 L_621 C_631 </chart>",
    "<chart> O_544 H_548 L_484 C_489 O_489 H_524 L_488 C_494 O_494 H_529 L_490 C_521 O_521 H_523 L_520 C_522 O_522 H_531 L_507 C_514 O_514 H_520 L_509 C_515 O_514 H_518 L_514 C_517 O_517 H_536 L_517 C_530 O_530 H_535 L_510 C_520 O_520 H_524 L_520 C_522 O_522 H_526 L_496 C_506 O_506 H_525 L_502 C_522 O_523 H_524 L_520 C_522 O_522 H_556 L_521 C_554 O_554 H_560 L_534 C_537 O_537 H_539 L_537 C_538 O_538 H_546 L_513 C_532 O_532 H_559 L_530 C_546 O_546 H_565 L_542 C_558 O_558 H_573 L_551 C_565 </chart>",
    "<chart> O_680 H_714 L_660 C_699 O_699 H_728 L_692 C_724 O_724 H_762 L_723 C_750 O_750 H_754 L_717 C_740 O_740 H_766 L_734 C_742 O_743 H_751 L_741 C_748 O_748 H_799 L_747 C_795 O_795 H_818 L_762 C_777 O_777 H_787 L_745 C_784 O_784 H_809 L_780 C_808 O_808 H_816 L_757 C_763 O_773 H_773 L_759 C_759 O_759 H_785 L_759 C_778 O_778 H_813 L_771 C_810 O_810 H_822 L_796 C_811 O_811 H_830 L_801 C_817 O_817 H_825 L_794 C_812 O_815 H_816 L_813 C_813 O_813 H_831 L_798 C_825 O_825 H_827 L_768 C_796 </chart>",
    "<chart> O_661 H_699 L_652 C_695 O_695 H_704 L_639 C_644 O_644 H_649 L_573 C_579 O_577 H_585 L_576 C_579 O_579 H_600 L_575 C_583 O_583 H_604 L_578 C_589 O_589 H_606 L_583 C_591 O_591 H_611 L_570 C_574 O_575 H_588 L_562 C_578 O_577 H_578 L_575 C_575 O_575 H_580 L_559 C_564 O_564 H_585 L_549 C_565 O_565 H_571 L_532 C_540 O_540 H_552 L_528 C_537 O_537 H_550 L_517 C_548 O_546 H_547 L_543 C_543 O_543 H_555 L_541 C_547 O_547 H_550 L_532 C_539 O_539 H_553 L_523 C_525 O_525 H_537 L_515 C_523 </chart>",
    "<chart> O_439 H_447 L_421 C_427 O_427 H_436 L_415 C_419 O_419 H_419 L_404 C_406 O_404 H_407 L_403 C_404 O_404 H_408 L_383 C_392 O_392 H_414 L_386 C_404 O_404 H_416 L_400 C_410 O_410 H_427 L_409 C_417 O_417 H_420 L_390 C_395 O_391 H_395 L_389 C_391 O_391 H_404 L_374 C_397 O_397 H_398 L_375 C_377 O_377 H_381 L_356 C_360 O_360 H_360 L_336 C_346 O_346 H_371 L_345 C_365 O_363 H_363 L_361 C_361 O_359 H_368 L_359 C_366 O_366 H_381 L_349 C_354 O_354 H_373 L_339 C_369 O_369 H_383 L_354 C_362 </chart>",
    "<chart> O_493 H_526 L_467 C_521 O_529 H_530 L_525 C_528 O_528 H_528 L_448 C_462 O_462 H_475 L_443 C_450 O_450 H_487 L_448 C_477 O_477 H_505 L_456 C_497 O_497 H_500 L_477 C_485 O_480 H_485 L_478 C_483 O_483 H_492 L_479 C_483 O_483 H_489 L_471 C_474 O_474 H_494 L_465 C_480 O_480 H_492 L_464 C_466 O_466 H_475 L_457 C_469 O_480 H_480 L_473 C_474 O_474 H_486 L_462 C_465 O_465 H_472 L_459 C_459 O_459 H_476 L_452 C_460 O_460 H_467 L_441 C_445 O_445 H_449 L_419 C_448 O_450 H_450 L_447 C_447 </chart>",
    "<chart> O_388 H_411 L_386 C_395 O_395 H_402 L_371 C_377 O_377 H_382 L_358 C_374 O_375 H_376 L_370 C_371 O_370 H_378 L_365 C_368 O_368 H_377 L_331 C_333 O_332 H_340 L_310 C_321 O_321 H_334 L_309 C_313 O_313 H_336 L_300 C_327 O_330 H_332 L_324 C_327 O_327 H_361 L_323 C_358 O_358 H_365 L_341 C_353 O_353 H_362 L_327 C_350 O_350 H_351 L_305 C_311 O_311 H_318 L_249 C_270 O_274 H_276 L_271 C_275 O_275 H_299 L_252 C_259 O_259 H_272 L_238 C_263 O_242 H_256 L_218 C_223 O_223 H_238 L_214 C_230 </chart>",
    "<chart> O_515 H_526 L_509 C_517 O_517 H_530 L_508 C_516 O_516 H_539 L_503 C_531 O_527 H_530 L_527 C_529 O_528 H_533 L_516 C_525 O_524 H_536 L_521 C_523 O_523 H_534 L_515 C_529 O_529 H_552 L_520 C_542 O_543 H_604 L_535 C_597 O_596 H_601 L_596 C_597 O_597 H_648 L_592 C_640 O_640 H_678 L_635 C_659 O_659 H_693 L_653 C_686 O_685 H_710 L_682 C_706 O_706 H_752 L_695 C_729 O_731 H_737 L_727 C_733 O_734 H_744 L_724 C_736 O_736 H_738 L_690 C_699 O_699 H_731 L_699 C_725 O_725 H_727 L_693 C_706 </chart>",
    "<chart> O_765 H_765 L_697 C_714 O_714 H_731 L_679 C_727 O_727 H_761 L_725 C_753 O_753 H_755 L_714 C_719 O_724 H_726 L_714 C_715 O_715 H_749 L_714 C_740 O_740 H_744 L_703 C_716 O_716 H_759 L_710 C_756 O_756 H_786 L_743 C_775 O_776 H_783 L_751 C_774 O_770 H_774 L_769 C_772 O_772 H_774 L_712 C_723 O_723 H_731 L_672 C_711 O_711 H_744 L_711 C_738 O_738 H_761 L_715 C_742 O_742 H_746 L_714 C_724 O_724 H_724 L_719 C_724 O_724 H_742 L_710 C_734 O_734 H_754 L_714 C_718 O_718 H_737 L_709 C_723 </chart>",
    "<chart> O_548 H_553 L_521 C_526 O_527 H_530 L_524 C_528 O_528 H_528 L_482 C_491 O_491 H_504 L_487 C_499 O_499 H_499 L_471 C_476 O_476 H_488 L_461 C_487 O_487 H_513 L_485 C_488 O_491 H_491 L_487 C_489 O_488 H_490 L_441 C_451 O_451 H_467 L_437 C_457 O_457 H_485 L_449 C_485 O_485 H_502 L_461 C_463 O_463 H_483 L_454 C_473 O_473 H_476 L_469 C_471 O_471 H_472 L_438 C_441 O_441 H_453 L_420 C_424 O_418 H_445 L_416 C_441 O_441 H_446 L_393 C_405 O_405 H_413 L_380 C_394 O_397 H_397 L_392 C_396 </chart>",
    "<chart> O_532 H_551 L_517 C_548 O_548 H_568 L_546 C_564 O_564 H_607 L_564 C_607 O_603 H_613 L_603 C_611 O_611 H_611 L_504 C_509 O_509 H_539 L_508 C_530 O_530 H_554 L_525 C_534 O_534 H_546 L_519 C_536 O_536 H_561 L_531 C_551 O_545 H_546 L_540 C_542 O_542 H_548 L_520 C_539 O_539 H_551 L_531 C_539 O_539 H_553 L_530 C_547 O_547 H_551 L_521 C_531 O_531 H_542 L_512 C_530 O_540 H_545 L_533 C_534 O_544 H_571 L_532 C_556 O_556 H_591 L_554 C_589 O_589 H_612 L_570 C_611 O_611 H_617 L_569 C_575 </chart>",
    "<chart> O_735 H_762 L_733 C_755 O_754 H_756 L_753 C_755 O_755 H_765 L_751 C_759 O_759 H_763 L_737 C_750 O_750 H_775 L_723 C_736 O_736 H_770 L_727 C_763 O_763 H_800 L_760 C_797 O_796 H_797 L_794 C_795 O_795 H_809 L_789 C_801 O_801 H_838 L_797 C_832 O_832 H_844 L_824 C_830 O_830 H_859 L_829 C_846 O_846 H_848 L_817 C_832 O_835 H_839 L_834 C_837 O_837 H_840 L_799 C_810 O_810 H_847 L_808 C_834 O_834 H_837 L_815 C_834 O_834 H_837 L_812 C_830 O_830 H_844 L_807 C_827 O_826 H_826 L_824 C_824 </chart>",
    "<chart> O_519 H_520 L_508 C_513 O_513 H_530 L_506 C_527 O_527 H_562 L_519 C_560 O_560 H_579 L_559 C_572 O_571 H_575 L_552 C_570 O_570 H_602 L_570 C_587 O_581 H_586 L_581 C_583 O_583 H_588 L_547 C_558 O_558 H_581 L_551 C_579 O_579 H_582 L_561 C_576 O_576 H_614 L_575 C_612 O_612 H_633 L_607 C_615 O_616 H_620 L_600 C_609 O_609 H_646 L_588 C_629 O_629 H_661 L_623 C_656 O_656 H_697 L_655 C_684 O_684 H_688 L_649 C_673 O_673 H_701 L_667 C_676 O_676 H_685 L_675 C_682 O_682 H_736 L_681 C_732 </chart>",
    "<chart> O_396 H_404 L_385 C_395 O_395 H_398 L_387 C_393 O_393 H_421 L_386 C_393 O_393 H_397 L_374 C_382 O_382 H_382 L_321 C_328 O_325 H_326 L_224 C_268 O_268 H_313 L_250 C_284 O_284 H_294 L_266 C_278 O_278 H_316 L_274 C_313 O_313 H_321 L_298 C_314 O_314 H_350 L_312 C_350 O_349 H_350 L_348 C_349 O_349 H_363 L_338 C_361 O_360 H_371 L_351 C_359 O_359 H_369 L_347 C_360 O_360 H_367 L_342 C_349 O_349 H_362 L_348 C_352 O_352 H_353 L_347 C_349 O_348 H_386 L_346 C_384 O_384 H_390 L_378 C_380 </chart>",
    "<chart> O_522 H_535 L_513 C_517 O_517 H_518 L_515 C_515 O_515 H_526 L_510 C_520 O_520 H_525 L_499 C_512 O_512 H_558 L_506 C_550 O_550 H_565 L_546 C_551 O_551 H_557 L_534 C_544 O_542 H_548 L_542 C_544 O_544 H_559 L_536 C_558 O_558 H_574 L_544 C_567 O_567 H_567 L_513 C_524 O_524 H_524 L_477 C_485 O_485 H_489 L_459 C_477 O_473 H_475 L_471 C_473 O_473 H_489 L_467 C_486 O_486 H_503 L_483 C_491 O_491 H_507 L_481 C_502 O_502 H_504 L_472 C_499 O_499 H_513 L_477 C_502 O_503 H_508 L_502 C_505 </chart>",
]
def run_all_ict_benchmarks(
    model,
    tokenizer,
    cfg,
    device=None,
    verbose: bool = True,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Wrapper tiện lợi để chạy benchmark ICT đầy đủ (Swept/FVG/Shift) + các
    benchmark khác trong benchmark.py (nếu muốn) — trả về dict điểm trung
    bình mỗi loại + "ict_total" (trung bình 3 loại) + các benchmark khác.
    """
    
    charts = []
    for c in CANDLES:
        candles = CandleParser(raw_text=c, swing_window=2).candles
        charts.append(candles)
    scores = run_ict_benchmark(model, tokenizer, cfg, charts, device, verbose, seed)
    # Có thể thêm các benchmark khác từ benchmark.py nếu muốn
    return scores


if __name__ == "__main__":
    from generate import load_model_for_inference

    model, tokenizer, cfg = load_model_for_inference("data/check_point/last_cp.pt")
    run_all_ict_benchmarks(model, tokenizer, cfg, verbose=True)