from typing import Optional

from app.trading.core.candle import Candle

DOJI_THRESHOLD_BINS = 2     # |Close - Open| phải > giá trị này mới tính Bull/Bear
                            # Thống kê: 17.3% nến rơi vào DOJI với threshold=2,
                            # BULL 41.5% / BEAR 41.2% — cân đối, hợp lý cho M1
                            # (nhiều giai đoạn đi ngang ngoài giờ giao dịch chính).

PIN_BAR_WICK_RATIO  = 0.6   # wick dài >= wick_ratio * range
PIN_BAR_BODY_RATIO  = 0.3   # body <= body_ratio * range
                            # Thống kê: Hammer 7.91% + Shooting Star 7.35% = ~15.3%
                            # tổng số nến. Tỷ lệ này CAO hơn mức "hiếm, nổi bật" điển
                            # hình (~3-8%), nhưng ĐÃ XÁC NHẬN GIỮ NGUYÊN có chủ đích:
                            # Pin Bar ở đây chỉ là 1 component binary trong entry
                            # condition tổng hợp (swept + shift + FVG + price action
                            # tại FVG), không phải bộ lọc cuối cùng. Mức độ "đẹp" của
                            # từng Pin Bar cụ thể sẽ được chấm Q-score graded riêng
                            # (công thức đo lường, không phải ngưỡng binary) — đúng
                            # nguyên tắc Binary vs Graded trong spec mục 3. Ngưỡng
                            # binary rộng ở bước này là hợp lý, việc phân biệt "Pin
                            # Bar đẹp" và "Pin Bar tầm thường" thuộc về bước graded.


class ICTPattern:
    @staticmethod
    def is_pin_bar(
        candle: Candle,
        wick_ratio: float = PIN_BAR_WICK_RATIO,
        body_ratio: float = PIN_BAR_BODY_RATIO
    ) -> bool:
        """
        Trả về "HAMMER" | "SHOOTING_STAR" | None.

        HAMMER         : lower wick dài, upper wick ngắn, body nhỏ.
        SHOOTING_STAR  : upper wick dài, lower wick ngắn, body nhỏ.
        Cả 2 wick đều dài (gần bằng nhau) -> None, không phải Pin Bar
        (xem tests/test_pin_bar.py — near_miss_both_wicks_long).
        """
        rng, body = candle.range(), candle.body()
        if body > body_ratio * rng:
            return None

        upper, lower = candle.upper_wick(), candle.lower_wick()

        if lower >= wick_ratio * rng and upper < lower * 0.5:
            return "HAMMER"
        if upper >= wick_ratio * rng and lower < upper * 0.5:
            return "SHOOTING_STAR"
        return None
    
    @staticmethod
    def classify_direction(candle: Candle, threshold_bins: int = DOJI_THRESHOLD_BINS) -> str:
        """
        Trả về "BULL" | "BEAR" | "DOJI".

        Dùng strict '>' (không phải '>='): body phải VƯỢT threshold mới được
        tính Bull/Bear, đúng bằng threshold vẫn là DOJI. Xem
        tests/test_bull_bear.py — boundary_at_threshold / boundary_just_below_threshold
        để biết rõ behavior này.
        """
        diff = candle.close - candle.open
        if diff > threshold_bins:
            return "BULL"
        if diff < -threshold_bins:
            return "BEAR"
        return "DOJI"
    
    @staticmethod
    def is_engulfing(prev: Candle, curr: Candle) -> Optional[str]:
        """
        Trả về "BULLISH_ENGULFING" | "BEARISH_ENGULFING" | None.

        Dùng '<=' / '>=' (non-strict) cho điều kiện "nuốt trọn" — khớp đúng
        biên (Open[curr] == Close[prev]) vẫn tính là engulfing. Xem
        tests/test_engulfing.py — boundary_exact_engulf.

        Caller chịu trách nhiệm xử lý edge_position (index=0, không có prev) —
        hàm này không tự bắt index âm, nhận thẳng 2 Candle.
        """
        prev_dir = ICTPattern.classify_direction(prev)
        curr_dir = ICTPattern.classify_direction(curr)

        if prev_dir == "BEAR" and curr_dir == "BULL":
            if curr.open <= prev.close and curr.close >= prev.open:
                return "BULLISH_ENGULFING"

        if prev_dir == "BULL" and curr_dir == "BEAR":
            if curr.open >= prev.close and curr.close <= prev.open:
                return "BEARISH_ENGULFING"

        return None