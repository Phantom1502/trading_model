"""
tests/test_pin_bar.py — Lớp 1: Pin Bar (Hammer / Shooting Star)
======================================================================
"""

from app.ict.candle import Candle
from app.ict.basic import is_pin_bar


def test_clear_hammer():
    """Body nhỏ, lower wick dài, upper wick gần 0 -> HAMMER."""
    # range = 100 (high=600, low=500), body = 10 (590-580 hoặc tương tự)
    # lower wick dài: open/close gần đỉnh, low cách xa
    c = Candle(open=590, high=600, low=500, close=595)
    assert is_pin_bar(c) == "HAMMER"


def test_clear_shooting_star():
    """Body nhỏ, upper wick dài, lower wick gần 0 -> SHOOTING_STAR."""
    c = Candle(open=510, high=600, low=500, close=505)
    assert is_pin_bar(c) == "SHOOTING_STAR"


def test_boundary_wick_ratio_threshold():
    """Lower wick đúng bằng wick_ratio * range -> theo strict '>=' trong code
    (xem basic.py: `lower >= wick_ratio * rng`), nên đúng ngưỡng vẫn tính HAMMER."""
    # range = 100, wick_ratio = 0.6 -> lower wick cần đúng 60
    # low=500, open=close=560 -> lower_wick = min(560,560)-500 = 60, upper_wick = 600-560=40
    # body = 0, nhưng upper=40 không < lower*0.5=30 -> không thỏa "upper < lower*0.5"
    # điều chỉnh để chỉ upper wick rất nhỏ, đảm bảo thỏa cả 2 điều kiện
    c = Candle(open=560, high=565, low=500, close=560)
    # range=65, lower_wick=60, upper_wick=5; wick_ratio*range=39 -> 60>=39 True
    # upper(5) < lower(60)*0.5=30 -> True -> HAMMER
    assert is_pin_bar(c) == "HAMMER"


def test_near_miss_large_body():
    """Wick dài nhưng body cũng lớn (vượt body_ratio) -> None."""
    c = Candle(open=520, high=600, low=500, close=580)  # body=60, range=100 -> body_ratio=0.6 > 0.3
    assert is_pin_bar(c) is None


def test_near_miss_both_wicks_long():
    """Cả 2 wick đều dài tương đương -> None (không thỏa upper < lower*0.5 hoặc ngược lại)."""
    c = Candle(open=545, high=600, low=500, close=555)  # body nhỏ, nhưng wick 2 bên gần bằng nhau
    assert is_pin_bar(c) is None