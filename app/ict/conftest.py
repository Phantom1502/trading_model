"""
conftest.py — fixture dùng chung
======================================
"""

import pytest


@pytest.fixture
def make_candle():
    """Helper tạo Candle nhanh trong test, tránh import lặp."""
    from app.ict.candle import Candle
    return Candle