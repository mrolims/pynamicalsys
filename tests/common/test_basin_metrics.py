"""
BasinMetrics.basin_entropy.

In a checkerboard of two states, every sub-box of even side holds both states
in equal numbers, so its entropy is log(2): one bit with log_base=2. The
logarithm base must be positive and different from 1, since log(1) = 0.
"""

import numpy as np
import pytest

from pynamicalsys import BasinMetrics


@pytest.fixture
def checkerboard():
    i, j = np.indices((20, 20))
    return BasinMetrics(((i + j) % 2).astype(np.float64))


def test_checkerboard_has_one_bit_per_sub_box(checkerboard):
    assert checkerboard.basin_entropy(n=4, log_base=2) == pytest.approx((1.0, 1.0))
    assert checkerboard.basin_entropy(n=4) == pytest.approx((np.log(2), np.log(2)))


@pytest.mark.parametrize("log_base, error", [(1, ValueError), (0, ValueError), (-2, ValueError), ("2", TypeError), (True, TypeError)])
def test_log_base_must_be_a_positive_number_other_than_one(checkerboard, log_base, error):
    with pytest.raises(error, match="log_base"):
        checkerboard.basin_entropy(n=4, log_base=log_base)
