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


# Rectangular sub-boxes: nx grid points along the first axis of the basin and
# ny along the second, each defaulting to n.


@pytest.fixture
def stripes():
    # The state alternates along the first axis and is constant along the second.
    i, _ = np.indices((12, 12))
    return BasinMetrics((i % 2).astype(np.float64))


def test_boxes_one_point_tall_contain_a_single_state(stripes):
    assert stripes.basin_entropy(n=4, nx=1, log_base=2) == pytest.approx((0.0, 0.0))


def test_boxes_two_points_tall_contain_both_states(stripes):
    # With n = 1 alone, every box would hold a single point and no entropy.
    assert stripes.basin_entropy(n=1, nx=2, ny=3, log_base=2) == pytest.approx((1.0, 1.0))


def test_ny_alone_keeps_n_along_the_first_axis():
    # In a checkerboard, a box of 3 x 1 holds the states two to one, while a
    # square box of 3 x 3 would hold them five to four.
    i, j = np.indices((12, 12))
    metrics = BasinMetrics(((i + j) % 2).astype(np.float64))
    entropy = -(2 / 3) * np.log2(2 / 3) - (1 / 3) * np.log2(1 / 3)
    assert metrics.basin_entropy(n=3, ny=1, log_base=2) == pytest.approx((entropy, entropy))


@pytest.mark.parametrize("name, value, error", [("nx", 5, ValueError), ("ny", 0, ValueError), ("nx", 2.0, TypeError), ("ny", True, TypeError)])
def test_box_sizes_must_be_positive_integers_that_divide_the_basin(stripes, name, value, error):
    with pytest.raises(error, match=name):
        stripes.basin_entropy(n=4, **{name: value})
