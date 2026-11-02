"""
DiscreteDynamicalSystem.escape_analysis with escape="entering".

`hole_size` broadcasts to `(n_exits, system_dimension)`, so holes can differ
between coordinates and between each other, and an infinite side length leaves
a coordinate unconstrained. `periods` wraps the holes around periodic
coordinates, and defaults to `2 * pi` for the first coordinate only.
"""

import numpy as np
import pytest
from numba import njit

from pynamicalsys import DiscreteDynamicalSystem as DDS


@njit
def fixed_point(u, parameters):
    return u.copy()


@njit
def doubling(u, parameters):
    return 2.0 * u


@pytest.fixture
def system():
    return DDS(mapping=fixed_point, system_dimension=2, number_of_parameters=0)


def test_scalar_and_full_array_agree():
    ds = DDS(model="standard map")
    ds.set_parameters([1.5])
    centers = np.array([[0.2, 0.3], [0.7, 0.8]])
    periods = [1.0, 1.0]
    for x in np.linspace(0.0, 1.0, 20, endpoint=False):
        kwargs = dict(exits=centers, escape="entering", periods=periods)
        scalar = ds.escape_analysis([x, 0.41], 500, hole_size=0.05, **kwargs)
        full = ds.escape_analysis(
            [x, 0.41], 500, hole_size=np.full((2, 2), 0.05), **kwargs
        )
        assert scalar == full


@pytest.mark.parametrize("y", [-1e6, -0.3, 0.0, 0.9, 1e6])
def test_infinite_side_leaves_a_coordinate_unconstrained(system, y):
    assert system.escape_analysis(
        [1.0, y], 5, exits=[1.0, 0.0], hole_size=[0.2, np.inf]
    ) == (0, 1)


def test_strip_rejects_points_outside_its_width(system):
    assert system.escape_analysis(
        [1.2, 0.0], 5, exits=[1.0, 0.0], hole_size=[0.2, np.inf]
    ) == (-1, 5)


def test_per_coordinate_sides_differ(system):
    kwargs = dict(exits=[1.0, 1.0], hole_size=[0.2, 1.0])
    assert system.escape_analysis([1.0, 1.4], 5, **kwargs) == (0, 1)
    assert system.escape_analysis([1.4, 1.0], 5, **kwargs) == (-1, 5)


def test_per_hole_sizes(system):
    centers = [[1.0, 1.0], [3.0, 3.0]]
    sizes = [[0.2], [1.0]]
    assert system.escape_analysis([3.4, 3.4], 5, exits=centers, hole_size=sizes) == (
        1,
        1,
    )
    assert system.escape_analysis([1.4, 1.4], 5, exits=centers, hole_size=sizes) == (
        -1,
        5,
    )


def test_per_hole_and_coordinate_sizes(system):
    centers = [[1.0, 1.0], [3.0, 3.0]]
    sizes = [[0.2, np.inf], [np.inf, 0.2]]
    assert system.escape_analysis([1.0, 50.0], 5, exits=centers, hole_size=sizes) == (
        0,
        1,
    )
    assert system.escape_analysis([50.0, 3.0], 5, exits=centers, hole_size=sizes) == (
        1,
        1,
    )


def test_hole_at_zero_wraps_by_default(system):
    kwargs = dict(exits=[0.0, 0.0], hole_size=[0.2, np.inf])
    below = 2.0 * np.pi - 0.05
    assert system.escape_analysis([below, 0.0], 5, **kwargs) == (0, 1)
    assert system.escape_analysis([0.05, 0.0], 5, **kwargs) == (0, 1)
    assert system.escape_analysis([np.pi, 0.0], 5, **kwargs) == (-1, 5)


def test_wrapping_can_be_turned_off(system):
    assert system.escape_analysis(
        [2.0 * np.pi - 0.05, 0.0],
        5,
        exits=[0.0, 0.0],
        hole_size=[0.2, np.inf],
        periods=[np.inf, np.inf],
    ) == (-1, 5)


def test_default_period_does_not_wrap_the_other_coordinates(system):
    assert system.escape_analysis(
        [0.0, 2.0 * np.pi], 5, exits=[0.0, 0.0], hole_size=0.2
    ) == (-1, 5)


def test_custom_period(system):
    assert system.escape_analysis(
        [0.98, 0.0],
        5,
        exits=[0.0, 0.0],
        hole_size=[0.1, np.inf],
        periods=[1.0, np.inf],
    ) == (0, 1)


def test_hole_on_the_other_coordinate_with_a_wrapped_first_coordinate(system):
    kwargs = dict(exits=[0.0, 1.0], hole_size=[np.inf, 0.2])
    assert system.escape_analysis([4.0, 1.05], 5, **kwargs) == (0, 1)
    assert system.escape_analysis([4.0, 1.5], 5, **kwargs) == (-1, 5)


def test_hole_wider_than_the_period_covers_the_circle(system):
    assert system.escape_analysis(
        [4.0, 0.0], 5, exits=[0.0, 0.0], hole_size=[7.0, 1.0]
    ) == (0, 1)


def test_diverging_trajectory_does_not_enter_a_hole():
    ds = DDS(mapping=doubling, system_dimension=2, number_of_parameters=0)
    max_time = 3000  # 2 ** 3000 overflows to infinity
    assert ds.escape_analysis(
        [1.0, 1.0], max_time, exits=[0.5, 0.5], hole_size=0.1
    ) == (-1, max_time)
    assert ds.escape_analysis(
        [1.0, 1.0],
        max_time,
        exits=[0.5, 0.5],
        hole_size=0.1,
        periods=[np.inf, np.inf],
    ) == (-1, max_time)


def test_nan_state_does_not_enter_a_hole():
    @njit
    def to_nan(u, parameters):
        return np.full(2, np.nan)

    ds = DDS(mapping=to_nan, system_dimension=2, number_of_parameters=0)
    for periods in (None, [np.inf, np.inf]):
        assert ds.escape_analysis(
            [0.0, 0.0], 5, exits=[0.0, 0.0], hole_size=0.1, periods=periods
        ) == (-1, 5)


@pytest.mark.parametrize(
    "hole_size",
    [0.0, -1.0, np.nan, [0.2, 0.0], [0.2, np.nan], [0.1, 0.2, 0.3], [[0.1], [0.2]]],
)
def test_invalid_hole_size_values(system, hole_size):
    with pytest.raises(ValueError, match="hole_size"):
        system.escape_analysis([1.0, 1.0], 5, exits=[0.0, 0.0], hole_size=hole_size)


@pytest.mark.parametrize("hole_size", ["wide", True])
def test_non_numeric_hole_size(system, hole_size):
    with pytest.raises(TypeError, match="hole_size"):
        system.escape_analysis([1.0, 1.0], 5, exits=[0.0, 0.0], hole_size=hole_size)


def test_missing_hole_size(system):
    with pytest.raises(ValueError, match="hole_size"):
        system.escape_analysis([1.0, 1.0], 5, exits=[0.0, 0.0])


@pytest.mark.parametrize("periods", [[1.0], [1.0, 1.0, 1.0], [0.0, 1.0], [-1.0, 1.0]])
def test_invalid_periods(system, periods):
    with pytest.raises(ValueError, match="periods"):
        system.escape_analysis(
            [1.0, 1.0], 5, exits=[0.0, 0.0], hole_size=0.1, periods=periods
        )
