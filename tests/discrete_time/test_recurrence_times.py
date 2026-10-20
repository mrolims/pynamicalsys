"""
Recurrence times with the box wrapped around the boundaries of periodic
coordinates.

The reference values are computed independently in NumPy from the stored
trajectory: the distance of each iterate to the reference point is measured
modulo the period of each coordinate, and the recurrence times are the gaps
between successive iterations inside the box.
"""

import numpy as np
import pytest

from pynamicalsys import DiscreteDynamicalSystem as DDS

TOTAL_TIME = 200_000
EPS = 0.02


@pytest.fixture
def system():
    ds = DDS(model="standard map")
    ds.set_parameters([1.5])
    return ds


def expected_recurrence_times(system, u, periods):
    # trajectory returns iterates 1, ..., TOTAL_TIME, without the initial state.
    orbit = system.trajectory(u, TOTAL_TIME)
    difference = orbit - np.asarray(u)
    finite = np.isfinite(periods)
    half = 0.5 * periods[finite]
    difference[:, finite] = (difference[:, finite] + half) % periods[finite] - half
    inside = np.all(np.abs(difference) <= EPS / 2, axis=1)
    returns = np.flatnonzero(inside) + 1
    return np.diff(np.concatenate(([0], returns))).astype(np.float64)


@pytest.mark.parametrize("u", [[0.005, 0.3], [0.3, 0.995], [0.002, 0.996]])
def test_box_wraps_around_periodic_boundaries(system, u):
    periods = np.array([1.0, 1.0])
    times = system.recurrence_times(u, TOTAL_TIME, eps=EPS, periods=periods)

    np.testing.assert_array_equal(times, expected_recurrence_times(system, u, periods))
    # Near a boundary, wrapping must find returns that the plain box misses.
    assert times.size > system.recurrence_times(u, TOTAL_TIME, eps=EPS).size


def test_only_the_periodic_coordinate_wraps(system):
    u = [0.005, 0.3]
    periods = np.array([1.0, np.inf])
    times = system.recurrence_times(u, TOTAL_TIME, eps=EPS, periods=periods)

    np.testing.assert_array_equal(times, expected_recurrence_times(system, u, periods))


def test_without_periods_nothing_wraps(system):
    u = [0.005, 0.3]
    np.testing.assert_array_equal(
        system.recurrence_times(u, TOTAL_TIME, eps=EPS),
        system.recurrence_times(u, TOTAL_TIME, eps=EPS, periods=[np.inf, np.inf]),
    )


def test_periods_far_from_the_boundaries_change_nothing(system):
    u = [0.2, 0.3]
    np.testing.assert_array_equal(
        system.recurrence_times(u, TOTAL_TIME, eps=EPS),
        system.recurrence_times(u, TOTAL_TIME, eps=EPS, periods=[1.0, 1.0]),
    )


@pytest.mark.parametrize("periods", [[1.0], [1.0, 1.0, 1.0], [1.0, 0.0], [-1.0, 1.0]])
def test_invalid_periods_raise(system, periods):
    with pytest.raises(ValueError, match="periods"):
        system.recurrence_times([0.2, 0.3], 100, periods=periods)
