"""
DiscreteDynamicalSystem.survival_probability.

`escape_analysis` returns `(-1, max_time)` for a trajectory that does not
escape within `max_time`, so an escape time equal to `max_time` counts as a
trajectory that has not escaped: it survives at every evaluation time up to
`max_time`, and the survival probability at `max_time` is the fraction of such
trajectories.
"""

import numpy as np
import pytest

from pynamicalsys import DiscreteDynamicalSystem as DDS


@pytest.fixture
def system():
    ds = DDS(model="standard map")
    ds.set_parameters([1.5])
    return ds


def test_escape_times_equal_to_max_time_have_not_escaped(system):
    escape_times = [1, 2, 3, 7, 10, 10]
    t, p = system.survival_probability(escape_times, 10)

    np.testing.assert_array_equal(t, np.arange(1, 11))
    expected = [5, 4, 3, 3, 3, 3, 2, 2, 2, 2]
    np.testing.assert_allclose(p, np.array(expected) / 6)


def test_survival_at_max_time_is_the_fraction_that_did_not_escape(system):
    rng = np.random.default_rng(1)
    max_time = 200
    exits = np.array([[0.0, 1.0], [0.4, 0.6]])
    results = np.array(
        [
            system.escape_analysis([x, 0.5], max_time, exits=exits, escape="exiting")
            for x in rng.uniform(0.0, 1.0, 500)
        ]
    )
    trapped = results[:, 0] == -1
    assert 0 < trapped.sum() < len(results)

    _, p = system.survival_probability(results[:, 1], max_time)
    assert p[-1] == pytest.approx(trapped.mean())
