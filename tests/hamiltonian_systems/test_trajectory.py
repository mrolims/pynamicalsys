"""
HamiltonianSystem.trajectory with sample_times.

The integrators use a constant step, so each sample time is rounded to the
nearest integration step, as in lyapunov, SALI, LDI, and GALI of this class.
The sampled rows must therefore be the rows of the full trajectory at those
steps, with the same values, and every trajectory of an ensemble is stored at
the same times.
"""

import numpy as np
import pytest

from pynamicalsys import HamiltonianSystem as HS

Q = [0.0, 0.1]
P = [0.3, 0.0]


@pytest.fixture
def system():
    hs = HS(model="henon heiles")
    hs.integrator("svy4", time_step=0.01)
    return hs


def test_sampled_rows_are_rows_of_the_full_trajectory(system):
    full = system.trajectory(Q, P, 10.0)
    sampled = system.trajectory(Q, P, 10.0, sample_times=[2.5, 0.5, 7.0, 2.5])
    # Sorted, without duplicates, at steps 50, 250, and 700.
    np.testing.assert_array_equal(sampled, full[[50, 250, 700]])


def test_sample_times_are_rounded_to_the_nearest_step(system):
    full = system.trajectory(Q, P, 1.0)
    sampled = system.trajectory(Q, P, 1.0, sample_times=[0.123, 0.4449])
    np.testing.assert_array_equal(sampled, full[[12, 44]])


def test_ensemble_is_sampled_at_the_same_times(system):
    q = np.array([Q, [0.0, 0.2]])
    p = np.array([P, [0.25, 0.0]])
    full = system.trajectory(q, p, 5.0)
    sampled = system.trajectory(q, p, 5.0, sample_times=[1.0, 3.0, 5.0])
    np.testing.assert_array_equal(sampled, full[:, [100, 300, 500]])


@pytest.mark.parametrize("sample_times", [[0.0], [-1.0], [10.5], [[1.0, 2.0]]])
def test_sample_times_must_lie_in_the_interval(system, sample_times):
    with pytest.raises(ValueError):
        system.trajectory(Q, P, 10.0, sample_times=sample_times)
