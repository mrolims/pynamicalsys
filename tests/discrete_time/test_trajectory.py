"""
DiscreteDynamicalSystem.trajectory: sample_times and the shape of ensembles.

The sampled states are checked against the full trajectory, which stores the
state after every iteration, so row k of the full trajectory is the state
after iteration k + 1.
"""

import warnings

import numpy as np
import pytest

from pynamicalsys import DiscreteDynamicalSystem as DDS

TOTAL_TIME = 1_000


@pytest.fixture
def system():
    ds = DDS(model="standard map")
    ds.set_parameters([1.5])
    return ds


@pytest.fixture
def logistic():
    ds = DDS(model="logistic map")
    ds.set_parameters([3.9])
    return ds


ENSEMBLE = np.random.default_rng(3).uniform(size=(4, 2))


def test_sample_times_pick_rows_of_the_full_trajectory(system):
    full = system.trajectory([0.2, 0.5], TOTAL_TIME)
    sample_times = [500, 1, 17, 17, 1_000]
    sampled = system.trajectory([0.2, 0.5], TOTAL_TIME, sample_times=sample_times)
    np.testing.assert_array_equal(sampled, full[np.unique(sample_times) - 1])


def test_sample_times_count_from_the_initial_condition(system):
    # With a transient, sample times are still absolute iterations.
    full = system.trajectory([0.2, 0.5], TOTAL_TIME)
    sampled = system.trajectory(
        [0.2, 0.5], TOTAL_TIME, transient_time=100, sample_times=[101, 400]
    )
    np.testing.assert_array_equal(sampled, full[[100, 399]])


@pytest.mark.parametrize("sample_times", [[0, 10], [10, TOTAL_TIME + 1]])
def test_sample_times_outside_the_range_raise(system, sample_times):
    with pytest.raises(ValueError, match="sample_times"):
        system.trajectory([0.2, 0.5], TOTAL_TIME, sample_times=sample_times)


def test_sample_times_at_or_before_the_transient_raise(system):
    with pytest.raises(ValueError, match="sample_times"):
        system.trajectory(
            [0.2, 0.5], TOTAL_TIME, transient_time=100, sample_times=[100, 200]
        )


def test_ensemble_without_concatenate_is_deprecated(system):
    with pytest.warns(DeprecationWarning, match="concatenate"):
        trajectories = system.trajectory(ENSEMBLE, TOTAL_TIME)
    assert trajectories.shape == (4 * TOTAL_TIME, 2)


def test_ensemble_with_concatenate_false(system):
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        stacked = system.trajectory(ENSEMBLE, TOTAL_TIME, concatenate=False)
        concatenated = system.trajectory(ENSEMBLE, TOTAL_TIME, concatenate=True)
    assert stacked.shape == (4, TOTAL_TIME, 2)
    np.testing.assert_array_equal(stacked.reshape(-1, 2), concatenated)
    for i, initial_condition in enumerate(ENSEMBLE):
        np.testing.assert_array_equal(
            stacked[i], system.trajectory(initial_condition, TOTAL_TIME)
        )


def test_ensemble_with_sample_times(system):
    sample_times = [10, 100, 1_000]
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        sampled = system.trajectory(ENSEMBLE, TOTAL_TIME, sample_times=sample_times)
    full = system.trajectory(ENSEMBLE, TOTAL_TIME, concatenate=False)
    np.testing.assert_array_equal(sampled, full[:, np.array(sample_times) - 1])
    concatenated = system.trajectory(
        ENSEMBLE, TOTAL_TIME, sample_times=sample_times, concatenate=True
    )
    np.testing.assert_array_equal(concatenated, sampled.reshape(-1, 2))


def test_one_dimensional_shapes(logistic):
    ensemble = [[0.3], [0.4], [0.5]]
    assert logistic.trajectory(0.3, 5).shape == (5,)
    assert logistic.trajectory(0.3, 5, sample_times=[2, 5]).shape == (2,)
    assert logistic.trajectory(ensemble, 5, concatenate=False).shape == (3, 5)
    assert logistic.trajectory(ensemble, 5, sample_times=[2, 5]).shape == (3, 2)
    assert logistic.trajectory(ensemble, 5, concatenate=True).shape == (15, 1)


def test_concatenate_must_be_boolean(system):
    with pytest.raises(TypeError, match="concatenate"):
        system.trajectory(ENSEMBLE, TOTAL_TIME, concatenate=1)
