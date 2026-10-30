"""
The early-stopping value of the discrete SALI, LDI, and GALI is called
`threshold`, as in ContinuousDynamicalSystem and HamiltonianSystem. The former
name `tol` still works but emits a DeprecationWarning.
"""

import warnings

import numpy as np
import pytest

from pynamicalsys import DiscreteDynamicalSystem as DDS

# A chaotic orbit of the standard map, whose indicators fall below 1e-8 well
# within 5000 iterations, so the threshold visibly changes the result.
U = [0.05, 0.05]
TOTAL_TIME = 5_000


@pytest.fixture
def system():
    ds = DDS(model="standard map")
    ds.set_parameters([1.5])
    return ds


INDICATORS = {
    "SALI": lambda ds, **kw: ds.SALI(U, TOTAL_TIME, **kw),
    "LDI": lambda ds, **kw: ds.LDI(U, TOTAL_TIME, 2, **kw),
    "GALI": lambda ds, **kw: ds.GALI(U, TOTAL_TIME, 2, **kw),
}


@pytest.mark.parametrize("name", INDICATORS)
def test_threshold_stops_the_computation(system, name):
    compute = INDICATORS[name]
    stopped = compute(system, threshold=1e-8, return_history=True)
    full = compute(system, threshold=0.0, return_history=True)

    # Every iteration is recorded, so the history ends exactly at the stop.
    stop = np.argmax(full < 1e-8)
    assert stop > 0
    np.testing.assert_array_equal(stopped, full[: stop + 1])


@pytest.mark.parametrize("name", INDICATORS)
def test_history_is_truncated_at_the_stop_with_sample_times(system, name):
    compute = INDICATORS[name]
    sample_times = np.arange(10, TOTAL_TIME + 1, 10)
    full = compute(system, threshold=0.0, return_history=True)
    stopped = compute(
        system, threshold=1e-8, return_history=True, sample_times=sample_times
    )

    # The stop happens at iteration stop_iteration; only the sample times up
    # to it are recorded, and there are no trailing zeros.
    stop_iteration = np.argmax(full < 1e-8) + 1
    recorded = sample_times[sample_times <= stop_iteration]
    assert 0 < len(recorded) < len(sample_times)
    np.testing.assert_array_equal(stopped, full[recorded - 1])


@pytest.mark.parametrize("name", INDICATORS)
def test_tol_is_a_deprecated_alias_of_threshold(system, name):
    compute = INDICATORS[name]
    expected = compute(system, threshold=1e-8)

    with pytest.warns(DeprecationWarning, match="Use threshold instead"):
        result = compute(system, tol=1e-8)

    assert result == expected


@pytest.mark.parametrize("name", INDICATORS)
def test_threshold_does_not_warn(system, name):
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        INDICATORS[name](system, threshold=1e-8)


@pytest.mark.parametrize("name", INDICATORS)
def test_passing_both_names_raises(system, name):
    with pytest.raises(TypeError, match="pass only threshold"):
        INDICATORS[name](system, threshold=1e-8, tol=1e-10)
