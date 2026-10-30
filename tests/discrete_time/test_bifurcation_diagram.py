"""
DiscreteDynamicalSystem.bifurcation_diagram: how the parameter vector is built.

Explicit parameters are the ones other than the swept parameter, in order. The
stored parameters, if any, are used when no explicit ones are given.
"""

import numpy as np
import pytest

from pynamicalsys import DiscreteDynamicalSystem as DDS

U0 = [0.1, 0.1]
PARAM_RANGE = (1.0, 1.4, 3)
TOTAL_TIME = 50
TRANSIENT_TIME = 40


def henon(parameters=None):
    ds = DDS(model="henon map")
    if parameters is not None:
        ds.set_parameters(parameters)
    return ds


def sweep(ds, **kwargs):
    return ds.bifurcation_diagram(
        U0, 0, PARAM_RANGE, TOTAL_TIME, transient_time=TRANSIENT_TIME, **kwargs
    )


def test_explicit_parameters_with_stored_parameters():
    # The explicit b overrides the stored one.
    _, explicit = sweep(henon([1.4, 0.1]), parameters=[0.3])
    _, stored = sweep(henon([1.4, 0.3]))
    assert explicit.shape == (3, TOTAL_TIME - TRANSIENT_TIME)
    np.testing.assert_array_equal(explicit, stored)


def test_explicit_parameters_without_stored_parameters():
    _, explicit = sweep(henon(), parameters=[0.3])
    _, stored = sweep(henon([1.4, 0.3]))
    np.testing.assert_array_equal(explicit, stored)


def test_explicit_parameters_go_around_the_swept_index():
    ds = henon()
    _, sweep_b = ds.bifurcation_diagram(
        U0, 1, (0.2, 0.3, 3), TOTAL_TIME, parameters=[1.4],
        transient_time=TRANSIENT_TIME,
    )
    _, stored = henon([1.4, 0.3]).bifurcation_diagram(
        U0, 1, (0.2, 0.3, 3), TOTAL_TIME, transient_time=TRANSIENT_TIME
    )
    np.testing.assert_array_equal(sweep_b, stored)
    np.testing.assert_array_equal(
        sweep_b[-1], henon([1.4, 0.3]).trajectory(
            U0, TOTAL_TIME, transient_time=TRANSIENT_TIME
        )[:, 0]
    )


def test_stored_parameters_are_not_modified():
    ds = henon([1.4, 0.3])
    sweep(ds)
    np.testing.assert_array_equal(ds.get_parameters(), [1.4, 0.3])


def test_missing_parameters_raise_for_multiparameter_systems():
    with pytest.raises(ValueError, match="parameters other than"):
        sweep(henon())


def test_explicit_parameters_of_the_wrong_length_raise():
    with pytest.raises(ValueError):
        sweep(henon([1.4, 0.3]), parameters=[0.3, 0.1])


def test_single_parameter_system_needs_no_parameters():
    ds = DDS(model="logistic map")
    values, results = ds.bifurcation_diagram(
        [0.2], 0, (3.0, 3.9, 4), 200, transient_time=100
    )
    expected = DDS(model="logistic map")
    expected.set_parameters(3.9)
    np.testing.assert_array_equal(
        results[-1], expected.trajectory([0.2], 200, transient_time=100)
    )
