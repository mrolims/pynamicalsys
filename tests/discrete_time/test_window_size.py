"""
`dig` and `rotation_number` take the length of each averaging window as
`window_size`. The former name, `total_time`, still works with a
DeprecationWarning, since elsewhere `total_time` counts every iteration from
the initial condition, including the transient.
"""

import warnings

import numpy as np
import pytest

from pynamicalsys import DiscreteDynamicalSystem as DDS

# A regular orbit, so the rotation number is finite and can be compared.
U = [0.2, 0.25]
WINDOW = 2_000


@pytest.fixture
def system():
    ds = DDS(model="unbounded standard map")
    ds.set_parameters([0.7])
    return ds


METHODS = ["dig", "rotation_number"]


@pytest.mark.parametrize("method", METHODS)
def test_window_size_is_positional_and_keyword(system, method):
    compute = getattr(system, method)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        positional = compute(U, WINDOW)
        keyword = compute(U, window_size=WINDOW)
    assert np.isfinite(positional)
    assert positional == keyword


@pytest.mark.parametrize("method", METHODS)
def test_total_time_is_a_deprecated_alias(system, method):
    compute = getattr(system, method)
    expected = compute(U, window_size=WINDOW)
    with pytest.warns(DeprecationWarning, match="Use window_size"):
        result = compute(U, total_time=WINDOW)
    assert result == expected


@pytest.mark.parametrize("method", METHODS)
def test_both_names_raise(system, method):
    with pytest.raises(TypeError, match="pass only window_size"):
        getattr(system, method)(U, window_size=WINDOW, total_time=WINDOW)


@pytest.mark.parametrize("method", METHODS)
def test_window_size_is_required(system, method):
    with pytest.raises(TypeError, match="window_size"):
        getattr(system, method)(U)


def test_classify_orbit_takes_window_size(system):
    result = system.classify_orbit(U, window_size=WINDOW)
    assert result["classification"] in ("chaotic", "island", "circle", "undetermined")
