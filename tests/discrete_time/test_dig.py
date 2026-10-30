"""
The dig indicator and the forms of observable it accepts.

An observable is a function of a single state. The former form, a function of
an array of states, is still accepted with a deprecation warning, and both
forms must give the same result.
"""

import numpy as np
import pytest

from pynamicalsys import DiscreteDynamicalSystem as DDS

TOTAL_TIME = 5000


@pytest.fixture
def system():
    system = DDS(model="standard map")
    system.set_parameters([0.7])
    return system


def test_regular_orbit_converges_faster_than_chaotic(system):
    regular = system.dig([0.0, 0.25], TOTAL_TIME)
    chaotic = system.dig([0.01, 0.01], TOTAL_TIME)
    assert regular > 10.0
    assert chaotic < 5.5


def test_array_observable_is_deprecated_and_equivalent(system):
    u = [0.0, 0.25]
    reference = system.dig(u, TOTAL_TIME, func=lambda u: np.cos(2 * np.pi * u[0]))

    with pytest.warns(DeprecationWarning):
        legacy = system.dig(
            u, TOTAL_TIME, func=lambda X: np.cos(2 * np.pi * X[:, 0])
        )

    assert legacy == pytest.approx(reference, abs=1e-6)


def test_observable_numba_cannot_compile(system):
    """Such an observable falls back to plain Python evaluation."""
    u = [0.0, 0.25]

    def observable(state):
        return float(np.cos(2 * np.pi * state[0])) + 0.0 * len(repr(state))

    reference = system.dig(u, TOTAL_TIME)
    assert system.dig(u, TOTAL_TIME, func=observable) == pytest.approx(
        reference, abs=1e-6
    )


def test_invalid_observable(system):
    with pytest.raises(TypeError):
        system.dig([0.0, 0.25], TOTAL_TIME, func=1.0)

    with pytest.raises(ValueError):
        system.dig([0.0, 0.25], TOTAL_TIME, func=lambda u: u)


def test_total_time_too_small(system):
    with pytest.raises(ValueError):
        system.dig([0.0, 0.25], 1)


def test_default_observable_period():
    """With period 2 pi, the default observable is cos(u[0])."""
    system = DDS(model="leonel map")
    system.set_parameters([1e-3, 1.0])
    u = [1.0, 1.5]

    explicit = system.dig(u, TOTAL_TIME, func=lambda u: np.cos(u[0]))
    assert system.dig(u, TOTAL_TIME, period=2 * np.pi) == explicit

    with pytest.raises(ValueError):
        system.dig(u, TOTAL_TIME, period=0.0)


@pytest.mark.parametrize("method", ["dig", "classify_orbit"])
def test_period_with_func_warns_and_is_ignored(method):
    system = DDS(model="unbounded standard map")
    system.set_parameters([0.7])
    u = [0.2, 0.25]
    observable = lambda u: np.cos(2 * np.pi * u[0])
    compute = getattr(system, method)

    expected = compute(u, TOTAL_TIME, func=observable)
    with pytest.warns(UserWarning, match="period"):
        result = compute(u, TOTAL_TIME, func=observable, period=2 * np.pi)
    assert result == expected


@pytest.mark.parametrize("method", ["dig", "classify_orbit"])
def test_period_is_validated_even_with_func(method):
    system = DDS(model="unbounded standard map")
    system.set_parameters([0.7])
    with pytest.raises(ValueError, match="period"):
        getattr(system, method)(
            [0.2, 0.25], TOTAL_TIME, func=lambda u: np.cos(u[0]), period=-1.0
        )
