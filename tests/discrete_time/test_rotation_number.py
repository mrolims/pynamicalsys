"""
Rotation numbers and orbit classification with weighted Birkhoff averages.

The reference values come from two sources.

1. The unperturbed standard map (k = 0) is a rigid rotation, x -> x + y, so the
   rotation number of every orbit equals its action y exactly. Irrational
   actions give rotational circles, rational ones give periodic orbits, and
   both are known in closed form.

2. For k > 0, the standard map has an elliptic fixed point at (0.5, 0), whose
   island has rotation number zero, and a chaotic layer around the hyperbolic
   fixed point at the origin.

The smallest-denominator search is checked against the examples worked out by
Sander and Meiss (2020).
"""

import numpy as np
import pytest

from pynamicalsys import DiscreteDynamicalSystem as DDS
from pynamicalsys.discrete_time.rotation import smallest_denominator

GOLDEN = (np.sqrt(5.0) - 1.0) / 2.0
TOTAL_TIME = 5000


def unbounded_standard_map(k):
    system = DDS(model="unbounded standard map")
    system.set_parameters([k])
    return system


@pytest.mark.parametrize(
    "x, delta, expected",
    [
        (0.12, 0.005, 17),  # Sander and Meiss (2020), Sec. 3.1
        (np.pi, 1e-3, 64),  # 201/64, not a convergent of pi
        (0.25, 1e-8, 4),
        (-0.25, 1e-8, 4),
        (2.0, 1e-8, 1),
        (1.0 / 3.0 + 1e-12, 1e-8, 3),
    ],
)
def test_smallest_denominator(x, delta, expected):
    assert smallest_denominator(x, delta) == expected


@pytest.mark.parametrize("method", ["wba", "average"])
@pytest.mark.parametrize("y0", [GOLDEN, 0.25, 2.25])
def test_rigid_rotation(method, y0):
    """At k = 0 the rotation number is the action, without reduction."""
    system = unbounded_standard_map(0.0)
    omega = system.rotation_number([0.1, y0], TOTAL_TIME, method=method)
    assert omega == pytest.approx(y0, abs=1e-12)


def test_wba_and_average_agree_on_circle():
    system = unbounded_standard_map(0.7)
    u = [0.0, 0.25]
    wba = system.rotation_number(u, TOTAL_TIME)
    average = system.rotation_number(u, 2 * TOTAL_TIME, method="average")
    assert np.isfinite(wba)
    assert average == pytest.approx(wba, abs=1e-3)


def test_island_around_elliptic_point():
    """Orbits cross y = 0 here, so the increment changes sign."""
    system = unbounded_standard_map(0.7)
    omega = system.rotation_number([0.5, 0.05], TOTAL_TIME)
    assert omega == pytest.approx(0.0, abs=1e-9)


def test_chaotic_orbit_returns_nan():
    system = unbounded_standard_map(0.7)
    assert np.isnan(system.rotation_number([0.01, 0.01], TOTAL_TIME))


@pytest.mark.parametrize(
    "k, u, expected",
    [
        (0.0, [0.1, GOLDEN], "circle"),
        (0.0, [0.1, 1.0 / 3.0], "island"),
        (0.0, [0.1, 3.0], "island"),
        (0.7, [0.0, 0.25], "circle"),
        (0.7, [0.5, 0.05], "island"),
        (0.7, [0.15, 0.30], "island"),
        (0.7, [0.01, 0.01], "chaotic"),
    ],
)
def test_classify_orbit(k, u, expected):
    result = unbounded_standard_map(k).classify_orbit(u, TOTAL_TIME)
    assert result["classification"] == expected
    assert np.isnan(result["rotation_number"]) == (expected == "chaotic")


def test_classify_orbit_custom_observable():
    system = unbounded_standard_map(0.7)
    result = system.classify_orbit(
        [0.0, 0.25], TOTAL_TIME, func=lambda u: np.sin(2 * np.pi * u[0])
    )
    assert result["classification"] == "circle"


def test_classify_orbit_rejects_array_observable():
    system = unbounded_standard_map(0.7)
    with pytest.raises(ValueError):
        system.classify_orbit(
            [0.0, 0.25], TOTAL_TIME, func=lambda X: np.cos(2 * np.pi * X[:, 0])
        )


def test_custom_frequency_map_matches_builtin():
    """A plain Python frequency map is compiled and gives the same result."""

    def frequency(u, parameters):
        return u[1] + parameters[0] * np.sin(2 * np.pi * u[0]) / (2 * np.pi)

    builtin = unbounded_standard_map(0.7)
    from_setter = unbounded_standard_map(0.7)
    from_setter.set_frequency_map(frequency)
    from_constructor = DDS(
        model="unbounded standard map", frequency_map=frequency
    )
    from_constructor.set_parameters([0.7])

    u = [0.0, 0.25]
    reference = builtin.rotation_number(u, TOTAL_TIME)
    assert from_setter.rotation_number(u, TOTAL_TIME) == reference
    assert from_constructor.rotation_number(u, TOTAL_TIME) == reference


def test_without_frequency_map_is_deprecated():
    """The former wrapped-increment average still works, with a warning."""
    system = DDS(model="standard map")
    system.set_parameters([0.0])

    with pytest.warns(DeprecationWarning):
        omega = system.rotation_number([0.1, 0.25], 100)
    assert omega == pytest.approx(0.25, abs=1e-12)

    with pytest.raises(ValueError):
        system.rotation_number([0.1, 0.25], 100, method="wba")

    with pytest.raises(ValueError):
        system.classify_orbit([0.1, 0.25], 100)


def test_mod_with_frequency_map_is_deprecated():
    system = unbounded_standard_map(0.0)
    with pytest.warns(DeprecationWarning):
        omega = system.rotation_number([0.1, 0.25], 100, mod=1.0)
    assert omega == pytest.approx(0.25, abs=1e-12)


def test_invalid_method():
    with pytest.raises(ValueError):
        unbounded_standard_map(0.7).rotation_number([0.0, 0.25], 100, method="x")


def test_default_observable_period():
    """With period 2 pi, the default observable is cos(u[0])."""
    system = DDS(model="leonel map")
    system.set_parameters([1e-3, 1.0])
    u = [1.0, 1.5]

    explicit = system.classify_orbit(u, TOTAL_TIME, func=lambda u: np.cos(u[0]))
    default = system.classify_orbit(u, TOTAL_TIME, period=2 * np.pi)
    assert default["dig"] == explicit["dig"]
    assert default["classification"] == "circle"


def test_classify_orbit_undetermined_until_omega_converges():
    # A regular orbit whose rotation number has converged to only about 8
    # digits with windows of 10000 iterations, below omega_threshold = 9. The
    # rationality test is not applied to it, and longer windows classify it.
    system = unbounded_standard_map(0.9)
    u = [0.6896, 0.5004]

    short = system.classify_orbit(u, 10_000)
    assert short["classification"] == "undetermined"
    assert short["dig"] >= 5.5
    assert short["dig_omega"] < 9.0
    assert np.isfinite(short["rotation_number"])

    long = system.classify_orbit(u, 40_000)
    assert long["classification"] == "circle"
    assert long["dig_omega"] >= 9.0
