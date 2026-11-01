"""
ContinuousDynamicalSystem.bifurcation_diagram.

At each parameter value, the sweep computes the selected reduced map as
poincare_section, stroboscopic_map, or maxima_map would, so every row must be
a coordinate of those methods called directly at that value, to the last bit.
With continuation, each value starts from the state at the last map point of
the previous one.
"""

import numpy as np
import pytest

from pynamicalsys import ContinuousDynamicalSystem as CDS

ROSSLER = [0.2, 0.2, 4.0]
DUFFING = [0.2, 1.0, 1.0, 0.425, 1.1]
VALUES = np.array([3.0, 4.0, 5.5])


def rossler(integrator):
    ds = CDS(model="rossler system")
    ds.set_parameters(ROSSLER)
    if integrator == "rk4":
        ds.integrator("rk4", time_step=0.01)
    else:
        ds.integrator("rk45", atol=1e-8, rtol=1e-8)
    return ds


def duffing():
    ds = CDS(model="duffing")
    ds.set_parameters(DUFFING)
    ds.integrator("rk4", time_step=0.01)
    return ds


U_ROSSLER = [1.0, 1.0, 0.0]
U_DUFFING = [1.0, 0.0]
PERIOD = 2.0 * np.pi / 1.1


def with_parameter(base, index, value):
    parameters = list(base)
    parameters[index] = value
    return parameters


CASES = {
    "PS": dict(section_index=0, section_value=0.0, crossing=1),
    "MM": dict(maxima_index=2),
}


@pytest.mark.parametrize("integrator", ["rk4", "rk45"])
@pytest.mark.parametrize("map_type", ["PS", "MM"])
def test_rows_are_the_reduced_maps_at_each_value(integrator, map_type):
    ds = rossler(integrator)
    options = CASES[map_type]
    param_values, results = ds.bifurcation_diagram(
        U_ROSSLER, 2, VALUES, 30, map_type, transient_time=50.0, **options
    )
    np.testing.assert_array_equal(param_values, VALUES)
    assert results.shape == (3, 30)

    for row, c in zip(results, VALUES):
        parameters = with_parameter(ROSSLER, 2, c)
        if map_type == "PS":
            points = ds.poincare_section(
                U_ROSSLER, 30, parameters=parameters, transient_time=50.0, **options
            )
            observable = 1  # the first coordinate other than section_index = 0
        else:
            points = ds.maxima_map(
                U_ROSSLER, 30, parameters=parameters, transient_time=50.0, **options
            )
            observable = 2  # maxima_index
        np.testing.assert_array_equal(row, points[:, 1 + observable])


def test_stroboscopic_rows_are_the_stroboscopic_maps():
    ds = duffing()
    gammas = np.array([0.3, 0.425])
    _, results = ds.bifurcation_diagram(
        U_DUFFING, 3, gammas, 20, "sm", sampling_time=PERIOD, transient_time=10 * PERIOD
    )
    for row, gamma in zip(results, gammas):
        points = ds.stroboscopic_map(
            U_DUFFING,
            20,
            sampling_time=PERIOD,
            parameters=with_parameter(DUFFING, 3, gamma),
            transient_time=10 * PERIOD,
        )
        np.testing.assert_array_equal(row, points[:, 1])


def test_continuation_starts_each_value_from_the_last_map_point():
    ds = rossler("rk4")
    _, results, last_state = ds.bifurcation_diagram(
        U_ROSSLER,
        2,
        VALUES,
        30,
        "MM",
        maxima_index=2,
        transient_time=50.0,
        continuation=True,
        return_last_state=True,
    )
    state = U_ROSSLER
    for row, c in zip(results, VALUES):
        points = ds.maxima_map(
            state, 30, 2, parameters=with_parameter(ROSSLER, 2, c), transient_time=50.0
        )
        np.testing.assert_array_equal(row, points[:, 3])
        state = points[-1, 1:]
    np.testing.assert_array_equal(last_state, state)


def test_explicit_observable_and_parameters():
    ds = rossler("rk4")
    _, stored = ds.bifurcation_diagram(
        U_ROSSLER, 2, VALUES, 10, "MM", maxima_index=2, observable_index=0
    )
    explicit = CDS(model="rossler system")
    explicit.integrator("rk4", time_step=0.01)
    _, given = explicit.bifurcation_diagram(
        U_ROSSLER,
        2,
        VALUES,
        10,
        "MM",
        parameters=[0.2, 0.2],
        maxima_index=2,
        observable_index=0,
    )
    np.testing.assert_array_equal(stored, given)

    points = ds.maxima_map(U_ROSSLER, 10, 2, parameters=with_parameter(ROSSLER, 2, 3.0))
    np.testing.assert_array_equal(stored[0], points[:, 1])


def test_poincare_default_observable_skips_the_section_coordinate():
    ds = rossler("rk4")
    _, results = ds.bifurcation_diagram(
        U_ROSSLER, 2, [4.0], 10, "PS", section_index=1, section_value=0.0, crossing=1
    )
    points = ds.poincare_section(U_ROSSLER, 10, 1, 0.0, crossing=1)
    np.testing.assert_array_equal(results[0], points[:, 1])  # coordinate 0


@pytest.mark.parametrize(
    "kwargs, error",
    [
        (dict(map_type="XX"), ValueError),
        (dict(map_type=2), TypeError),
        (dict(map_type="PS", section_index=0), ValueError),
        (dict(map_type="PS", section_index=3, section_value=0.0, crossing=1), ValueError),
        (dict(map_type="PS", section_index=0, section_value=0.0, crossing=2), ValueError),
        (dict(map_type="SM"), ValueError),
        (dict(map_type="SM", sampling_time=0.0), ValueError),
        (dict(map_type="MM"), ValueError),
        (dict(map_type="MM", maxima_index=3), ValueError),
        (dict(map_type="MM", maxima_index=2, observable_index=3), ValueError),
        (dict(map_type="MM", maxima_index=2, continuation=1), TypeError),
        (dict(map_type="MM", maxima_index=2, return_last_state="yes"), TypeError),
        (dict(map_type="MM", maxima_index=2, parameters=[0.2]), ValueError),
    ],
)
def test_invalid_arguments(kwargs, error):
    ds = rossler("rk4")
    with pytest.raises(error):
        ds.bifurcation_diagram(U_ROSSLER, 2, VALUES, 5, **kwargs)


def test_invalid_sweep_arguments():
    ds = rossler("rk4")
    with pytest.raises(ValueError, match="param_index"):
        ds.bifurcation_diagram(U_ROSSLER, 3, VALUES, 5, "MM", maxima_index=2)
    with pytest.raises(ValueError, match="num_intersections"):
        ds.bifurcation_diagram(U_ROSSLER, 2, VALUES, 0, "MM", maxima_index=2)
    with pytest.raises(TypeError):
        ds.bifurcation_diagram(U_ROSSLER, 2, VALUES, 5)  # map_type is required
