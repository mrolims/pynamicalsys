"""
ContinuousDynamicalSystem.poincare_section: crossing times with rk45, accuracy
of the interpolation, and points exactly on the section.

The section is checked against the trajectory computed with the same
integrator, which takes the same accepted steps. Each crossing must be the
cubic Hermite interpolation, in time and state, between the two trajectory
points that bracket it. With rk45, some steps are rejected and retried with a
shorter step, so the crossing time must use the accepted step and not the
requested one.

The Hermite interpolation uses the vector field at both ends of the step, so
its error is far below that of a linear interpolation: with rk4 and a step of
0.01 on the Lorenz system, about 1e-4 instead of 5e-2.

A point exactly on the section is counted once, when the trajectory arrives
on it, and the initial point is counted when it lies on the section.
"""

import numpy as np
import pytest
from numba import njit

from pynamicalsys import ContinuousDynamicalSystem as CDS
from pynamicalsys.common.poincare import hermite_crossing, hermite_point
from pynamicalsys.continuous_time.models import lorenz_system

LORENZ_PARAMETERS = np.array([10.0, 28.0, 8.0 / 3.0])

SECTION_INDEX = 2
SECTION_VALUE = 27.0
NUM_INTERSECTIONS = 200


def test_rk45_crossings_interpolate_the_accepted_steps():
    ds = CDS(model="lorenz system")
    ds.set_parameters([10.0, 28.0, 8.0 / 3.0])
    ds.integrator("rk45", atol=1e-6, rtol=1e-6)
    u = [0.1, 0.1, 0.1]

    section = ds.poincare_section(
        u,
        NUM_INTERSECTIONS,
        section_index=SECTION_INDEX,
        section_value=SECTION_VALUE,
        crossing=0,
    )
    trajectory = ds.trajectory(u, total_time=section[-1, 0] + 1.0)

    time = trajectory[:, 0]
    z = trajectory[:, SECTION_INDEX + 1]
    sign_change = (z[:-1] - SECTION_VALUE) * (z[1:] - SECTION_VALUE) < 0.0
    i = np.nonzero(sign_change)[0][:NUM_INTERSECTIONS]

    expected = np.empty_like(section)
    for row, k in enumerate(i):
        h = time[k + 1] - time[k]
        u_old, u_new = trajectory[k, 1:], trajectory[k + 1, 1:]
        du_old = lorenz_system(time[k], u_old, LORENZ_PARAMETERS)
        du_new = lorenz_system(time[k + 1], u_new, LORENZ_PARAMETERS)
        s = hermite_crossing(
            u_old[SECTION_INDEX] - SECTION_VALUE,
            u_new[SECTION_INDEX] - SECTION_VALUE,
            du_old[SECTION_INDEX],
            du_new[SECTION_INDEX],
            h,
        )
        expected[row, 0] = time[k] + s * h
        expected[row, 1:] = hermite_point(u_old, u_new, du_old, du_new, h, s)

    np.testing.assert_allclose(section, expected, rtol=1e-14)
    assert np.all((section[:, 0] >= time[i]) & (section[:, 0] <= time[i + 1]))


def test_crossings_have_the_accuracy_of_the_integrator():
    ds = CDS(model="lorenz system")
    ds.set_parameters(LORENZ_PARAMETERS)
    u = [1.0, 1.0, 1.0]
    options = dict(
        section_index=SECTION_INDEX, section_value=SECTION_VALUE, crossing=0
    )

    ds.integrator("rk45", atol=1e-13, rtol=1e-13)
    reference = ds.poincare_section(u, 10, **options)
    ds.integrator("rk4", time_step=0.01)
    section = ds.poincare_section(u, 10, **options)

    np.testing.assert_allclose(section, reference, rtol=0, atol=1e-3)


@njit
def cubic_eom(time, u, parameters):
    # u[0] = t - t**3 / 3 for u[0](0) = 0. The right-hand side is quadratic
    # in time, so rk4 integrates it exactly up to round-off.
    return np.array([1.0 - time**2])


@pytest.fixture
def cubic():
    ds = CDS(equations_of_motion=cubic_eom, system_dimension=1, number_of_parameters=0)
    ds.integrator("rk4", time_step=0.125)
    return ds


def test_step_landing_on_the_section_is_counted(cubic):
    # The section value is the state after four steps, so the fourth step
    # lands exactly on it at t = 0.5. The trajectory crosses it again,
    # downward, between t = 1 and t = 1.5, and never again: with crossing=1,
    # a missed landing would make poincare_section run forever.
    section_value = cubic.trajectory([0.0], total_time=0.5)[-1, 1]
    section = cubic.poincare_section(
        [0.0], 1, section_index=0, section_value=section_value, crossing=0
    )
    np.testing.assert_array_equal(section, [[0.5, section_value]])


def test_initial_point_on_the_section_is_counted(cubic):
    # The trajectory starts on the section and crosses it again, downward, at
    # sqrt(3), and never again.
    section = cubic.poincare_section(
        [0.0], 1, section_index=0, section_value=0.0, crossing=0
    )
    np.testing.assert_array_equal(section, [[0.0, 0.0]])
