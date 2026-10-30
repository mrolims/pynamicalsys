"""
ContinuousDynamicalSystem.poincare_section: crossing times with rk45.

The section is checked against the trajectory computed with the same
integrator, which takes the same accepted steps. Each crossing must be the
linear interpolation, in time and state, between the two trajectory points
that bracket it. With rk45, some steps are rejected and retried with a
shorter step, so the crossing time must use the accepted step and not the
requested one.
"""

import numpy as np

from pynamicalsys import ContinuousDynamicalSystem as CDS

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
    lam = (SECTION_VALUE - z[i]) / (z[i + 1] - z[i])

    expected_states = (1.0 - lam)[:, None] * trajectory[i, 1:] + lam[
        :, None
    ] * trajectory[i + 1, 1:]
    np.testing.assert_allclose(section[:, 1:], expected_states, rtol=1e-14)

    expected_times = time[i] + lam * (time[i + 1] - time[i])
    np.testing.assert_allclose(section[:, 0], expected_times, rtol=1e-14)
    assert np.all((section[:, 0] >= time[i]) & (section[:, 0] <= time[i + 1]))
