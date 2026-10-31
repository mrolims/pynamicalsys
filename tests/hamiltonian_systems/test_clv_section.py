"""
HamiltonianSystem.CLV and CLV_angles with poincare_section=True.

The section is defined by `section_index`, `section_value`, and `crossing`,
so every returned point must lie on the requested section and cross it in the
requested direction.

The crossings are interpolated between the stored samples, which are the same
with and without the section, with cubic Hermite polynomials that use the
vector field at the samples. Each crossing time must lie between the times of
the two samples that bracket it, including after a warmup and when
`qr_time_step` is not a multiple of the integration step, and the energy of
the section points must stay close to that of the trajectory even when the
samples are far apart.
"""

import numpy as np
import pytest

from pynamicalsys import HamiltonianSystem as HS
from pynamicalsys.common.poincare import hermite_crossing

Q0 = [0.0, 0.1]
P0 = [0.3, 0.0]
SECTION = dict(section_index=1, section_value=0.05, crossing=-1)


@pytest.fixture(scope="module")
def system():
    hs = HS(model="henon heiles")
    hs.integrator("svy4", time_step=0.01)
    return hs


def check_on_section(section):
    # Columns are [time, x, y, px, py]; y is coordinate 1 and py its momentum,
    # so a downward crossing of y has py < 0.
    np.testing.assert_allclose(section[:, 2], SECTION["section_value"], atol=1e-12)
    assert np.all(section[:, 4] < 0.0)


def test_clv_uses_the_requested_section(system):
    _, section = system.CLV(Q0, P0, 50.0, poincare_section=True, **SECTION)
    assert section.shape[0] > 0
    check_on_section(section)


def test_clv_angles_use_the_requested_section(system):
    _, section = system.CLV_angles(
        Q0, P0, 50.0, pairs=(0, 3), poincare_section=True, **SECTION
    )
    assert section.shape[0] > 0
    check_on_section(section)


@pytest.mark.parametrize(
    "options",
    [
        dict(warmup_time=10.0),
        dict(qr_time_step=0.025),
    ],
)
def test_clv_section_times_interpolate_the_stored_times(system, options):
    section_options = dict(section_index=0, section_value=0.0, crossing=1)
    _, stored = system.CLV(Q0, P0, 50.0, **options)
    _, section = system.CLV(
        Q0, P0, 50.0, poincare_section=True, **section_options, **options
    )

    time = stored[:, 0]
    x = stored[:, 1]
    px = stored[:, 3]
    # Upward crossings of x = 0, counting a sample exactly on the section once.
    i = np.nonzero((x[:-1] < 0.0) & (x[1:] >= 0.0))[0]
    if x[0] == 0.0 and px[0] > 0.0:
        i = np.concatenate(([0], i))
    # For Hénon-Heiles, dx/dt = px.
    expected = np.array(
        [
            time[k]
            + hermite_crossing(x[k], x[k + 1], px[k], px[k + 1], time[k + 1] - time[k])
            * (time[k + 1] - time[k])
            for k in i
        ]
    )

    assert section.shape[0] == i.size > 0
    np.testing.assert_allclose(section[:, 0], expected, rtol=1e-14)
    assert np.all((section[:, 0] >= time[i]) & (section[:, 0] <= time[i + 1]))


@pytest.mark.parametrize("integrator", ["svy4", "imp"])
def test_clv_section_points_keep_the_energy(integrator):
    energy = 1.0 / 8.0
    y, py = -0.2, 0.0
    px = np.sqrt(2.0 * (energy - (y**2 / 2.0 - y**3 / 3.0)) - py**2)

    hs = HS(model="henon heiles")
    hs.integrator(integrator, time_step=0.01)
    _, section = hs.CLV(
        [0.0, y],
        [px, py],
        500.0,
        qr_time_step=0.1,
        poincare_section=True,
        section_index=0,
        section_value=0.0,
        crossing=1,
    )

    x, y, px, py = section[:, 1], section[:, 2], section[:, 3], section[:, 4]
    hamiltonian = (px**2 + py**2) / 2.0 + (x**2 + y**2) / 2.0 + x**2 * y - y**3 / 3.0
    np.testing.assert_allclose(hamiltonian, energy, rtol=0, atol=2e-6)
