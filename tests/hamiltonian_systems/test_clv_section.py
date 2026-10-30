"""
HamiltonianSystem.CLV and CLV_angles with poincare_section=True.

The section is defined by `section_index`, `section_value`, and `crossing`,
so every returned point must lie on the requested section and cross it in the
requested direction.

The crossings are interpolated between the stored samples, which are the same
with and without the section, so each crossing time must lie between the
times of the two samples that bracket it, including after a warmup and when
`qr_time_step` is not a multiple of the integration step.
"""

import numpy as np
import pytest

from pynamicalsys import HamiltonianSystem as HS

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
    lam = -x[i] / (x[i + 1] - x[i])
    expected = time[i] + lam * (time[i + 1] - time[i])

    assert section.shape[0] == i.size > 0
    np.testing.assert_allclose(section[:, 0], expected, rtol=1e-14)
