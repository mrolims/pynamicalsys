"""
HamiltonianSystem.CLV and CLV_angles with poincare_section=True.

The section is defined by `section_index`, `section_value`, and `crossing`,
so every returned point must lie on the requested section and cross it in the
requested direction.
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
