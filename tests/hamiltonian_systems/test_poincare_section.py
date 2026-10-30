"""
HamiltonianSystem.poincare_section: points exactly on the section.

The first degree of freedom is a free particle with unit momentum, so with the
step 0.125 its coordinate is a multiple of 0.125 without round-off and lands
exactly on the section q = 0.5 (taken as periodic with period 1) once every
eight steps. The second degree of freedom is a harmonic oscillator. A point on
the section is counted once, when the trajectory arrives on it, and the initial
point is counted when it lies on the section.
"""

import numpy as np
import pytest
from numba import njit

from pynamicalsys import HamiltonianSystem as HS


@njit
def grad_T(p, parameters):
    return p.copy()


@njit
def grad_V(q, parameters):
    out = np.zeros_like(q)
    out[1] = q[1]
    return out


@njit
def hess_T(p, parameters):
    return np.eye(2)


@njit
def hess_V(q, parameters):
    out = np.zeros((2, 2))
    out[1, 1] = 1.0
    return out


@pytest.fixture
def system():
    hs = HS(
        grad_T=grad_T,
        grad_V=grad_V,
        hess_T=hess_T,
        hess_V=hess_V,
        degrees_of_freedom=2,
        number_of_parameters=0,
    )
    hs.integrator("vv2", time_step=0.125)
    return hs


def section_times(system, q0, crossing):
    section = system.poincare_section(
        [q0, 1.0],
        [1.0, 0.0],
        3,
        section_index=0,
        section_value=0.5,
        crossing=crossing,
        periodic_section_coordinate=True,
        period=1.0,
    )
    return section[:, 0]


@pytest.mark.parametrize("crossing", [0, 1])
def test_landing_on_the_section_is_counted_once(system, crossing):
    np.testing.assert_array_equal(
        section_times(system, 0.0, crossing), [0.5, 1.5, 2.5]
    )


@pytest.mark.parametrize("crossing", [0, 1])
def test_initial_point_on_the_section_is_counted(system, crossing):
    np.testing.assert_array_equal(
        section_times(system, 0.5, crossing), [0.0, 1.0, 2.0]
    )
