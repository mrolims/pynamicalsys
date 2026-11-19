"""
Unit test for constructing pseudo-orbits using Bollt's method.

The method tested in this file may also detect a cycle as auxiliary
information. This detection is not part of the definition of a
pseudo-orbit. The reference is:

    E. M. Bollt, "The Path Towards a Longer Life: On Invariant Sets and
    the Escape Time Landscape", International Journal of Bifurcation and
    Chaos, 15(5), 1615--1624 (2005).
"""

import numpy as np
import pytest

from pynamicalsys import DiscreteDynamicalSystem as DDS


def test_build_pseudo_orbit_uses_stored_parameters():
    system = DDS(model="logistic map")
    system.set_parameters([3.9])

    orbit, period = system.build_pseudo_orbit(
        0.3,
        exits=[[0.0, 1.0]],
        total_time=8,
        min_lifetime=2,
        max_escape_time=4,
        n_dim=1,
        seed=1,
    )

    assert orbit.shape == (8, 1)
    assert period == -1 or period > 0
    assert np.isfinite(orbit).all()


@pytest.mark.parametrize("refine", ["gradient", "sampling"])
def test_build_pseudo_orbit_henon_map(refine):
    ds = DDS(model="henon map")
    ds.set_parameters([1.50, 0.3])

    orbit, period = ds.build_pseudo_orbit(
        u=[0.1, 0.2],
        exits=[[-2, 2], [-2, 2]],
        total_time=1000,
        refine=refine,
        seed=1,
    )

    assert orbit.shape == (1000, 2)
    assert period == -1 or period > 0
    assert np.isfinite(orbit).all()


@pytest.mark.parametrize("refine", ["invalid", 1])
def test_build_pseudo_orbit_rejects_invalid_refinement(refine):
    system = DDS(model="logistic map")
    system.set_parameters([3.9])

    with pytest.raises((TypeError, ValueError), match="refine"):
        system.build_pseudo_orbit(0.3, exits=[[0.0, 1.0]], refine=refine)
