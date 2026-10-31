"""
Every method that draws random numbers uses the same default seed, 13121989,
in DiscreteDynamicalSystem, ContinuousDynamicalSystem, HamiltonianSystem, and
BasinMetrics. Calling a method without `seed` must therefore give exactly the
same result as passing `seed=13121989`.
"""

import numpy as np
import pytest

from pynamicalsys import BasinMetrics
from pynamicalsys import ContinuousDynamicalSystem as CDS
from pynamicalsys import DiscreteDynamicalSystem as DDS
from pynamicalsys import HamiltonianSystem as HS

SEED = 13121989


def henon_map():
    ds = DDS(model="henon map")
    ds.set_parameters([1.4, 0.3])
    return ds


def standard_map():
    ds = DDS(model="standard map")
    ds.set_parameters([1.5])
    return ds


def lorenz():
    ds = CDS(model="lorenz system")
    ds.set_parameters([10.0, 28.0, 8.0 / 3.0])
    ds.integrator("rk4", time_step=0.01)
    return ds


def henon_heiles():
    hs = HS(model="henon heiles")
    hs.integrator("svy4", time_step=0.01)
    return hs


U_D = [0.05, 0.05]
U_C = [1.0, 1.0, 1.0]
Q, P = [0.0, 0.1], [0.3, 0.0]

CASES = {
    # Short runs without a tail, so that the vectors still depend on the seed.
    "discrete CLV": (henon_map, lambda s, **k: s.CLV([0.1, 0.1], 20, **k)),
    "discrete CLV_angles": (
        henon_map,
        lambda s, **k: s.CLV_angles([0.1, 0.1], 20, pairs=(0, 1), use_abs=False, **k),
    ),
    "discrete SALI": (standard_map, lambda s, **k: s.SALI(U_D, 20, return_history=True, **k)),
    "discrete LDI": (standard_map, lambda s, **k: s.LDI(U_D, 20, 2, return_history=True, **k)),
    "discrete GALI": (standard_map, lambda s, **k: s.GALI(U_D, 20, 2, return_history=True, **k)),
    "continuous lyapunov": (lorenz, lambda s, **k: s.lyapunov(U_C, 2.0, **k)),
    "continuous CLV": (lorenz, lambda s, **k: s.CLV(U_C, 2.0, tail_time=1.0, **k)),
    "continuous CLV_angles": (
        lorenz,
        lambda s, **k: s.CLV_angles(U_C, 2.0, pairs=(0, 2), tail_time=1.0, **k),
    ),
    "continuous SALI": (lorenz, lambda s, **k: s.SALI(U_C, 2.0, return_history=True, **k)),
    "continuous LDI": (lorenz, lambda s, **k: s.LDI(U_C, 2.0, 2, return_history=True, **k)),
    "continuous GALI": (lorenz, lambda s, **k: s.GALI(U_C, 2.0, 2, return_history=True, **k)),
    "Hamiltonian lyapunov": (henon_heiles, lambda s, **k: s.lyapunov(Q, P, 2.0, **k)),
    "Hamiltonian CLV": (henon_heiles, lambda s, **k: s.CLV(Q, P, 2.0, tail_time=1.0, **k)),
    "Hamiltonian CLV_angles": (
        henon_heiles,
        lambda s, **k: s.CLV_angles(Q, P, 2.0, pairs=(0, 3), tail_time=1.0, **k),
    ),
    "Hamiltonian SALI": (henon_heiles, lambda s, **k: s.SALI(Q, P, 2.0, return_history=True, **k)),
    "Hamiltonian LDI": (henon_heiles, lambda s, **k: s.LDI(Q, P, 2.0, 2, return_history=True, **k)),
    "Hamiltonian GALI": (henon_heiles, lambda s, **k: s.GALI(Q, P, 2.0, 2, return_history=True, **k)),
}


def as_arrays(result):
    return [np.asarray(r) for r in (result if isinstance(result, tuple) else (result,))]


@pytest.mark.parametrize("name", CASES)
def test_default_seed_is_13121989(name):
    make, compute = CASES[name]
    system = make()
    default = as_arrays(compute(system))
    explicit = as_arrays(compute(system, seed=SEED))
    for a, b in zip(default, explicit):
        np.testing.assert_array_equal(a, b)


class SideOfLine:
    """A stand-in for a map: the side of x = 0 on which a point lies."""

    def escape_analysis(self, u, max_time, parameters, exits, escape):
        return int(u[0] > 0.0), 0


def test_uncertainty_fraction_mapping_default_seed_is_13121989():
    x = np.linspace(-1.0, 1.0, 40)
    X, Y = np.meshgrid(x, x)
    metrics = BasinMetrics((X > 0.0).astype(np.float64))
    options = dict(
        mapping=SideOfLine(),
        parameters=[0.0],
        exits=[],
        n_samples=200,
        p_samples=4,
        n_eps=3,
        n_jobs=1,
    )
    _, default = metrics.uncertainty_fraction_mapping(X, Y, **options)
    _, explicit = metrics.uncertainty_fraction_mapping(X, Y, seed=SEED, **options)
    np.testing.assert_array_equal(default, explicit)
