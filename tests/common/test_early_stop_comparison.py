"""
SALI, LDI, and GALI stop early when the indicator falls strictly below
`threshold`, in DiscreteDynamicalSystem, ContinuousDynamicalSystem, and
HamiltonianSystem alike.

The threshold is set to a value that the indicator takes at one of its stored
samples, so a comparison with `<=` would stop at that sample, while `<` goes
on to the first sample whose value is strictly smaller.
"""

import numpy as np
import pytest

from pynamicalsys import ContinuousDynamicalSystem as CDS
from pynamicalsys import DiscreteDynamicalSystem as DDS
from pynamicalsys import HamiltonianSystem as HS


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


U_DISCRETE = [0.05, 0.05]
U_CONTINUOUS = [1.0, 1.0, 1.0]
# A chaotic orbit of the Hénon-Heiles system at energy 1/8.
Q_HAMILTONIAN = [0.0, -0.2]
P_HAMILTONIAN = [np.sqrt(2.0 * (1.0 / 8.0 - (0.02 + 0.008 / 3.0))), 0.0]

CASES = {
    "discrete SALI": (standard_map, lambda ds, **kw: ds.SALI(U_DISCRETE, 2000, **kw)),
    "discrete LDI": (standard_map, lambda ds, **kw: ds.LDI(U_DISCRETE, 2000, 2, **kw)),
    "discrete GALI": (standard_map, lambda ds, **kw: ds.GALI(U_DISCRETE, 2000, 2, **kw)),
    "continuous SALI": (lorenz, lambda ds, **kw: ds.SALI(U_CONTINUOUS, 10.0, **kw)),
    "continuous LDI": (lorenz, lambda ds, **kw: ds.LDI(U_CONTINUOUS, 10.0, 2, **kw)),
    "continuous GALI": (lorenz, lambda ds, **kw: ds.GALI(U_CONTINUOUS, 10.0, 2, **kw)),
    "Hamiltonian SALI": (
        henon_heiles,
        lambda hs, **kw: hs.SALI(Q_HAMILTONIAN, P_HAMILTONIAN, 200.0, **kw),
    ),
    "Hamiltonian LDI": (
        henon_heiles,
        lambda hs, **kw: hs.LDI(Q_HAMILTONIAN, P_HAMILTONIAN, 200.0, 2, **kw),
    ),
    "Hamiltonian GALI": (
        henon_heiles,
        lambda hs, **kw: hs.GALI(Q_HAMILTONIAN, P_HAMILTONIAN, 200.0, 2, **kw),
    ),
}


def values_of(history):
    # The discrete classes return the values alone, the others [time, value].
    return history if history.ndim == 1 else history[:, 1]


@pytest.mark.parametrize("name", CASES)
def test_stops_only_strictly_below_threshold(name):
    make, compute = CASES[name]
    system = make()
    values = values_of(compute(system, threshold=0.0, return_history=True))

    # A sample that is a new running minimum and is later undercut.
    records = [
        k
        for k in range(1, len(values) - 1)
        if values[k] < values[:k].min() and values[k + 1 :].min() < values[k]
    ]
    assert records
    k = records[len(records) // 2]
    first_below = k + 1 + np.argmax(values[k + 1 :] < values[k])

    stopped = values_of(compute(system, threshold=values[k], return_history=True))

    assert len(stopped) == first_below + 1
    np.testing.assert_array_equal(stopped, values[: first_below + 1])
