# variational.py

# Copyright (C) 2025 Matheus Rolim Sales
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.


import numpy as np
from numba import njit
from numpy.typing import NDArray

from pynamicalsys.common.types import flow_t, flow_jacobian_t


@njit
def variational_equations(
    time: np.float64,
    state: NDArray[np.float64],
    parameters: NDArray[np.float64],
    equations_of_motion: flow_t,
    jacobian: flow_jacobian_t,
    number_of_deviation_vectors: int | None = None,
) -> NDArray[np.float64]:
    """
    Compute the combined equations of motion and variational equations for a
    continuous-time dynamical system.

    This function evolves both the physical state and a set of deviation
    vectors. It is intended for tangent-space methods such as Lyapunov
    exponents and related chaos indicators.

    Parameters
    ----------
    time : np.float64
        Current integration time.
    state : NDArray[np.float64]
        Extended state vector containing:
        - the system state in the first `neq` entries
        - the flattened deviation matrix in the remaining entries
    parameters : NDArray[np.float64]
        System parameters passed to the equations of motion and Jacobian.
    equations_of_motion : flow_t
        Continuous-time vector field with signature
        `(time, u, parameters) -> dudt`.
    jacobian : flow_jacobian_t
        Jacobian of the vector field with signature
        `(time, u, parameters) -> J`.
    number_of_deviation_vectors : int | None, optional
        Number of deviation vectors stored in `state`.
        If None, this number is inferred assuming the extended state contains
        `neq` deviation vectors, so that its total size is `neq + neq**2`.

    Returns
    -------
    NDArray[np.float64]
        Time derivative of the extended state vector, with the same shape as
        `state`. The output contains:
        - the phase-space velocity `dudt` in the first `neq` entries
        - the flattened tangent-space evolution `dvdt = J @ v` in the remaining
          entries

    Notes
    -----
    Let `u(t)` be the system state and `v(t)` the deviation matrix. This
    function computes

    - `du/dt = F(t, u, parameters)`
    - `dv/dt = J(t, u, parameters) @ v`

    where `J` is the Jacobian of `F` with respect to `u`.

    The input `state` is interpreted as a concatenation of the state vector and
    the flattened deviation matrix. If `number_of_deviation_vectors` is not
    provided, the function assumes the standard square tangent evolution used
    when computing the full Lyapunov spectrum.
    """
    state = state.copy()
    nt = len(state)

    if number_of_deviation_vectors is not None:
        ndv = int(number_of_deviation_vectors)
        neq = nt // (1 + ndv)
    else:
        neq = int((-1 + np.sqrt(1 + 4 * nt)) / 2)
        ndv = neq

    u = state[:neq].copy()
    v = state[neq:].reshape(neq, ndv).copy()

    J = jacobian(time, u, parameters)
    dudt = equations_of_motion(time, u, parameters)
    dvdt = J @ v

    dstatedt = np.empty(nt, dtype=np.float64)
    dstatedt[:neq] = dudt
    dstatedt[neq:] = dvdt.reshape(neq * ndv)

    return dstatedt
