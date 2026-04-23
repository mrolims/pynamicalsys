# ldi.py

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


from numpy.typing import NDArray
import numpy as np
from numba import njit

from pynamicalsys.common.types import grad_t, hess_t, symplectic_tangent_step_t
from pynamicalsys.common.linalg import qr


@njit
def ldi_k(
    q: NDArray[np.float64],
    p: NDArray[np.float64],
    total_time: np.float64,
    time_step: np.float64,
    parameters: NDArray[np.float64],
    grad_T: grad_t,
    grad_V: grad_t,
    hess_T: hess_t,
    hess_V: hess_t,
    k: int,
    return_history: bool,
    seed: int,
    integrator_traj_tan: symplectic_tangent_step_t,
    threshold: np.float64,
) -> NDArray[np.float64]:
    """
    Compute the Linear Dependence Index of order `k` for a Hamiltonian system.

    Parameters
    ----------
    q : NDArray[np.float64]
        Initial generalized coordinates of shape `(dof,)`.
    p : NDArray[np.float64]
        Initial generalized momenta of shape `(dof,)`.
    total_time : np.float64
        Total integration time.
    time_step : np.float64
        Integration time step.
    parameters : NDArray[np.float64]
        Additional system parameters.
    grad_T : grad_t
        Gradient of the kinetic energy with respect to the momenta.
    grad_V : grad_t
        Gradient of the potential energy with respect to the coordinates.
    hess_T : hess_t
        Hessian of the kinetic energy with respect to the momenta.
    hess_V : hess_t
        Hessian of the potential energy with respect to the coordinates.
    k : int
        Number of deviation vectors used in the computation.
    return_history : bool
        If True, return the time evolution of LDI.
    seed : int
        Random seed used to initialize the deviation vectors.
    integrator_traj_tan : symplectic_tangent_step_t
        Symplectic step for the trajectory and tangent dynamics.
    threshold : np.float64
        Early stopping threshold. Integration stops when LDI falls below this value.

    Returns
    -------
    NDArray[np.float64]
        - If `return_history=True`, returns an array of shape `(n_samples, 2)`
          whose first column is time and whose second column is `LDI_k`.
        - If `return_history=False`, returns an array of shape `(1, 2)`
          containing the final time and final `LDI_k` value.
    """
    num_steps = round(total_time / time_step)
    dof = len(q)
    neq = 2 * dof

    np.random.seed(seed)
    dv = -np.float64(1.0) + np.float64(2.0) * np.random.rand(neq, k)
    dv, _ = qr(dv)
    dv = np.ascontiguousarray(dv)

    history = np.zeros((num_steps, 2), dtype=np.float64)
    ldi_val = np.float64(0.0)
    time = np.float64(0.0)
    count = 0

    for i in range(num_steps):
        time = np.float64(i + 1) * time_step

        q, p, dv = integrator_traj_tan(
            q,
            p,
            dv,
            time_step,
            grad_T,
            grad_V,
            hess_T,
            hess_V,
            parameters,
        )

        for j in range(k):
            norm = np.linalg.norm(dv[:, j])
            dv[:, j] /= norm

        _, S, _ = np.linalg.svd(dv, full_matrices=False)
        ldi_val = np.exp(np.sum(np.log(S)))

        if return_history:
            history[count, 0] = time
            history[count, 1] = ldi_val
            count += 1

        if ldi_val <= threshold:
            break

    if return_history:
        return history[:count]

    result = np.zeros((1, 2), dtype=np.float64)
    result[0, 0] = time
    result[0, 1] = ldi_val
    return result
