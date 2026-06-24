# trajectory.py

# Copyright (C) 2025-2026 Matheus Rolim Sales
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
from numba import njit, prange
from pynamicalsys.common.types import system_func_t, symplectic_step_t


@njit
def generate_trajectory(
    q: NDArray[np.float64],
    p: NDArray[np.float64],
    total_time: np.float64,
    parameters: NDArray[np.float64],
    grad_T: system_func_t,
    grad_V: system_func_t,
    time_step: np.float64,
    integrator: symplectic_step_t,
    tol: np.float64 = np.float64(1e-12),
    max_iter: int = 50,
) -> NDArray[np.float64]:
    """
    Generate a single Hamiltonian trajectory using a symplectic integrator.

    Parameters
    ----------
    q : NDArray[np.float64]
        Initial generalized coordinates of shape `(dof,)`.
    p : NDArray[np.float64]
        Initial generalized momenta of shape `(dof,)`.
    total_time : np.float64
        Total integration time.
    parameters : NDArray[np.float64]
        Additional parameters passed to `grad_T` and `grad_V`.
    grad_T : system_func_t
        Gradient of the kinetic energy with respect to the momenta.
    grad_V : system_func_t
        Gradient of the potential energy with respect to the coordinates.
    time_step : np.float64
        Integration time step.
    integrator : symplectic_step_t
        Symplectic integration step.
    tol : np.float64
        Newton convergence tolerance on the residual norm.
    max_iter : int
        Maximum Newton iterations per step.

    Returns
    -------
    NDArray[np.float64]
        Trajectory array of shape `(num_steps + 1, 2 * dof + 1)` with:
        - column 0 containing time
        - columns `1:dof+1` containing coordinates
        - columns `dof+1:2*dof+1` containing momenta
    """
    num_steps = round(total_time / time_step)
    dof = len(q)

    result = np.zeros((num_steps + 1, 2 * dof + 1), dtype=np.float64)
    result[0, 1 : dof + 1] = q
    result[0, dof + 1 :] = p

    for i in range(1, num_steps + 1):
        q, p = integrator(q, p, time_step, grad_T, grad_V, parameters, tol, max_iter)
        result[i, 0] = np.float64(i) * time_step
        result[i, 1 : dof + 1] = q
        result[i, dof + 1 :] = p

    return result


@njit(parallel=True)
def ensemble_trajectories(
    q: NDArray[np.float64],
    p: NDArray[np.float64],
    total_time: np.float64,
    parameters: NDArray[np.float64],
    grad_T: system_func_t,
    grad_V: system_func_t,
    time_step: np.float64,
    integrator: symplectic_step_t,
    tol: np.float64 = np.float64(1e-12),
    max_iter: int = 50,
) -> NDArray[np.float64]:
    """
    Generate an ensemble of Hamiltonian trajectories using a symplectic integrator.

    Parameters
    ----------
    q : NDArray[np.float64]
        Initial generalized coordinates of shape `(num_ic, dof)`.
    p : NDArray[np.float64]
        Initial generalized momenta of shape `(num_ic, dof)`.
    total_time : np.float64
        Total integration time.
    parameters : NDArray[np.float64]
        Additional parameters passed to `grad_T` and `grad_V`.
    grad_T : system_func_t
        Gradient of the kinetic energy with respect to the momenta.
    grad_V : system_func_t
        Gradient of the potential energy with respect to the coordinates.
    time_step : np.float64
        Integration time step.
    integrator : symplectic_step_t
        Symplectic integration step.
    tol : np.float64
        Newton convergence tolerance on the residual norm.
    max_iter : int
        Maximum Newton iterations per step.

    Returns
    -------
    NDArray[np.float64]
        Array of shape `(num_ic, num_steps + 1, 2 * dof + 1)` containing the
        trajectories for all initial conditions.
    """
    num_steps = round(total_time / time_step)
    num_ic, dof = q.shape

    trajectories = np.zeros(
        (num_ic, num_steps + 1, 2 * dof + 1),
        dtype=np.float64,
    )

    for i in prange(num_ic):
        trajectories[i] = generate_trajectory(
            q[i],
            p[i],
            total_time,
            parameters,
            grad_T,
            grad_V,
            time_step,
            integrator,
            tol=tol,
            max_iter=max_iter,
        )

    return trajectories
