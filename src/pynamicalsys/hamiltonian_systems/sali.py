# sali.py

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


import numpy as np
from numba import njit
from numpy.typing import NDArray
from pynamicalsys.common.linalg import qr
from pynamicalsys.common.types import system_func_t, symplectic_tangent_step_t


"""
TODO

- Factor out the common SALI logic. The only difference
  between the integrators is the derivative callbacks
  and signatures:
      (grad_T, grad_V, hess_T, hess_V) vs. (eom, hess_H).
"""


@njit
def sali_sep(
    q: NDArray[np.float64],
    p: NDArray[np.float64],
    total_time: np.float64,
    time_step: np.float64,
    parameters: NDArray[np.float64],
    grad_T: system_func_t,
    grad_V: system_func_t,
    hess_T: system_func_t,
    hess_V: system_func_t,
    return_history: bool,
    seed: int,
    threshold: np.float64,
    integrator_traj_tan: symplectic_tangent_step_t,
) -> NDArray[np.float64]:
    """
    Compute the Smaller Alignment Index (SALI) for a separable Hamiltonian
    system, H(q, p) = T(p) + V(q), integrated with an explicit symplectic
    stepper (velocity Verlet or fourth-order Yoshida).

    Requires grad_T, grad_V, hess_T, and hess_V to advance the trajectory
    and tangent vectors.

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
    grad_T : system_func_t
        Gradient of the kinetic energy with respect to the momenta.
    grad_V : system_func_t
        Gradient of the potential energy with respect to the coordinates.
    hess_T : system_func_t
        Hessian of the kinetic energy with respect to the momenta.
    hess_V : system_func_t
        Hessian of the potential energy with respect to the coordinates.
    return_history : bool
        If True, return the time evolution of SALI.
    seed : int
        Random seed used to initialize the deviation vectors.
    threshold : np.float64
        Early stopping threshold. Integration stops when SALI falls below this value.
    integrator_traj_tan : symplectic_tangent_step_t
        Symplectic step for the trajectory and tangent dynamics.

    Returns
    -------
    NDArray[np.float64]
        - If `return_history=True`, returns an array of shape `(n_samples, 2)`
          whose first column is time and whose second column is SALI.
        - If `return_history=False`, returns an array of shape `(1, 2)`
          containing the final time and final SALI value.
    """
    num_steps = round(total_time / time_step)
    dof = len(q)
    neq = 2 * dof

    np.random.seed(seed)
    dv = -np.float64(1.0) + np.float64(2.0) * np.random.rand(neq, 2)
    dv, _ = qr(dv)
    dv = np.ascontiguousarray(dv)

    history = np.zeros((num_steps, 2), dtype=np.float64)
    sali_val = np.float64(0.0)
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

        for j in range(2):
            norm = np.linalg.norm(dv[:, j])
            dv[:, j] /= norm

        pai = np.linalg.norm(dv[:, 0] + dv[:, 1])
        aai = np.linalg.norm(dv[:, 0] - dv[:, 1])
        sali_val = min(pai, aai)

        if return_history:
            history[count, 0] = time
            history[count, 1] = sali_val
            count += 1

        if sali_val <= threshold:
            break

    if return_history:
        return history[:count]

    result = np.zeros((1, 2), dtype=np.float64)
    result[0, 0] = time
    result[0, 1] = sali_val
    return result


@njit
def sali_imp(
    q: NDArray[np.float64],
    p: NDArray[np.float64],
    total_time: np.float64,
    time_step: np.float64,
    parameters: NDArray[np.float64],
    eom: system_func_t,
    hess_H: system_func_t,
    return_history: bool,
    seed: int,
    threshold: np.float64,
    tol: np.float64,
    max_iter: int,
    integrator_traj_tan: symplectic_tangent_step_t,
) -> NDArray[np.float64]:
    """
    Compute the Smaller Alignment Index (SALI) for a general (possibly
    non-separable) Hamiltonian system H(q, p), integrated with the implicit
    midpoint method.

    Requires eom and hess_H to jointly advance the trajectory and tangent
    vectors via implicit_midpoint_step_traj_tan.

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
    eom : system_func_t
        Equations of motion of the system.
    hess_H : system_func_t
        Hessian of the Hamiltonian w.r.t. z = (q, p).
    return_history : bool
        If True, return the time evolution of SALI.
    seed : int
        Random seed used to initialize the deviation vectors.
    threshold : np.float64
        Early stopping threshold. Integration stops when SALI falls below this value.
    tol : np.float64
        Newton convergence tolerance on the residual norm.
    max_iter : int
        Maximum Newton iterations per step.
    integrator_traj_tan : symplectic_tangent_step_t
        Symplectic step for the trajectory and tangent dynamics.

    Returns
    -------
    NDArray[np.float64]
        - If `return_history=True`, returns an array of shape `(n_samples, 2)`
          whose first column is time and whose second column is SALI.
        - If `return_history=False`, returns an array of shape `(1, 2)`
          containing the final time and final SALI value.
    """
    num_steps = round(total_time / time_step)
    dof = len(q)
    neq = 2 * dof

    np.random.seed(seed)
    dv = -np.float64(1.0) + np.float64(2.0) * np.random.rand(neq, 2)
    dv, _ = qr(dv)
    dv = np.ascontiguousarray(dv)

    history = np.zeros((num_steps, 2), dtype=np.float64)
    sali_val = np.float64(0.0)
    time = np.float64(0.0)
    count = 0

    for i in range(num_steps):
        time = np.float64(i + 1) * time_step

        q, p, dv = integrator_traj_tan(
            q, p, dv, time_step, eom, hess_H, parameters, tol, max_iter
        )

        for j in range(2):
            norm = np.linalg.norm(dv[:, j])
            dv[:, j] /= norm

        pai = np.linalg.norm(dv[:, 0] + dv[:, 1])
        aai = np.linalg.norm(dv[:, 0] - dv[:, 1])
        sali_val = min(pai, aai)

        if return_history:
            history[count, 0] = time
            history[count, 1] = sali_val
            count += 1

        if sali_val <= threshold:
            break

    if return_history:
        return history[:count]

    result = np.zeros((1, 2), dtype=np.float64)
    result[0, 0] = time
    result[0, 1] = sali_val
    return result
