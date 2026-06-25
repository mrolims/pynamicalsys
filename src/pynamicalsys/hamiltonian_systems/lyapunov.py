# lyapunov.py

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

"""
TODO

- Factor out the common Lyapunov logic. The only difference
  between the integrators is the derivative callbacks
  and signatures:
      (grad_T, grad_V, hess_T, hess_V) vs. (eom, hess_H).
"""

from numpy.typing import NDArray
import numpy as np
from numba import njit
from pynamicalsys.common.types import (
    system_func_t,
    symplectic_tangent_step_t,
)
from pynamicalsys.common.linalg import qr


@njit
def lyapunov_spectrum_sep(
    q: NDArray[np.float64],
    p: NDArray[np.float64],
    total_time: np.float64,
    time_step: np.float64,
    parameters: NDArray[np.float64],
    grad_T: system_func_t,
    grad_V: system_func_t,
    hess_T: system_func_t,
    hess_V: system_func_t,
    num_exponents: int,
    qr_interval: int,
    return_history: bool,
    seed: int,
    log_base: np.float64,
    method: str,
    integrator_traj_tan: symplectic_tangent_step_t,
) -> NDArray[np.float64]:
    """
    Compute the Lyapunov spectrum of a separable Hamiltonian system,
    H(q, p) = T(p) + V(q), integrated with an explicit symplectic
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
        Integration step size.
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
    num_exponents : int
        Number of Lyapunov exponents to compute.
    qr_interval : int
        Number of integration steps between successive QR re-orthonormalizations.
    return_history : bool
        If True, return the time evolution of the spectrum.
    seed : int
        Random seed used to initialize the deviation vectors.
    log_base : np.float64
        Base of the logarithm used to normalize the exponents.
    method : str
        QR method used in the orthonormalization step. Supported options are:

        - `"QR"`:
          Use the internal reduced modified Gram-Schmidt QR routine `qr`.

        - `"QR_HH"`:
          Use `numpy.linalg.qr`, based on Householder reflections.

    integrator_traj_tan : symplectic_tangent_step_t
        Symplectic step for the trajectory and tangent dynamics.

    Returns
    -------
    NDArray[np.float64]
        - If `return_history=True`, returns an array of shape
          `(num_steps // qr_interval, num_exponents + 1)` whose first column is
          time and whose remaining columns are the running Lyapunov exponents.
        - If `return_history=False`, returns an array of shape `(1, num_exponents)`
          containing the final Lyapunov spectrum.
    """
    num_steps = round(total_time / time_step)
    dof = len(q)
    neq = 2 * dof

    np.random.seed(seed)
    dv = -np.float64(1.0) + np.float64(2.0) * np.random.rand(neq, num_exponents)

    if method == "QR":
        dv, _ = qr(dv)
    elif method == "QR_HH":
        dv, _ = np.linalg.qr(dv)
    else:
        raise ValueError("method must be 'QR' or 'QR_HH'")

    dv = np.ascontiguousarray(dv)

    exponents = np.zeros(num_exponents, dtype=np.float64)
    n_qr = num_steps // qr_interval
    history = np.zeros((n_qr, num_exponents + 1), dtype=np.float64)

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

        if (i + 1) % qr_interval == 0:
            if method == "QR":
                dv, R = qr(dv)
            else:
                dv, R = np.linalg.qr(dv)

            dv = np.ascontiguousarray(dv)
            exponents += np.log(np.abs(np.diag(R)))

            if return_history:
                history[count, 0] = time
                history[count, 1:] = exponents / time
                count += 1

    if return_history:
        history[:, 1:] /= np.log(log_base)
        return history

    spectrum = np.zeros((1, num_exponents), dtype=np.float64)
    spectrum[0, :] = exponents / (total_time * np.log(log_base))
    return spectrum


@njit
def lyapunov_spectrum_imp(
    q: NDArray[np.float64],
    p: NDArray[np.float64],
    total_time: np.float64,
    time_step: np.float64,
    parameters: NDArray[np.float64],
    eom: system_func_t,
    hess_H: system_func_t,
    num_exponents: int,
    qr_interval: int,
    return_history: bool,
    seed: int,
    log_base: np.float64,
    method: str,
    tol: np.float64,
    max_iter: int,
    integrator_traj_tan: symplectic_tangent_step_t,
) -> NDArray[np.float64]:
    """
    Compute the Lyapunov spectrum of a general (possibly non-separable)
    Hamiltonian system H(q, p), integrated with the implicit midpoint method.

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
        Integration step size.
    parameters : NDArray[np.float64]
        Additional system parameters.
    eom : system_func_t
        Equations of motion of the system.
    hess_H : system_func_t
        Hessian of the Hamiltonian w.r.t. z = (q, p).
    num_exponents : int
        Number of Lyapunov exponents to compute.
    qr_interval : int
        Number of integration steps between successive QR re-orthonormalizations.
    return_history : bool
        If True, return the time evolution of the spectrum.
    seed : int
        Random seed used to initialize the deviation vectors.
    log_base : np.float64
        Base of the logarithm used to normalize the exponents.
    method : str
        QR method used in the orthonormalization step. Supported options are:

        - `"QR"`:
          Use the internal reduced modified Gram-Schmidt QR routine `qr`.

        - `"QR_HH"`:
          Use `numpy.linalg.qr`, based on Householder reflections.

    integrator_traj_tan : symplectic_tangent_step_t
        Symplectic step for the trajectory and tangent dynamics.
    tol : np.float64
        Newton convergence tolerance on the residual norm.
    max_iter : int
        Maximum Newton iterations per step.

    Returns
    -------
    NDArray[np.float64]
        - If `return_history=True`, returns an array of shape
          `(num_steps // qr_interval, num_exponents + 1)` whose first column is
          time and whose remaining columns are the running Lyapunov exponents.
        - If `return_history=False`, returns an array of shape `(1, num_exponents)`
          containing the final Lyapunov spectrum.
    """
    num_steps = round(total_time / time_step)
    dof = len(q)
    neq = 2 * dof

    np.random.seed(seed)
    dv = -np.float64(1.0) + np.float64(2.0) * np.random.rand(neq, num_exponents)

    if method == "QR":
        dv, _ = qr(dv)
    elif method == "QR_HH":
        dv, _ = np.linalg.qr(dv)
    else:
        raise ValueError("method must be 'QR' or 'QR_HH'")

    dv = np.ascontiguousarray(dv)

    exponents = np.zeros(num_exponents, dtype=np.float64)
    n_qr = num_steps // qr_interval
    history = np.zeros((n_qr, num_exponents + 1), dtype=np.float64)

    count = 0
    for i in range(num_steps):
        time = np.float64(i + 1) * time_step

        q, p, dv = integrator_traj_tan(
            q, p, dv, time_step, eom, hess_H, parameters, tol, max_iter
        )

        if (i + 1) % qr_interval == 0:
            if method == "QR":
                dv, R = qr(dv)
            else:
                dv, R = np.linalg.qr(dv)

            dv = np.ascontiguousarray(dv)
            exponents += np.log(np.abs(np.diag(R)))

            if return_history:
                history[count, 0] = time
                history[count, 1:] = exponents / time
                count += 1

    if return_history:
        history[:, 1:] /= np.log(log_base)
        return history

    spectrum = np.zeros((1, num_exponents), dtype=np.float64)
    spectrum[0, :] = exponents / (total_time * np.log(log_base))
    return spectrum


@njit
def largest_lyapunov_exponent_sep(
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
    log_base: np.float64,
    integrator_traj_tan: symplectic_tangent_step_t,
) -> NDArray[np.float64]:
    """
    Compute the largest Lyapunov exponent of a separable Hamiltonian
    systems H(q, p) = T(p) + V(q), integrated with an explicit
    symplectic stepper (velocity Verlet or fourth-order Yoshida).

    Requires grad_T, grad_V, hess_T, and hess_V to advance the trajectory
    and tangent vector.

    Parameters
    ----------
    q : NDArray[np.float64]
        Initial generalized coordinates of shape `(dof,)`.
    p : NDArray[np.float64]
        Initial generalized momenta of shape `(dof,)`.
    total_time : np.float64
        Total integration time.
    time_step : np.float64
        Integration step size.
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
        If True, return the time evolution of the largest Lyapunov exponent.
    seed : int
        Random seed used to initialize the deviation vector.
    log_base : np.float64
        Base of the logarithm used to normalize the exponent.
    integrator_traj_tan : symplectic_tangent_step_t
        Symplectic step for the trajectory and tangent dynamics.

    Returns
    -------
    NDArray[np.float64]
        - If `return_history=True`, returns an array of shape `(num_steps, 2)`
          whose first column is time and whose second column is the running
          largest Lyapunov exponent.
        - If `return_history=False`, returns an array of shape `(1, 1)`
          containing the final largest Lyapunov exponent.
    """
    num_steps = round(total_time / time_step)
    dof = len(q)

    np.random.seed(seed)
    dv = np.random.uniform(-np.float64(1.0), np.float64(1.0), 2 * dof)
    norm = np.linalg.norm(dv)
    dv /= norm
    dv = dv.reshape(2 * dof, 1)
    dv = np.ascontiguousarray(dv)

    lyapunov_exponent = np.float64(0.0)
    history = np.zeros((num_steps, 2), dtype=np.float64)
    time = np.float64(0.0)

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

        norm = np.linalg.norm(dv[:, 0])
        lyapunov_exponent += np.log(norm)
        dv /= norm

        if return_history:
            history[i, 0] = time
            history[i, 1] = lyapunov_exponent / time

    if return_history:
        history[:, 1] /= np.log(log_base)
        return history

    result = np.zeros((1, 1), dtype=np.float64)
    result[0, 0] = lyapunov_exponent / (time * np.log(log_base))
    return result


@njit
def largest_lyapunov_exponent_imp(
    q: NDArray[np.float64],
    p: NDArray[np.float64],
    total_time: np.float64,
    time_step: np.float64,
    parameters: NDArray[np.float64],
    eom: system_func_t,
    hess_H: system_func_t,
    return_history: bool,
    seed: int,
    log_base: np.float64,
    tol: np.float64,
    max_iter: int,
    integrator_traj_tan: symplectic_tangent_step_t,
) -> NDArray[np.float64]:
    """
    Compute the largest Lyapunov exponent of a general (possibly non-separable)
    Hamiltonian system H(q, p), integrated with the implicit midpoint method.

    Requires eom and hess_H to jointly advance the trajectory and tangent
    vector via implicit_midpoint_step_traj_tan.

    Parameters
    ----------
    q : NDArray[np.float64]
        Initial generalized coordinates of shape `(dof,)`.
    p : NDArray[np.float64]
        Initial generalized momenta of shape `(dof,)`.
    total_time : np.float64
        Total integration time.
    time_step : np.float64
        Integration step size.
    parameters : NDArray[np.float64]
        Additional system parameters.
    eom : system_func_t
        Equations of motion of the system.
    hess_H : system_func_t
        Hessian of the Hamiltonian w.r.t. z = (q, p).
    return_history : bool
        If True, return the time evolution of the largest Lyapunov exponent.
    seed : int
        Random seed used to initialize the deviation vector.
    log_base : np.float64
        Base of the logarithm used to normalize the exponent.
    integrator_traj_tan : symplectic_tangent_step_t
        Symplectic step for the trajectory and tangent dynamics.
    tol : np.float64
        Newton convergence tolerance on the residual norm.
    max_iter : int
        Maximum Newton iterations per step.

    Returns
    -------
    NDArray[np.float64]
        - If `return_history=True`, returns an array of shape `(num_steps, 2)`
          whose first column is time and whose second column is the running
          largest Lyapunov exponent.
        - If `return_history=False`, returns an array of shape `(1, 1)`
          containing the final largest Lyapunov exponent.
    """
    num_steps = round(total_time / time_step)
    dof = len(q)

    np.random.seed(seed)
    dv = np.random.uniform(-np.float64(1.0), np.float64(1.0), 2 * dof)
    norm = np.linalg.norm(dv)
    dv /= norm
    dv = dv.reshape(2 * dof, 1)
    dv = np.ascontiguousarray(dv)

    lyapunov_exponent = np.float64(0.0)
    history = np.zeros((num_steps, 2), dtype=np.float64)
    time = np.float64(0.0)

    for i in range(num_steps):
        time = np.float64(i + 1) * time_step

        q, p, dv = integrator_traj_tan(
            q, p, dv, time_step, eom, hess_H, parameters, tol, max_iter
        )

        norm = np.linalg.norm(dv[:, 0])
        lyapunov_exponent += np.log(norm)
        dv /= norm

        if return_history:
            history[i, 0] = time
            history[i, 1] = lyapunov_exponent / time

    if return_history:
        history[:, 1] /= np.log(log_base)
        return history

    result = np.zeros((1, 1), dtype=np.float64)
    result[0, 0] = lyapunov_exponent / (time * np.log(log_base))
    return result
