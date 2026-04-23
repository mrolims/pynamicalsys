# lyapunov.py

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
def lyapunov_spectrum(
    q: NDArray[np.float64],
    p: NDArray[np.float64],
    total_time: np.float64,
    time_step: np.float64,
    parameters: NDArray[np.float64],
    grad_T: grad_t,
    grad_V: grad_t,
    hess_T: hess_t,
    hess_V: hess_t,
    num_exponents: int,
    qr_interval: int,
    return_history: bool,
    seed: int,
    log_base: np.float64,
    method: str,
    integrator_traj_tan: symplectic_tangent_step_t,
) -> NDArray[np.float64]:
    """
    Compute the Lyapunov spectrum of a Hamiltonian system.

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
    grad_T : grad_t
        Gradient of the kinetic energy with respect to the momenta.
    grad_V : grad_t
        Gradient of the potential energy with respect to the coordinates.
    hess_T : hess_t
        Hessian of the kinetic energy with respect to the momenta.
    hess_V : hess_t
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
def largest_lyapunov_exponent(
    q: NDArray[np.float64],
    p: NDArray[np.float64],
    total_time: np.float64,
    time_step: np.float64,
    parameters: NDArray[np.float64],
    grad_T: grad_t,
    grad_V: grad_t,
    hess_T: hess_t,
    hess_V: hess_t,
    return_history: bool,
    seed: int,
    log_base: np.float64,
    integrator_traj_tan: symplectic_tangent_step_t,
) -> NDArray[np.float64]:
    """
    Compute the largest Lyapunov exponent of a Hamiltonian system.

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
    grad_T : grad_t
        Gradient of the kinetic energy with respect to the momenta.
    grad_V : grad_t
        Gradient of the potential energy with respect to the coordinates.
    hess_T : hess_t
        Hessian of the kinetic energy with respect to the momenta.
    hess_V : hess_t
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
