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

from numpy.typing import NDArray
import numpy as np
from numba import njit

from pynamicalsys.common.types import flow_t, flow_jacobian_t
from pynamicalsys.common.linalg import qr
from pynamicalsys.continuous_time.step_methods import rk4_step_wrapped
from pynamicalsys.continuous_time.step import evolve_system, step


@njit
def lyapunov_exponents(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: np.float64,
    equations_of_motion: flow_t,
    jacobian: flow_jacobian_t,
    num_exponents: int,
    transient_time: np.float64 | None = None,
    time_step: np.float64 = np.float64(0.01),
    atol: np.float64 = np.float64(1e-6),
    rtol: np.float64 = np.float64(1e-3),
    integrator=rk4_step_wrapped,
    return_history: bool = False,
    seed: int = 13,
    method: str = "QR",
) -> NDArray[np.float64]:
    """
    Compute the first `num_exponents` Lyapunov exponents of a continuous-time
    dynamical system.

    Parameters
    ----------
    u : NDArray[np.float64]
        Initial state vector.
    parameters : NDArray[np.float64]
        System parameters passed to the equations of motion and Jacobian.
    total_time : np.float64
        Final integration time.
    equations_of_motion : flow_t
        Continuous-time vector field with signature
        `(time, u, parameters) -> dudt`.
    jacobian : flow_jacobian_t
        Jacobian of the vector field with signature
        `(time, u, parameters) -> J`.
    num_exponents : int
        Number of Lyapunov exponents to compute.
    transient_time : np.float64 | None, optional
        Initial integration time discarded before exponent accumulation.
    time_step : np.float64, optional
        Initial integration step size.
    atol : np.float64, optional
        Absolute tolerance used by adaptive integrators.
    rtol : np.float64, optional
        Relative tolerance used by adaptive integrators.
    integrator : callable, optional
        Step method with unified signature returning
        `(u_next, time_next, time_step_next, accept)`.
    return_history : bool, optional
        If True, return the time evolution of the finite-time Lyapunov
        exponents.
    seed : int, optional
        Random seed used to initialize the deviation vectors.
    QR : callable, optional
        QR decomposition routine used to reorthonormalize the deviation
        vectors.

    Returns
    -------
    NDArray[np.float64]
        - If `return_history=False`, returns an array of shape `(1, num_exponents)`
          containing the final Lyapunov exponents.
        - If `return_history=True`, returns an array of shape
          `(n_samples, num_exponents + 1)` whose first column contains time and
          whose remaining columns contain the finite-time exponents.

    Notes
    -----
    The computation evolves the state together with `num_exponents` deviation
    vectors. After each accepted integration step, the deviation vectors are
    reorthonormalized through a QR decomposition and the logarithms of the
    diagonal entries of `R` are accumulated.
    """
    neq = len(u)
    nt = neq + neq * num_exponents

    u = u.copy()

    if transient_time is not None:
        u = evolve_system(
            u=u,
            parameters=parameters,
            total_time=transient_time,
            equations_of_motion=equations_of_motion,
            time_step=time_step,
            atol=atol,
            rtol=rtol,
            integrator=integrator,
        )
        time = transient_time
    else:
        time = np.float64(0.0)

    uv = np.zeros(nt, dtype=np.float64)
    uv[:neq] = u.copy()

    np.random.seed(seed)
    uv[neq:] = -1.0 + 2.0 * np.random.rand(nt - neq)
    v = uv[neq:].reshape(neq, num_exponents)
    if method == "QR":
        v, _ = qr(v)
    else:
        v, _ = np.linalg.qr(v)
        v = np.ascontiguousarray(v)
    uv[neq:] = v.reshape(neq * num_exponents)

    exponents = np.zeros(num_exponents, dtype=np.float64)
    history = []

    t_ref = transient_time if transient_time is not None else np.float64(0.0)

    while time < total_time:
        if time + time_step > total_time:
            time_step = total_time - time

        uv, time, time_step = step(
            time=time,
            u=uv,
            parameters=parameters,
            equations_of_motion=equations_of_motion,
            jacobian=jacobian,
            time_step=time_step,
            atol=atol,
            rtol=rtol,
            integrator=integrator,
            number_of_deviation_vectors=num_exponents,
        )

        v = uv[neq:].reshape(neq, num_exponents).copy()
        if method == "QR":
            v, R = qr(v)
        else:
            v, R = np.linalg.qr(v)
            v = np.ascontiguousarray(v)
            R = np.ascontiguousarray(R)
        exponents += np.log(np.abs(np.diag(R)))

        if return_history:
            result = [time]
            for i in range(num_exponents):
                result.append(exponents[i] / (time - t_ref))
            history.append(result)

        uv[neq:] = v.reshape(neq * num_exponents)

    if return_history:
        return np.asarray(history, dtype=np.float64)

    result = np.empty((1, num_exponents), dtype=np.float64)
    for i in range(num_exponents):
        result[0, i] = exponents[i] / (time - t_ref)

    return result


@njit
def maximum_lyapunov_exponent(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: np.float64,
    equations_of_motion: flow_t,
    jacobian: flow_jacobian_t,
    transient_time: np.float64 | None = None,
    time_step: np.float64 = np.float64(0.01),
    atol: np.float64 = np.float64(1e-6),
    rtol: np.float64 = np.float64(1e-3),
    integrator=rk4_step_wrapped,
    return_history: bool = False,
    seed: int = 13,
) -> NDArray[np.float64]:
    """
    Compute the maximum Lyapunov exponent of a continuous-time dynamical system.

    Parameters
    ----------
    u : NDArray[np.float64]
        Initial state vector.
    parameters : NDArray[np.float64]
        System parameters passed to the equations of motion and Jacobian.
    total_time : np.float64
        Final integration time.
    equations_of_motion : flow_t
        Continuous-time vector field with signature
        `(time, u, parameters) -> dudt`.
    jacobian : flow_jacobian_t
        Jacobian of the vector field with signature
        `(time, u, parameters) -> J`.
    transient_time : np.float64 | None, optional
        Initial integration time discarded before exponent accumulation.
    time_step : np.float64, optional
        Initial integration step size.
    atol : np.float64, optional
        Absolute tolerance used by adaptive integrators.
    rtol : np.float64, optional
        Relative tolerance used by adaptive integrators.
    integrator : callable, optional
        Step method with unified signature returning
        `(u_next, time_next, time_step_next, accept)`.
    return_history : bool, optional
        If True, return the time evolution of the finite-time maximum Lyapunov
        exponent.
    seed : int, optional
        Random seed used to initialize the deviation vector.

    Returns
    -------
    NDArray[np.float64]
        - If `return_history=False`, returns an array of shape `(1, 1)`
          containing the final maximum Lyapunov exponent.
        - If `return_history=True`, returns an array of shape `(n_samples, 2)`
          whose first column contains time and whose second column contains the
          finite-time maximum Lyapunov exponent.

    Notes
    -----
    The computation evolves the state together with one deviation vector. After
    each accepted integration step, the deviation vector is renormalized and
    the logarithm of its norm growth is accumulated.
    """
    neq = len(u)
    nt = neq + neq

    u = u.copy()

    if transient_time is not None:
        u = evolve_system(
            u=u,
            parameters=parameters,
            total_time=transient_time,
            equations_of_motion=equations_of_motion,
            time_step=time_step,
            atol=atol,
            rtol=rtol,
            integrator=integrator,
        )
        time = transient_time
    else:
        time = np.float64(0.0)

    uv = np.zeros(nt, dtype=np.float64)
    uv[:neq] = u.copy()

    np.random.seed(seed)
    uv[neq:] = -1.0 + 2.0 * np.random.rand(nt - neq)
    norm = np.linalg.norm(uv[neq:])
    uv[neq:] /= norm

    exponent = np.float64(0.0)
    history = []

    t_ref = transient_time if transient_time is not None else np.float64(0.0)

    while time < total_time:
        if time + time_step > total_time:
            time_step = total_time - time

        uv, time, time_step = step(
            time=time,
            u=uv,
            parameters=parameters,
            equations_of_motion=equations_of_motion,
            jacobian=jacobian,
            time_step=time_step,
            atol=atol,
            rtol=rtol,
            integrator=integrator,
            number_of_deviation_vectors=1,
        )

        norm = np.linalg.norm(uv[neq:])
        exponent += np.log(np.abs(norm))
        uv[neq:] /= norm

        if return_history:
            history.append([time, exponent / (time - t_ref)])

    if return_history:
        return np.asarray(history, dtype=np.float64)

    result = np.empty((1, 1), dtype=np.float64)
    result[0, 0] = exponent / (time - t_ref)

    return result
