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
from pynamicalsys.common.linalg import qr, qr_inplace_log_diagonal
from pynamicalsys.continuous_time.step_methods import rk4_step_wrapped
from pynamicalsys.continuous_time.step import evolve_system, step_to
from pynamicalsys.continuous_time.buffers import ensure_row, new_buffer


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
    sample_times: NDArray[np.float64] | None = None,
    seed: int = 13,
    method: str = "QR",
    qr_interval: int = 1,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
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
    sample_times : NDArray[np.float64] | None, optional
        Sorted times in `(transient_time, total_time]` at which the history
        is stored when `return_history=True`. The integration lands exactly
        on each of them. If None, the history is stored at every
        reorthonormalization.
    seed : int, optional
        Random seed used to initialize the deviation vectors.
    method : str, optional
        QR decomposition used to reorthonormalize the deviation vectors.
        Supported options are:

        - `"QR"`: custom modified Gram-Schmidt QR (default).
        - `"QR_HH"`: `numpy.linalg.qr`, based on Householder reflections.
    qr_interval : int, optional
        Number of accepted integration steps between successive QR
        reorthonormalizations. A reorthonormalization is also performed at the
        final integration time.

    Returns
    -------
    tuple[NDArray[np.float64], NDArray[np.float64]]
        The result and the state of the system at the time the integration
        stopped. The result is:

        - If `return_history=False`, returns an array of shape `(1, num_exponents)`
          containing the final Lyapunov exponents.
        - If `return_history=True`, returns an array of shape
          `(n_samples, num_exponents + 1)` whose first column contains time and
          whose remaining columns contain the finite-time exponents. One row
          is stored at each reorthonormalization.

    Notes
    -----
    The computation evolves the state together with `num_exponents` deviation
    vectors. Every `qr_interval` accepted integration steps, and at the final
    integration time, the deviation vectors are reorthonormalized through a QR
    decomposition and the logarithms of the diagonal entries of `R` are
    accumulated.
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
    log_diagonal = np.empty(num_exponents, dtype=np.float64)
    history = new_buffer(num_exponents + 1)
    count = 0

    t_ref = transient_time if transient_time is not None else np.float64(0.0)

    steps_since_qr = 0
    num_samples = len(sample_times) if sample_times is not None else 0
    sample_index = 0
    while time < total_time:
        if sample_times is not None and sample_index < num_samples:
            target_time = sample_times[sample_index]
        else:
            target_time = total_time

        uv, time, time_step, reached = step_to(
            time=time,
            u=uv,
            target_time=target_time,
            parameters=parameters,
            equations_of_motion=equations_of_motion,
            jacobian=jacobian,
            time_step=time_step,
            atol=atol,
            rtol=rtol,
            integrator=integrator,
            number_of_deviation_vectors=num_exponents,
        )

        at_sample = reached and sample_times is not None and sample_index < num_samples
        if at_sample:
            sample_index += 1

        steps_since_qr += 1
        if steps_since_qr < qr_interval and time < total_time and not at_sample:
            continue
        steps_since_qr = 0

        if method == "QR":
            # Orthonormalize the deviation vectors in place, in the
            # extended state itself
            qr_inplace_log_diagonal(
                uv[neq:].reshape(neq, num_exponents), log_diagonal
            )
            exponents += log_diagonal
        else:
            v = uv[neq:].reshape(neq, num_exponents).copy()
            v, R = np.linalg.qr(v)
            v = np.ascontiguousarray(v)
            R = np.ascontiguousarray(R)
            exponents += np.log(np.abs(np.diag(R)))
            uv[neq:] = v.reshape(neq * num_exponents)

        if return_history and (sample_times is None or at_sample):
            history = ensure_row(history, count)
            history[count, 0] = time
            for i in range(num_exponents):
                history[count, i + 1] = exponents[i] / (time - t_ref)
            count += 1

    if return_history:
        return history[:count].copy(), uv[:neq].copy()

    result = np.empty((1, num_exponents), dtype=np.float64)
    for i in range(num_exponents):
        result[0, i] = exponents[i] / (time - t_ref)

    return result, uv[:neq].copy()


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
    sample_times: NDArray[np.float64] | None = None,
    seed: int = 13,
    qr_interval: int = 1,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
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
    sample_times : NDArray[np.float64] | None, optional
        Sorted times in `(transient_time, total_time]` at which the history
        is stored when `return_history=True`. The integration lands exactly
        on each of them. If None, the history is stored at every
        renormalization.
    seed : int, optional
        Random seed used to initialize the deviation vector.
    qr_interval : int, optional
        Number of accepted integration steps between successive
        renormalizations of the deviation vector. A renormalization is also
        performed at the final integration time.

    Returns
    -------
    tuple[NDArray[np.float64], NDArray[np.float64]]
        The result and the state of the system at the time the integration
        stopped. The result is:

        - If `return_history=False`, returns an array of shape `(1, 1)`
          containing the final maximum Lyapunov exponent.
        - If `return_history=True`, returns an array of shape `(n_samples, 2)`
          whose first column contains time and whose second column contains the
          finite-time maximum Lyapunov exponent. One row is stored at each
          renormalization.

    Notes
    -----
    The computation evolves the state together with one deviation vector.
    Every `qr_interval` accepted integration steps, and at the final
    integration time, the deviation vector is renormalized and the logarithm
    of its norm growth is accumulated.
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
    history = new_buffer(2)
    count = 0

    t_ref = transient_time if transient_time is not None else np.float64(0.0)

    steps_since_qr = 0
    num_samples = len(sample_times) if sample_times is not None else 0
    sample_index = 0
    while time < total_time:
        if sample_times is not None and sample_index < num_samples:
            target_time = sample_times[sample_index]
        else:
            target_time = total_time

        uv, time, time_step, reached = step_to(
            time=time,
            u=uv,
            target_time=target_time,
            parameters=parameters,
            equations_of_motion=equations_of_motion,
            jacobian=jacobian,
            time_step=time_step,
            atol=atol,
            rtol=rtol,
            integrator=integrator,
            number_of_deviation_vectors=1,
        )

        at_sample = reached and sample_times is not None and sample_index < num_samples
        if at_sample:
            sample_index += 1

        steps_since_qr += 1
        if steps_since_qr < qr_interval and time < total_time and not at_sample:
            continue
        steps_since_qr = 0

        norm = np.linalg.norm(uv[neq:])
        exponent += np.log(np.abs(norm))
        uv[neq:] /= norm

        if return_history and (sample_times is None or at_sample):
            history = ensure_row(history, count)
            history[count, 0] = time
            history[count, 1] = exponent / (time - t_ref)
            count += 1

    if return_history:
        return history[:count].copy(), uv[:neq].copy()

    result = np.empty((1, 1), dtype=np.float64)
    result[0, 0] = exponent / (time - t_ref)

    return result, uv[:neq].copy()
