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


from typing import Callable, Optional, Tuple
import numpy as np
from numba import njit
from numpy.typing import NDArray
from pynamicalsys.common.types import int_t, numeric_t, map_t, jacobian_t
from pynamicalsys.common.utils import householder_qr


@njit
def lyapunov_1D(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: int_t,
    mapping: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    derivative_mapping: Callable,
    sample_times: Optional[NDArray[np.integer]] = None,
    return_history: bool = False,
    transient_time: Optional[int_t] = None,
    log_base: numeric_t = np.e,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute the Lyapunov exponent for a 1-dimensional discrete dynamical system.

    Parameters
    ----------
    u : NDArray[np.float64]
        Initial condition of the system. For a 1-dimensional map, `u` must
        be a 1D array with shape `(1,)`.
    parameters : NDArray[np.float64]
        Parameter array passed to `mapping` and `derivative_mapping`.
    total_time : int_t
        Total number of iterations used in the computation.
    mapping : Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]
        Function defining the time evolution of the system.
    derivative_mapping : Callable
        Function returning the Jacobian of the map. For a 1-dimensional
        system, it must return a `(1, 1)` array containing the derivative
        of the map at the current state.
    sample_times : Optional[NDArray[np.integer]], optional
        Array of iteration times at which the finite-time Lyapunov exponent
        is recorded when `return_history=True`.
    return_history : bool, optional
        If True, return the finite-time Lyapunov exponent evaluated at
        `sample_times`. Otherwise, return only the final exponent estimate.
    transient_time : Optional[int_t], optional
        Number of initial iterations to discard as transient before starting
        the exponent accumulation. If None, no transient is discarded.
    log_base : numeric_t, optional
        Base of the logarithm used in the exponent calculation. Default is `np.e`.

    Returns
    -------
    Tuple[NDArray[np.float64], NDArray[np.float64]]
        - If `return_history=False`, returns:
          - a 1D array with shape `(1,)` containing the final Lyapunov exponent
          - the final state of shape `(1,)`
        - If `return_history=True`, returns:
          - a 1D array with shape `(len(sample_times),)` containing the
            finite-time Lyapunov exponent
          - the final state of shape `(1,)`

    Notes
    -----
    This function is compiled with Numba via `@njit`.
    """
    num_samples = len(sample_times) if sample_times is not None else 0
    log_den = np.log(log_base)
    exponent = 0.0

    sample_size = total_time - (transient_time if transient_time is not None else 0)
    if transient_time is not None:
        for _ in range(transient_time):
            u = mapping(u, parameters)

    history = np.empty(0, dtype=np.float64)
    if return_history and sample_times is not None:
        history = np.zeros(num_samples, dtype=np.float64)

    sample_idx = 0
    for n in range(sample_size):
        u = mapping(u, parameters)
        du = derivative_mapping(u, parameters, mapping)
        exponent += np.log(np.abs(du[0, 0]))

        if (
            return_history
            and sample_times is not None
            and sample_idx < num_samples
            and n + 1 == sample_times[sample_idx]
        ):
            history[sample_idx] = exponent / ((n + 1) * log_den)
            sample_idx += 1

    if return_history:
        return history, u

    return np.array([exponent / (sample_size * log_den)]), u


@njit
def lyapunov_er(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: int_t,
    mapping: map_t,
    jacobian: jacobian_t,
    sample_times: Optional[NDArray[np.integer]] = None,
    return_history: bool = False,
    transient_time: Optional[int_t] = None,
    log_base: numeric_t = np.e,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute the two Lyapunov exponents of a 2-dimensional discrete dynamical
    system using the Eckmann-Ruelle method.

    Parameters
    ----------
    u : NDArray[np.float64]
        Initial condition of shape `(2,)`.
    parameters : NDArray[np.float64]
        Parameter array passed to `mapping` and `jacobian`.
    total_time : int_t
        Total number of iterations used in the computation.
    mapping : Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]
        Function defining the time evolution of the system.
    jacobian : Callable[[NDArray[np.float64], NDArray[np.float64], Callable], NDArray[np.float64]]
        Function returning the Jacobian matrix of shape `(2, 2)`.
    sample_times : Optional[NDArray[np.integer]], optional
        Array of iteration times at which the finite-time Lyapunov exponents
        are recorded when `return_history=True`.
    return_history : bool, optional
        If True, return the finite-time exponent estimates evaluated at
        `sample_times`. Otherwise, return only the final exponent estimates.
        Default is False.
    transient_time : Optional[int_t], optional
        Number of initial iterations to discard as transient before starting
        the exponent accumulation. If None, no transient is discarded.
    log_base : numeric_t, optional
        Base of the logarithm used in the exponent calculation. Default is `np.e`.

    Returns
    -------
    Tuple[NDArray[np.float64], NDArray[np.float64]]
        - If `return_history=False`, returns:
          - a 1D array of shape `(2,)` containing the final Lyapunov exponents
          - the final state of shape `(2,)`
        - If `return_history=True`, returns:
          - a 2D array of shape `(len(sample_times), 2)` containing the
            finite-time Lyapunov exponents
          - the final state of shape `(2,)`

    Raises
    ------
    ValueError
        If `return_history=True` and any value in `sample_times` is greater
        than `total_time - transient_time`.

    Notes
    -----
    This implementation is specific to 2-dimensional maps and uses the
    Eckmann-Ruelle continuous QR-angle formulation.
    """
    neq = len(u)
    num_samples = len(sample_times) if sample_times is not None else 0
    exponents = np.zeros(neq, dtype=np.float64)
    eigvals = np.zeros(neq, dtype=np.float64)
    beta0 = 0.0
    inv_log = 1 / np.log(log_base)

    u = np.ascontiguousarray(u)

    sample_size = total_time - (transient_time if transient_time is not None else 0)
    if transient_time is not None:
        for _ in range(transient_time):
            u = mapping(u, parameters)

    history = np.empty((0, neq), dtype=np.float64)
    if return_history and sample_times is not None:
        history = np.zeros((len(sample_times), neq), dtype=np.float64)

    sample_idx = 0
    for n in range(sample_size):
        u = mapping(u, parameters)
        J = jacobian(u, parameters, mapping)

        cb0, sb0 = np.cos(beta0), np.sin(beta0)
        beta = np.arctan2(
            -J[1, 0] * cb0 + J[1, 1] * sb0,
            J[0, 0] * cb0 - J[0, 1] * sb0,
        )

        cb, sb = np.cos(beta), np.sin(beta)
        eigvals[0] = (J[0, 0] * cb - J[1, 0] * sb) * cb0 - (
            J[0, 1] * cb - J[1, 1] * sb
        ) * sb0
        eigvals[1] = (J[0, 0] * sb + J[1, 0] * cb) * sb0 + (
            J[0, 1] * sb + J[1, 1] * cb
        ) * cb0

        exponents += np.log(np.abs(eigvals))
        beta0 = beta

        if (
            return_history
            and sample_times is not None
            and sample_idx < num_samples
            and n + 1 == sample_times[sample_idx]
        ):
            history[sample_idx, :] = (exponents * inv_log) / (n + 1)
            sample_idx += 1

    if return_history:
        return history, u

    return ((exponents * inv_log) / sample_size).reshape(1, -1), u


@njit
def maximum_lyapunov_er(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: int_t,
    mapping: map_t,
    jacobian: jacobian_t,
    sample_times: Optional[NDArray[np.integer]] = None,
    return_history: bool = False,
    transient_time: Optional[int_t] = None,
    log_base: numeric_t = np.e,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute the maximum Lyapunov exponent of a 2-dimensional discrete dynamical
    system using the Eckmann-Ruelle method.

    Parameters
    ----------
    u : NDArray[np.float64]
        Initial condition of shape `(2,)`.
    parameters : NDArray[np.float64]
        Parameter array passed to `mapping` and `jacobian`.
    total_time : int_t
        Total number of iterations used in the computation.
    mapping : map_t
        Function defining the time evolution of the system.
    jacobian : jacobian_t
        Function returning the Jacobian matrix of shape `(2, 2)`.
    sample_times : Optional[NDArray[np.integer]], optional
        Array of iteration times at which the finite-time maximum Lyapunov
        exponent is recorded when `return_history=True`.
    return_history : bool, optional
        If True, return the finite-time maximum Lyapunov exponent evaluated
        at `sample_times`. Otherwise, return only the final exponent estimate.
    transient_time : Optional[int_t], optional
        Number of initial iterations to discard as transient before starting
        the exponent accumulation. If None, no transient is discarded.
    log_base : numeric_t, optional
        Base of the logarithm used in the exponent calculation. Default is `np.e`.

    Returns
    -------
    Tuple[NDArray[np.float64], NDArray[np.float64]]
        - If `return_history=False`, returns:
          - a 1D array with shape `(1,)` containing the final maximum
            Lyapunov exponent
          - the final state of shape `(2,)`
        - If `return_history=True`, returns:
          - a 1D array with shape `(len(sample_times),)` containing the
            finite-time maximum Lyapunov exponent
          - the final state of shape `(2,)`

    Notes
    -----
    This implementation is specific to 2-dimensional maps and uses the
    Eckmann-Ruelle continuous QR-angle formulation.
    """
    exponent = 0.0
    beta0 = 0.0
    eigval = 0.0
    log_den = np.log(log_base)

    u = np.ascontiguousarray(u)
    num_samples = len(sample_times) if sample_times is not None else 0

    sample_size = total_time - (transient_time if transient_time is not None else 0)
    if transient_time is not None:
        for _ in range(transient_time):
            u = mapping(u, parameters)

    history = np.empty(0, dtype=np.float64)
    if return_history and sample_times is not None:
        history = np.zeros(num_samples, dtype=np.float64)

    sample_idx = 0
    for n in range(sample_size):
        u = mapping(u, parameters)
        J = jacobian(u, parameters, mapping)

        cb0, sb0 = np.cos(beta0), np.sin(beta0)
        beta = np.arctan2(
            -J[1, 0] * cb0 + J[1, 1] * sb0,
            J[0, 0] * cb0 - J[0, 1] * sb0,
        )

        cb, sb = np.cos(beta), np.sin(beta)
        eigval = (J[0, 0] * cb - J[1, 0] * sb) * cb0 - (
            J[0, 1] * cb - J[1, 1] * sb
        ) * sb0

        exponent += np.log(np.abs(eigval))
        beta0 = beta

        if (
            return_history
            and sample_times is not None
            and sample_idx < num_samples
            and n + 1 == sample_times[sample_idx]
        ):
            history[sample_idx] = exponent / ((n + 1) * log_den)
            sample_idx += 1

    if return_history:
        return history, u

    return np.array([exponent / (sample_size * log_den)]), u


@njit
def lyapunov_qr(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: int_t,
    mapping: map_t,
    jacobian: jacobian_t,
    num_exponents: int,
    sample_times: Optional[NDArray[np.integer]] = None,
    QR: Callable[
        [NDArray[np.float64]], Tuple[NDArray[np.float64], NDArray[np.float64]]
    ] = np.linalg.qr,
    return_history: bool = False,
    transient_time: Optional[int_t] = None,
    log_base: numeric_t = np.e,
    seed: int_t = 1312,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute Lyapunov exponents of a discrete dynamical system using QR
    decomposition.

    This method evolves a set of perturbation vectors along the trajectory and
    periodically orthonormalizes them using a QR decomposition. It can be used
    for systems of arbitrary dimension.

    Parameters
    ----------
    u : NDArray[np.float64]
        Initial condition of shape `(neq,)`, where `neq` is the system dimension.
    parameters : NDArray[np.float64]
        Parameter array passed to `mapping` and `jacobian`.
    total_time : int_t
        Total number of iterations used in the computation.
    mapping : map_t
        Function defining the time evolution of the system.
    jacobian : jacobian_t
        Function returning the Jacobian matrix of shape `(neq, neq)`.
    num_exponents : int_t
        Number of Lyapunov exponents to compute.
    sample_times : Optional[NDArray[np.integer]], optional
        Array of iteration times at which the finite-time Lyapunov exponents
        are recorded when `return_history=True`.
    QR : Callable[[NDArray[np.float64]], Tuple[NDArray[np.float64], NDArray[np.float64]]], optional
        QR decomposition routine used to orthonormalize the perturbation
        vectors. Default is `qr`.
    return_history : bool, optional
        If True, return the finite-time Lyapunov exponents evaluated at
        `sample_times`. Otherwise, return only the final exponent estimates.
        Default is False.
    transient_time : Optional[int_t], optional
        Number of initial iterations to discard as transient before starting
        the exponent accumulation. If None, no transient is discarded.
    log_base : numeric_t, optional
        Base of the logarithm used in the exponent calculation. Default is `np.e`.
    seed : int_t, optional
        Seed used to initialize the random perturbation vectors. Default is 1312.

    Returns
    -------
    Tuple[NDArray[np.float64], NDArray[np.float64]]
        - If `return_history=False`, returns:
          - a 1D array of shape `(num_exponents,)` containing the final
            Lyapunov exponents
          - the final state of shape `(neq,)`
        - If `return_history=True`, returns:
          - a 2D array of shape `(len(sample_times), num_exponents)`
            containing the finite-time Lyapunov exponents
          - the final state of shape `(neq,)`

    Notes
    -----
    At each iteration, the tangent vectors are evolved by the Jacobian and then
    orthonormalized using a QR decomposition. The Lyapunov exponents are
    obtained from the accumulated logarithms of the absolute values of the
    diagonal entries of the `R` matrix.

    For `return_history=True`, the function returns the corresponding
    finite-time estimates evaluated at the requested sampling times.

    This function is compiled with Numba via `@njit`.

    References
    ----------
    [1] A. Wolf et al., "Determining Lyapunov exponents from a time series",
        Physica D 16, 285-317 (1985).
    """
    np.random.seed(seed)
    neq = len(u)
    num_samples = len(sample_times) if sample_times is not None else 0
    log_den = np.log(log_base)

    v = np.ascontiguousarray(np.random.rand(neq, num_exponents))
    v, _ = QR(v)
    exponents = np.zeros(num_exponents, dtype=np.float64)
    u = np.ascontiguousarray(u.copy())

    sample_size = total_time - (transient_time if transient_time is not None else 0)
    if transient_time is not None:
        for _ in range(transient_time):
            u = mapping(u, parameters)

    history = np.empty((0, num_exponents), dtype=np.float64)
    if return_history and sample_times is not None:
        history = np.zeros((num_samples, num_exponents), dtype=np.float64)

    sample_idx = 0
    for n in range(sample_size):
        u = np.ascontiguousarray(mapping(u, parameters))
        J = np.ascontiguousarray(jacobian(u, parameters, mapping))

        for i in range(num_exponents):
            v[:, i] = J @ np.ascontiguousarray(v[:, i])

        v, R = QR(v)
        exponents += np.log(np.abs(np.diag(R)))

        if (
            return_history
            and sample_times is not None
            and sample_idx < num_samples
            and n + 1 == sample_times[sample_idx]
        ):
            history[sample_idx, :] = exponents / ((n + 1) * log_den)
            sample_idx += 1

    if return_history:
        return history, u

    return (exponents / (sample_size * log_den)).reshape(1, -1), u


def finite_time_lyapunov(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: int_t,
    finite_time: int_t,
    mapping: map_t,
    jacobian: jacobian_t,
    num_exponents: int,
    method: str = "QR",
    transient_time: Optional[int_t] = None,
    log_base: numeric_t = np.e,
    return_points: bool = False,
) -> NDArray[np.float64] | Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute finite-time Lyapunov exponents for a discrete dynamical system.

    The trajectory is divided into consecutive non-overlapping windows of
    length `finite_time`, and Lyapunov exponents are computed independently
    in each window.

    Parameters
    ----------
    u : NDArray[np.float64]
        Initial condition of shape `(neq,)`, where `neq` is the system dimension.
    parameters : NDArray[np.float64]
        Parameter array passed to `mapping` and `jacobian`.
    total_time : int_t
        Total number of iterations used in the computation.
    finite_time : int_t
        Length of each finite-time window.
    mapping : map_t
        Function defining the time evolution of the system.
    jacobian : jacobian_t
        Function returning the Jacobian matrix of the map.
    num_exponents : int_t
        Number of Lyapunov exponents to compute in each window.
    method : str, optional
        Method used to compute the exponents:
        - `"ER"`: Eckmann-Ruelle method
        - `"QR"`: QR decomposition
        - `"QR_HH"`: Householder QR decomposition

        Default is `"QR"`.
    transient_time : Optional[int_t], optional
        Number of initial iterations to discard before starting the
        finite-time analysis. If None, no transient is discarded.
    log_base : numeric_t, optional
        Base of the logarithm used in the exponent calculation.
        Default is `np.e`.
    return_points : bool, optional
        If True, also return the phase-space point at the beginning of each
        finite-time window. Default is False.

    Returns
    -------
    NDArray[np.float64] or Tuple[NDArray[np.float64], NDArray[np.float64]]
        - If `return_points=False`, returns a 2D array of shape
          `(num_windows, num_exponents)` containing the finite-time
          Lyapunov exponents.
        - If `return_points=True`, returns:
          - `exponents`: 2D array of shape `(num_windows, num_exponents)`
          - `phase_space_points`: 2D array of shape `(num_windows, neq)`
            containing the phase-space point at the beginning of each window

        Here,
        `num_windows = (total_time - transient_time) // finite_time`.

    Raises
    ------
    ValueError
        If `finite_time` is greater than the number of available iterations
        after removing the transient, or if `method` is not `"ER"`, `"QR"`,
        or `"QR_HH"`.
        Also raised if `method="ER"` and `num_exponents` is not compatible
        with a 2-dimensional system.

    Notes
    -----
    The finite-time exponents are computed over consecutive non-overlapping
    windows, so each row of the output corresponds to one independent
    finite-time estimate along the trajectory.
    """
    method = method.upper()

    sample_size = total_time - (transient_time if transient_time is not None else 0)
    if transient_time is not None:
        for _ in range(transient_time):
            u = mapping(u, parameters)

    if finite_time > sample_size:
        raise ValueError(
            f"finite_time ({finite_time}) exceeds available samples ({sample_size})"
        )

    neq = len(u)

    if method == "ER":
        if neq != 2:
            raise ValueError("method='ER' is only valid for 2-dimensional systems")
        if num_exponents < 1 or num_exponents > 2:
            raise ValueError("For method='ER', num_exponents must be 1 or 2")
    elif method not in ("QR", "QR_HH"):
        raise ValueError("method must be 'ER', 'QR', or 'QR_HH'")

    num_windows = sample_size // finite_time
    exponents = np.zeros((num_windows, num_exponents), dtype=np.float64)
    phase_space_points = np.zeros((num_windows, neq), dtype=np.float64)

    for i in range(num_windows):
        phase_space_points[i] = u

        if method == "ER":
            if num_exponents == 1:
                window_exponents, u_new = maximum_lyapunov_er(
                    u,
                    parameters,
                    finite_time,
                    mapping,
                    jacobian,
                    log_base=log_base,
                )
            else:
                window_exponents, u_new = lyapunov_er(
                    u,
                    parameters,
                    finite_time,
                    mapping,
                    jacobian,
                    log_base=log_base,
                )
        elif method == "QR":
            window_exponents, u_new = lyapunov_qr(
                u,
                parameters,
                finite_time,
                mapping,
                jacobian,
                num_exponents,
                log_base=log_base,
            )
        else:  # method == "QR_HH"
            window_exponents, u_new = lyapunov_qr(
                u,
                parameters,
                finite_time,
                mapping,
                jacobian,
                num_exponents,
                QR=householder_qr,
                log_base=log_base,
            )

        exponents[i] = window_exponents.ravel()
        u = u_new.copy()

    if return_points:
        return exponents, phase_space_points
    return exponents
