# dynamical_indicators.py

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

from typing import Callable, Optional, Tuple, Union

import numpy as np
from numba import njit
from numpy.typing import NDArray

from pynamicalsys.common.recurrence_quantification_analysis import (
    RTEConfig,
    recurrence_matrix,
    white_vertline_distr,
)
from pynamicalsys.common.time_series_metrics import hurst_exponent
from pynamicalsys.common.utils import householder_qr, qr, wedge_norm
from pynamicalsys.discrete_time.trajectory_analysis import (
    generate_trajectory,
    iterate_mapping,
)


@njit
def lyapunov_1D(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: int,
    mapping: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    derivative_mapping: Callable[
        [NDArray[np.float64], NDArray[np.float64], Callable], NDArray[np.float64]
    ],
    num_exponents: int,  # Added just to match signature
    sample_times: Union[NDArray[np.int32], NDArray[np.int64]],
    return_history: bool = False,
    transient_time: Optional[int] = None,
    log_base: float = np.e,
) -> Union[NDArray[np.float64], float]:
    """
    Compute the Lyapunov exponent for a 1-dimensional dynamical system.

    The Lyapunov exponent characterizes the rate of separation of infinitesimally close
    trajectories, serving as a measure of chaos (λ > 0 indicates chaos).

    Parameters
    ----------
    u : NDArray[np.float64]
        Initial state vector (shape: `(1,)` for 1D systems).
    parameters : NDArray[np.float64]
        System parameters passed to `mapping` and `derivative_mapping`.
    total_time : int
        Total number of iterations (time steps) to compute.
    mapping : Callable[[NDArray, NDArray], NDArray]
        Function defining the system's evolution: `u_next = mapping(u, parameters)`.
    derivative_mapping : Callable[[NDArray, NDArray, Callable], NDArray]
        Function returning the derivative of `mapping` (Jacobian for 1D systems).
    sample_times: Union[NDArray[np.int32], NDArray[np.int64]],
        Specific time steps to record the exponent (if `return_history=True`).
    return_history : bool, optional
        If True, returns the Lyapunov exponent estimate at each step (default: False).
    transient_time : Optional[int], optional
        Number of initial iterations to discard as transient (default: None).
    log_base : float, optional
        Logarithm base for exponent calculation (default: e).

    Returns
    -------
    Union[NDArray[np.float64], float]
        - If `return_history=False`: Final Lyapunov exponent (scalar).
        - If `return_history=True`: Array of exponent estimates over time.

    Notes
    -----
    - The Lyapunov exponent (λ) is computed as:
        λ = (1/N) Σ log|f'(u_i)|, where N = `total_time - transient_time`.
    - For 1D systems, `derivative_mapping` should return a 1x1 Jacobian (scalar value).
    - Uses Numba (`@njit`) for accelerated computation.
    """

    # Handle transient time
    if transient_time is not None:
        sample_size = total_time - transient_time
        for _ in range(transient_time):
            u = mapping(u, parameters)
    else:
        sample_size = total_time

    # Initialize history tracking
    if return_history:
        if sample_times.max() > sample_size:
            raise ValueError("sample_times must be ≤ total_time - transient_time")
        history = np.zeros(len(sample_times))

    sample_idx = 0
    exponent = 0.0
    prev_i = 0
    for st in sample_times:
        steps = st - prev_i
        for _ in range(steps):
            u = mapping(u, parameters)
            du = derivative_mapping(u, parameters, mapping)
            exponent += np.log(np.abs(du[0, 0])) / np.log(log_base)

        if return_history:
            history[sample_idx] = exponent / st
            sample_idx += 1
        prev_i = st

    return history if return_history else np.array([exponent / sample_size])


@njit
def lyapunov_er(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: int,
    mapping: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    jacobian: Callable[
        [NDArray[np.float64], NDArray[np.float64], Callable], NDArray[np.float64]
    ],
    num_exponents: int,  # Added just to match signature
    sample_times: Union[NDArray[np.int32], NDArray[np.int64]],
    return_history: bool = False,
    transient_time: Optional[int] = None,
    log_base: float = np.e,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute Lyapunov exponents using the Eckmann-Ruelle (ER) method for 2D systems.

    This method tracks the evolution of perturbations via continuous QR decomposition
    using rotational angles, providing numerically stable exponent estimates.

    Parameters
    ----------
    u : NDArray[np.float64]
        Initial state vector (shape: `(2,)` for 2D systems).
    parameters : NDArray[np.float64]
        System parameters passed to `mapping` and `jacobian`.
    total_time : int
        Total number of iterations (time steps) to compute.
    mapping : Callable[[NDArray, NDArray], NDArray]
        System evolution function: `u_next = mapping(u, parameters)`.
    jacobian : Callable[[NDArray, NDArray, Callable], NDArray]
        Function returning the Jacobian matrix (shape: `(2, 2)`).
    sample_times: Union[NDArray[np.int32], NDArray[np.int64]],
        Specific time steps to record exponents (if `return_history=True`).
    return_history : bool, optional
        If True, returns exponent convergence history (default: False).
    transient_time : Optional[int], optional
        Number of initial iterations to discard as transient (default: None).
    log_base : float, optional
        Logarithm base for exponent calculation (default: e).

    Returns
    -------
    Tuple[NDArray[np.float64], NDArray[np.float64]]
        - If `return_history=True`:
            - `history`: Array of exponent estimates (shape: `(sample_size, 2)` or `(len(sample_times), 2)`)
            - `final_state`: System state at termination (shape: `(2,)`)
        - If `return_history=False`:
            - `exponents`: Final Lyapunov exponents (shape: `(2, 1)`)
            - `final_state`: System state at termination (shape: `(2,)`)

    Notes
    -----
    - **Method**: Uses rotation angles for continuous QR decomposition [1].
    - **Stability**: More robust than Gram-Schmidt for 2D systems.
    - **Limitation**: Designed specifically for 2D maps (`neq=2`).
    - **Numerics**: Exponents are averaged as:
        λ_i = (1/N) Σ log|T_ii|, where T is the transformation matrix.

    References
    ----------
    [1] J. Eckmann & D. Ruelle, "Ergodic theory of chaos and strange attractors",
        Rev. Mod. Phys. 57, 617 (1985).
    """

    neq = len(u)
    exponents = np.zeros(neq)
    beta0 = 0.0  # Initial rotation angle
    u_contig = np.ascontiguousarray(u)

    # Handle transient time
    if transient_time is not None:
        sample_size = total_time - transient_time
        for _ in range(transient_time):
            u_contig = mapping(u_contig, parameters)
    else:
        sample_size = total_time

    # Initialize history tracking
    if return_history:
        if sample_times.max() > sample_size:
            raise ValueError("sample_times must be ≤ total_time - transient_time")
        history = np.zeros((len(sample_times), neq))

    sample_idx = 0
    eigvals = np.zeros(neq)
    log_base_inv = 1.0 / np.log(log_base)
    prev_i = 0
    for st in sample_times:
        steps = st - prev_i
        for _ in range(steps):
            u_contig = mapping(u_contig, parameters)
            J = jacobian(u_contig, parameters, mapping)

            cb0, sb0 = np.cos(beta0), np.sin(beta0)
            beta = np.arctan2(
                -J[1, 0] * cb0 + J[1, 1] * sb0, J[0, 0] * cb0 - J[0, 1] * sb0
            )

            cb, sb = np.cos(beta), np.sin(beta)
            eigvals[0] = (J[0, 0] * cb - J[1, 0] * sb) * cb0 - (
                J[0, 1] * cb - J[1, 1] * sb
            ) * sb0
            eigvals[1] = (J[0, 0] * sb + J[1, 0] * cb) * sb0 + (
                J[0, 1] * sb + J[1, 1] * cb
            ) * cb0

            exponents += np.log(np.abs(eigvals)) * log_base_inv

            beta0 = beta

        if return_history:
            history[sample_idx] = exponents / st
            sample_idx += 1
        prev_i = st

    # Format output
    if return_history:
        return history, u_contig
    else:
        aux_exponents = np.zeros((neq, 1))
        aux_exponents[:, 0] = exponents / sample_size
        return aux_exponents, u_contig


@njit
def maximum_lyapunov_er(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: int,
    mapping: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    jacobian: Callable[
        [NDArray[np.float64], NDArray[np.float64], Callable], NDArray[np.float64]
    ],
    num_exponents: int,  # Added just to match signature
    sample_times: Union[NDArray[np.int32], NDArray[np.int64]],
    return_history: bool = False,
    transient_time: Optional[int] = None,
    log_base: float = np.e,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute the maximum Lyapunov exponent using the Eckmann-Ruelle (ER) method for 2D systems.

    This method tracks the evolution of perturbations via continuous QR decomposition
    using rotational angles, providing numerically stable exponent estimates.

    Parameters
    ----------
    u : NDArray[np.float64]
        Initial state vector (shape: `(2,)` for 2D systems).
    parameters : NDArray[np.float64]
        System parameters passed to `mapping` and `jacobian`.
    total_time : int
        Total number of iterations (time steps) to compute.
    mapping : Callable[[NDArray, NDArray], NDArray]
        System evolution function: `u_next = mapping(u, parameters)`.
    jacobian : Callable[[NDArray, NDArray, Callable], NDArray]
        Function returning the Jacobian matrix (shape: `(2, 2)`).
    sample_times: Union[NDArray[np.int32], NDArray[np.int64]],
        Specific time steps to record exponents (if `return_history=True`).
    return_history : bool, optional
        If True, returns exponent convergence history (default: False).
    transient_time : Optional[int], optional
        Number of initial iterations to discard as transient (default: None).
    log_base : float, optional
        Logarithm base for exponent calculation (default: e).

    Returns
    -------
    Tuple[NDArray[np.float64], NDArray[np.float64]]
        - If `return_history=True`:
            - `history`: Array of exponent estimates (shape: `(sample_size, 2)` or `(len(sample_times), 2)`)
            - `final_state`: System state at termination (shape: `(2,)`)
        - If `return_history=False`:
            - `exponents`: Final Lyapunov exponents (shape: `(2, 1)`)
            - `final_state`: System state at termination (shape: `(2,)`)

    Notes
    -----
    - **Method**: Uses rotation angles for continuous QR decomposition [1].
    - **Stability**: More robust than Gram-Schmidt for 2D systems.
    - **Limitation**: Designed specifically for 2D maps (`neq=2`).
    - **Numerics**: Exponents are averaged as:
        λ_i = (1/N) Σ log|T_ii|, where T is the transformation matrix.

    References
    ----------
    [1] J. Eckmann & D. Ruelle, "Ergodic theory of chaos and strange attractors",
        Rev. Mod. Phys. 57, 617 (1985).
    """

    neq = len(u)
    exponent = 0.0
    beta0 = 0.0  # Initial rotation angle
    u_contig = np.ascontiguousarray(u)

    # Handle transient time
    if transient_time is not None:
        sample_size = total_time - transient_time
        for _ in range(transient_time):
            u_contig = mapping(u_contig, parameters)
    else:
        sample_size = total_time

    # Initialize history tracking
    if return_history:
        if sample_times.max() > sample_size:
            raise ValueError("sample_times must be ≤ total_time - transient_time")
        history = np.zeros(len(sample_times))

    sample_idx = 0
    eigval = 0.0
    log_base_inv = 1.0 / np.log(log_base)
    prev_i = 0
    for st in sample_times:
        steps = st - prev_i
        for _ in range(steps):
            u_contig = mapping(u_contig, parameters)
            J = jacobian(u_contig, parameters, mapping)

            cb0, sb0 = np.cos(beta0), np.sin(beta0)
            beta = np.arctan2(
                -J[1, 0] * cb0 + J[1, 1] * sb0, J[0, 0] * cb0 - J[0, 1] * sb0
            )

            cb, sb = np.cos(beta), np.sin(beta)
            eigval = (J[0, 0] * cb - J[1, 0] * sb) * cb0 - (
                J[0, 1] * cb - J[1, 1] * sb
            ) * sb0

            exponent += np.log(np.abs(eigval)) * log_base_inv

            beta0 = beta

        if return_history:
            history[sample_idx] = exponent / st
            sample_idx += 1
        prev_i = st

    # Format output
    if return_history:
        return history, u_contig
    else:
        return np.array([exponent / sample_size]), u_contig


@njit
def lyapunov_qr(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: int,
    mapping: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    jacobian: Callable[
        [NDArray[np.float64], NDArray[np.float64], Callable], NDArray[np.float64]
    ],
    num_exponents: int,
    sample_times: Union[NDArray[np.int32], NDArray[np.int64]],
    QR: Callable[
        [NDArray[np.float64]], Tuple[NDArray[np.float64], NDArray[np.float64]]
    ] = qr,
    return_history: bool = False,
    transient_time: Optional[int] = None,
    log_base: float = np.e,
    seed: int = 13,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute Lyapunov exponents using QR decomposition (Gram-Schmidt) for N-dimensional systems.

    This method tracks the evolution of perturbation vectors with periodic orthogonalization
    via QR decomposition, suitable for systems of arbitrary dimension.

    Parameters
    ----------
    u : NDArray[np.float64]
        Initial state vector (shape: `(neq,)`).
    parameters : NDArray[np.float64]
        System parameters passed to `mapping` and `jacobian`.
    total_time : int
        Total number of iterations (time steps) to compute.
    mapping : Callable[[NDArray, NDArray], NDArray]
        System evolution function: `u_next = mapping(u, parameters)`.
    jacobian : Callable[[NDArray, NDArray, Callable], NDArray]
        Function returning the Jacobian matrix (shape: `(neq, neq)`).
    QR : Callable[[NDArray], Tuple[NDArray, NDArray]], optional
        QR decomposition function (default: `numpy.linalg.qr`).
    return_history : bool, optional
        If True, returns exponent convergence history (default: False).
    sample_times : Optional[NDArray[np.int32]], optional
        Specific time steps to record exponents (if `return_history=True`).
    transient_time : Optional[int], optional
        Number of initial iterations to discard as transient (default: None).
    log_base : float, optional
        Logarithm base for exponent calculation (default: e).

    Returns
    -------
    Tuple[NDArray[np.float64], NDArray[np.float64]]
        - If `return_history=True`:
            - `history`: Array of exponent estimates (shape: `(sample_size, neq)` or `(len(sample_times), neq)`)
            - `final_state`: System state at termination (shape: `(neq,)`)
        - If `return_history=False`:
            - `exponents`: Final Lyapunov exponents (shape: `(neq, 1)`)
            - `final_state`: System state at termination (shape: `(neq,)`)

    Notes
    -----
    - **Method**: Uses QR decomposition for orthogonalization [1].
    - **Dimensionality**: Works for systems of any dimension (`neq ≥ 1`).
    - **Numerics**:
        - Exponents computed as: λ_i = (1/N) Σ log|R_ii|, where R is from QR decomposition.
    - **Performance**: Optimized with Numba's `@njit`.

    References
    ----------
    [1] A. Wolf et al., "Determining Lyapunov exponents from a time series",
        Physica D 16D, 285-317 (1985).
    """

    np.random.seed(seed)
    neq = len(u)
    v = np.ascontiguousarray(np.random.rand(neq, num_exponents))
    v, _ = qr(v)  # Initialize orthonormal vectors
    exponents = np.zeros(num_exponents)
    u_contig = np.ascontiguousarray(u.copy())

    # Handle transient time
    if transient_time is not None:
        sample_size = total_time - transient_time
        for _ in range(transient_time):
            u_contig = mapping(u_contig, parameters)
    else:
        sample_size = total_time

    # Initialize history tracking
    if return_history:
        if sample_times.max() > sample_size:
            raise ValueError("sample_times must be ≤ total_time - transient_time")
        history = np.zeros((len(sample_times), num_exponents))

    sample_idx = 0
    log_base_inv = 1.0 / np.log(log_base)
    prev_i = 0
    for st in sample_times:
        steps = st - prev_i
        for _ in range(steps):
            u_contig = mapping(u_contig, parameters)
            J = np.ascontiguousarray(jacobian(u_contig, parameters, mapping))
            # Evolve and orthogonalize vectors
            for i in range(num_exponents):
                v[:, i] = np.ascontiguousarray(J) @ np.ascontiguousarray(v[:, i])
            v, R = QR(v)
            exponents += np.log(np.abs(np.diag(R))) * log_base_inv

        if return_history:
            history[sample_idx] = exponents / st
            sample_idx += 1
        prev_i = st

    # Format output
    if return_history:
        return history, u_contig
    else:
        aux_exponents = np.zeros((num_exponents, 1))
        aux_exponents[:, 0] = exponents / sample_size
        return aux_exponents, u_contig


def finite_time_lyapunov(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: int,
    finite_time: int,
    mapping: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    jacobian: Callable[
        [NDArray[np.float64], NDArray[np.float64], Callable], NDArray[np.float64]
    ],
    num_exponents: int,
    method: str = "QR",
    transient_time: Optional[int] = None,
    log_base: float = np.e,
    return_points: bool = False,
) -> Union[NDArray[np.float64], Tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """
    Compute finite-time Lyapunov exponents (FTLEs) for a dynamical system.

    FTLEs reveal how chaotic behavior varies over different time scales by computing
    Lyapunov exponents over sliding windows. Supports both Eckmann-Ruelle (ER) and
    QR-based methods (Gram-Schmidt or Householder).

    Parameters
    ----------
    u : NDArray[np.float64]
        Initial state vector (shape: `(neq,)`).
    parameters : NDArray[np.float64]
        System parameters passed to `mapping` and `jacobian`.
    total_time : int
        Total number of iterations to simulate.
    finite_time : int
        Length of each analysis window (iterations).
    mapping : Callable[[NDArray, NDArray], NDArray]
        System evolution function: `u_next = mapping(u, parameters)`.
    jacobian : Callable[[NDArray, NDArray, Callable], NDArray]
        Function returning the Jacobian matrix (shape: `(neq, neq)`).
    method : str, optional
        Computation method: 'ER' (2D only), 'QR' (Gram-Schmidt), or 'QR_HH' (Householder)
        (default: 'ER').
    transient_time : Optional[int], optional
        Initial iterations to discard (default: None).
    log_base : float, optional
        Logarithm base for exponent calculation (default: e).

    Returns
    -------
    NDArray[np.float64]
        Array of FTLEs (shape: `(num_windows, neq)`), where:
        `num_windows = floor((total_time - transient_time) / finite_time)`

    Raises
    ------
    ValueError
        - If `method` is invalid

    Notes
    -----
    - **Window Processing**: Total time is divided into non-overlapping windows.
    - **Method Selection**:
        - 'QR': General N-dimensional (Gram-Schmidt orthogonalization)
        - 'QR_HH': More stable for ill-conditioned systems (Householder QR)
    - **Numerics**: Each window's exponents are independent estimates.
    """
    # Handle transient
    if transient_time is not None:
        sample_size = total_time - transient_time
        for _ in range(transient_time):
            u = mapping(u, parameters)
    else:
        sample_size = total_time

    # Validate window size
    if finite_time > sample_size:
        raise ValueError(
            f"finite_time ({finite_time}) exceeds available samples ({sample_size})"
        )

    neq = len(u)
    num_windows = sample_size // finite_time
    exponents = np.zeros((num_windows, num_exponents))
    phase_space_points = np.zeros((num_windows, neq))
    sample_times = np.arange(finite_time)
    # Compute exponents for each window
    for i in range(num_windows):
        if num_exponents == 1 and method == "ER":
            window_exponents, u_new = maximum_lyapunov_er(
                u,
                parameters,
                finite_time,
                mapping,
                jacobian,
                num_exponents,
                sample_times,
                log_base=log_base,
            )
        elif num_exponents > 1 and method == "ER":
            window_exponents, u_new = lyapunov_er(
                u,
                parameters,
                finite_time,
                mapping,
                jacobian,
                num_exponents,
                sample_times,
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
                sample_times,
                log_base=log_base,
            )
        elif method == "QR_HH":
            window_exponents, u_new = lyapunov_qr(
                u,
                parameters,
                finite_time,
                mapping,
                jacobian,
                num_exponents,
                sample_times,
                QR=householder_qr,
                log_base=log_base,
            )
        else:
            raise ValueError("method must be 'ER', 'QR', or 'QR_HH'")

        exponents[i] = window_exponents.flatten()
        phase_space_points[i] = u
        u = u_new.copy()

    if return_points:
        return exponents, phase_space_points
    else:
        return exponents


def dig(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: int,
    mapping: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    transient_time: Optional[int] = None,
) -> float:
    """Compute the Dynamic Indicator for Globalness (DIG) of a trajectory.

    Parameters
    ----------
    u : NDArray[np.float64]
        Initial condition of shape (d,)
    parameters : NDArray[np.float64]
        System parameters
    total_time : int
        Total number of iterations (must be even and >= 100)
    mapping : Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]
        System mapping function (must be Numba-compatible)
    func : Callable[[NDArray[np.float64]], NDArray[np.float64]]
        Observable function
    transient_time : Optional[int]
        Burn-in period to discard

    Returns
    -------
    float
        DIG value (higher values indicate better convergence)
        Returns 16 if perfect convergence detected

    Notes
    -----
    - Implements the weighted Birkhoff average method
    - Requires total_time to be even (split into two halves)
    - For reliable results, total_time should be >= 1000
    """

    u = u.copy()

    # Handle transient
    if transient_time is not None:
        if transient_time >= total_time:
            raise ValueError("transient_time must be < total_time")
        u = iterate_mapping(u, parameters, transient_time, mapping)
        sample_size = total_time - transient_time
    else:
        sample_size = total_time

    N = sample_size // 2
    if N < 2:
        raise ValueError("Effective sample size too small after transient removal")

    N = sample_size // 2

    t = np.arange(1, N) / N
    S = np.exp(-1 / (t * (1 - t))).sum()
    w = np.exp(-1 / (t * (1 - t))) / S

    # Weighted Birkhoff average for the first half of iterations
    time_series = generate_trajectory(u, parameters, N, mapping)
    WB0 = (w * func(time_series[:-1, :])).sum()

    # Weighted Birkhoff average for the second half of iterations
    u = time_series[-1, :]
    time_series = generate_trajectory(u, parameters, N, mapping)
    WB1 = (w * func(time_series[:-1, :])).sum()

    return -np.log10(abs(WB0 - WB1))


@njit
def SALI(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: int,
    mapping: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    jacobian: Callable[
        [NDArray[np.float64], NDArray[np.float64], Callable], NDArray[np.float64]
    ],
    sample_times: Union[NDArray[np.int32], NDArray[np.int64]],
    return_history: bool = False,
    tol: float = 1e-16,
    transient_time: Optional[int] = None,
    seed: int = 13,
) -> Union[NDArray[np.float64], Tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """
    Compute the Smallest Alignment Index (SALI) for a dynamical system.

    SALI quantifies chaos by tracking the alignment of deviation vectors in tangent space.
    For regular motion, SALI oscillates near 1; for chaotic motion, it decays exponentially.

    Parameters
    ----------
    u : NDArray[np.float64]
        Initial state vector of the system (shape: `(neq,)`).
    parameters : NDArray[np.float64]
        System parameters (shape: arbitrary, passed to `mapping` and `jacobian`).
    total_time : int
        Total number of iterations (time steps) to simulate.
    mapping : Callable[[NDArray, NDArray], NDArray]
        Function representing the system's time evolution: `u_next = mapping(u, parameters)`.
    jacobian : Callable[[NDArray, NDArray, Callable], NDArray]
        Function computing the Jacobian matrix of `mapping` at state `u`.
    sample_times: Union[NDArray[np.int32], NDArray[np.int64]],
        Specific time steps at which to record SALI (if `return_history=True`). Must be sorted.
    return_history : bool, optional
        If True, return SALI values at each time step (or `sample_times`). Default: False.
    tol : float, optional
        Tolerance for early stopping if SALI < `tol` (default: 1e-16).
    transient_time : Optional[int], optional
        Number of initial iterations to discard as transient (default: None).
    seed : int, optional
        Random seed for reproducibility (default: 13)

    Returns
    -------
    Union[NDArray[np.float64], Tuple[NDArray[np.float64], NDArray[np.float64]]]
        - If `return_history=False`: Final SALI value (shape: `(1,)`).
        - If `return_history=True`: Array of SALI values at sampled times.

    Raises
    ------
    ValueError
        If `sample_times` contains values exceeding `total_time`.

    Notes
    -----
    - Uses QR decomposition to initialize orthonormal deviation vectors.
    - Computes both Parallel (PAI) and Antiparallel (AAI) Alignment Indices.
    - Early termination occurs if SALI < `tol` (indicating chaotic behavior).
    - Optimized with `@njit` for performance.
    """

    np.random.seed(seed)  # For reproducibility

    neq = len(u)

    # Only need 2 vectors for SALI
    v = np.ascontiguousarray(np.random.rand(neq, 2))
    v, _ = qr(v)

    # Handle transient time
    if transient_time is not None:
        sample_size = total_time - transient_time
        for _ in range(transient_time):
            u = mapping(u, parameters)
    else:
        sample_size = total_time

    # Initialize history tracking
    if return_history:
        if sample_times.max() > sample_size:
            raise ValueError("sample_times must be ≤ total_time - transient_time")
        history = np.zeros(len(sample_times))

    sample_idx = 0
    prev_i = 0
    for st in sample_times:
        steps = st - prev_i
        for _ in range(steps):
            u = mapping(u, parameters)
            J = np.ascontiguousarray(jacobian(u, parameters, mapping))

            for i in range(2):
                v[:, i] = np.ascontiguousarray(J) @ np.ascontiguousarray(v[:, i])
                v[:, i] /= np.linalg.norm(v[:, i])

            # Compute SALI
            PAI = np.linalg.norm(v[:, 0] + v[:, 1])
            AAI = np.linalg.norm(v[:, 0] - v[:, 1])
            sali_val = min(PAI, AAI)

        if return_history:
            history[sample_idx] = sali_val
            sample_idx += 1
        prev_i = st

        if sali_val < tol:
            break

    return history if return_history else np.array([sali_val])


def LDI_k(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: int,
    mapping: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    jacobian: Callable[
        [NDArray[np.float64], NDArray[np.float64], Callable], NDArray[np.float64]
    ],
    k: int,
    sample_times: Union[NDArray[np.int32], NDArray[np.int64]],
    return_history: bool = False,
    tol: float = 1e-16,
    transient_time: Optional[int] = None,
    seed: int = 13,
) -> Union[NDArray[np.float64], Tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """
    Compute the linear dependence index (LDI) for a dynamical system.

    LDI is a measure of chaos in dynamical systems, calculated using the evolution
    of `k` initially orthonormal deviation vectors under the system's Jacobian.

    Parameters
    ----------
    u : NDArray[np.float64]
        Initial state vector of the system (shape: `(neq,)`).
    parameters : NDArray[np.float64]
        System parameters (shape: arbitrary, passed to `mapping` and `jacobian`).
    total_time : int
        Total number of iterations (time steps) to simulate.
    mapping : Callable[[NDArray, NDArray], NDArray]
        Function representing the system's time evolution (maps state `u` to next state).
    jacobian : Callable[[NDArray, NDArray, Callable], NDArray]
        Function computing the Jacobian matrix of `mapping` at state `u`.
    k : int
        Number of deviation vectors to track.
    sample_times: Union[NDArray[np.int32], NDArray[np.int64]],
        Specific time steps at which to record LDI (if `return_history=True`).
    return_history : bool, optional
        If True, return GALI values at each time step (or `sample_times`). Default: False.
    tol : float, optional
        Tolerance for early stopping if GALI drops below this value (default: 1e-16).
    transient_time : Optional[int], optional
        Number of initial iterations to discard as transient (default: None).
    seed : int, optional
        Random seed for reproducibility (default 13)

    Returns
    -------
    Union[NDArray[np.float64], Tuple[NDArray[np.float64], NDArray[np.float64]]]
        - If `return_history=False`: Final LDI value (shape: `(1,)`).
        - If `return_history=True`: Array of LDI values at each sampled time.

    Raises
    ------
    ValueError
        If `sample_times` contains values exceeding `total_time`.

    Notes
    -----
    - Early termination occurs if LDI < `tol` (indicating chaotic behavior).
    """

    np.random.seed(seed)  # For reproducibility

    neq = len(u)

    # Generate random orthonormal deviation vectors
    v = np.ascontiguousarray(np.random.rand(neq, k))
    v, _ = qr(v)

    if transient_time is not None:
        # Discard transient time
        sample_size = total_time - transient_time
        for i in range(transient_time):
            u = mapping(u, parameters)
    else:
        sample_size = total_time

    # Initialize history tracking
    if return_history:
        if sample_times.max() > sample_size:
            raise ValueError("sample_times must be ≤ total_time - transient_time")
        history = np.zeros(len(sample_times))

    sample_idx = 0
    prev_j = 0
    for st in sample_times:
        steps = st - prev_j
        for _ in range(steps):
            u = mapping(u, parameters)
            J = np.ascontiguousarray(jacobian(u, parameters, mapping))

            # Update deviation vectors
            for i in range(k):
                v[:, i] = np.ascontiguousarray(J) @ np.ascontiguousarray(v[:, i])
                v[:, i] = v[:, i] / np.linalg.norm(v[:, i])

            # Compute LDI
            S = np.linalg.svd(v, full_matrices=False, compute_uv=False)
            # ldi = np.prod(S)  # LDI is the product of the singular values
            ldi = np.exp(np.sum(np.log(S)))  # LDI is the product of all singular values
            # Instead of computing prod(S) directly, which could lead to underflows
            # or overflows, we compute the sum_{i=1}^k log(S_i) and then take the
            # exponential of this sum.

        if return_history:
            history[sample_idx] = ldi
            sample_idx += 1
        prev_j = st

        if ldi < tol:
            break

    if return_history:
        return history
    else:
        return np.array([ldi])


def GALI_k(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: int,
    mapping: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    jacobian: Callable[
        [NDArray[np.float64], NDArray[np.float64], Callable], NDArray[np.float64]
    ],
    k: int,
    sample_times: Union[NDArray[np.int32], NDArray[np.int64]],
    return_history: bool = False,
    tol: float = 1e-16,
    transient_time: Optional[int] = None,
    seed: int = 13,
) -> Union[NDArray[np.float64], Tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """
    Compute the Generalized Aligment Index (GALI) for a dynamical system.

    GALI is a measure of chaos in dynamical systems, calculated using the evolution
    of `k` initially orthonormal deviation vectors under the system's Jacobian.

    Parameters
    ----------
    u : NDArray[np.float64]
        Initial state vector of the system (shape: `(neq,)`).
    parameters : NDArray[np.float64]
        System parameters (shape: arbitrary, passed to `mapping` and `jacobian`).
    total_time : int
        Total number of iterations (time steps) to simulate.
    mapping : Callable[[NDArray, NDArray], NDArray]
        Function representing the system's time evolution (maps state `u` to next state).
    jacobian : Callable[[NDArray, NDArray, Callable], NDArray]
        Function computing the Jacobian matrix of `mapping` at state `u`.
    k : int
        Number of deviation vectors to track.
    sample_times: Union[NDArray[np.int32], NDArray[np.int64]],
        Specific time steps at which to record LDI (if `return_history=True`).
    return_history : bool, optional
        If True, return GALI values at each time step (or `sample_times`). Default: False.
    tol : float, optional
        Tolerance for early stopping if GALI drops below this value (default: 1e-16).
    transient_time : Optional[int], optional
        Number of initial iterations to discard as transient (default: None).
    seed : int, optional
        Random seed for reproducibility (default 13)

    Returns
    -------
    Union[NDArray[np.float64], Tuple[NDArray[np.float64], NDArray[np.float64]]]
        - If `return_history=False`: Final GALI value (shape: `(1,)`).
        - If `return_history=True`: Array of GALI values at each sampled time.

    Raises
    ------
    ValueError
        If `sample_times` contains values exceeding `total_time`.

    Notes
    -----
    - Early termination occurs if GALI < `tol` (indicating chaotic behavior).
    """

    np.random.seed(seed)  # For reproducibility

    neq = len(u)

    # Generate random orthonormal deviation vectors
    v = np.ascontiguousarray(np.random.rand(neq, k))
    v, _ = qr(v)

    if transient_time is not None:
        # Discard transient time
        sample_size = total_time - transient_time
        for i in range(transient_time):
            u = mapping(u, parameters)
    else:
        sample_size = total_time

    # Initialize history tracking
    if return_history:
        if sample_times.max() > sample_size:
            raise ValueError("sample_times must be ≤ total_time - transient_time")
        history = np.zeros(len(sample_times))

    sample_idx = 0
    prev_j = 0
    for st in sample_times:
        steps = st - prev_j
        for _ in range(steps):
            u = mapping(u, parameters)
            J = np.ascontiguousarray(jacobian(u, parameters, mapping))

            # Update deviation vectors
            for i in range(k):
                v[:, i] = np.ascontiguousarray(J) @ np.ascontiguousarray(v[:, i])
                v[:, i] = v[:, i] / np.linalg.norm(v[:, i])

            # Compute GALI
            gali = wedge_norm(v)

        if return_history:
            history[sample_idx] = gali
            sample_idx += 1
        prev_j = st

        if gali < tol:
            break

    if return_history:
        return history
    else:
        return np.array([gali])


def hurst_exponent_wrapped(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: int,
    mapping: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    wmin: int = 2,
    transient_time: Optional[int] = None,
    return_last: bool = False,
) -> NDArray[np.float64]:
    u = u.copy()
    neq = len(u)
    H = np.zeros(neq)

    time_series = generate_trajectory(
        u, parameters, total_time, mapping, transient_time=transient_time
    )

    H = hurst_exponent(time_series, wmin=wmin)

    if return_last:
        result = np.zeros(2 * neq)
        result[:neq] = H
        result[neq:] = time_series[-1, :]
        return result
    else:
        return H


def finite_time_hurst_exponent(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: int,
    finite_time: int,
    mapping: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    wmin: int = 2,
    return_points: bool = False,
) -> Union[NDArray[np.float64], Tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """
    Compute finite-time Hurst exponents for a dynamical system.

    Parameters
    ----------
    u : NDArray[np.float64]
        Initial condition vector of shape (n,).
    parameters : NDArray[np.float64]
        Parameters passed to the mapping function.
    total_time : int
        Total number of iterations used to generate the trajectory.
    finite_time : int
        Length of each analysis window (iterations).
    mapping : Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]
        A function that defines the system dynamics, i.e., how `u` evolves over time given `parameters`.
    wmin : int, optional
        Minimum window size for the rescaled range calculation. Default is 2.

    Returns
    -------
    NDArray[np.float64]
        Array of estimated Hurst exponents for each window.

    Notes
    -----
    The function computes the Hurst exponent for non-overlapping windows of size `finite_time`.
    """

    u = u.copy()

    num_windows = total_time // finite_time
    H_values = np.zeros((num_windows, len(u)))
    phase_space_points = np.zeros((num_windows, len(u)))

    # Compute Hurst exponent for each window
    for i in range(num_windows):
        time_series = generate_trajectory(u, parameters, finite_time, mapping)
        H_values[i] = hurst_exponent(time_series, wmin=wmin)
        phase_space_points[i] = time_series[-1, :]
        u = time_series[-1, :]

    if return_points:
        return H_values, phase_space_points
    else:
        return H_values


@njit
def lyapunov_vectors():
    # ! To be implemented...
    pass


@njit
def lagrangian_descriptors(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: int,
    mapping: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    backwards_mapping: Callable[
        [NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]
    ],
    mod: float = 1.0,
    transient_time: Optional[int] = None,
) -> NDArray[np.float64]:
    """Compute Lagrangian Descriptors (LDs) for a dynamical system.

    Parameters
    ----------
    u : NDArray[np.float64]
        Initial condition of shape (d,), where d is system dimension
    parameters : NDArray[np.float64]
        System parameters of shape (p,)
    total_time : int
        Total number of iterations (must be > 0)
    mapping : Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]
        Forward mapping function: u_{n+1} = mapping(u_n, parameters)
    backwards_mapping : Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]
        Backward mapping function: u_{n-1} = backwards_mapping(u_n, parameters)
    transient_time : Optional[int], optional
        Number of initial iterations to discard (default None)

    Returns
    -------
    NDArray[np.float64]
        Array of shape (2,) containing:
        - LDs[0]: Forward LD (sum of forward trajectory distances)
        - LDs[1]: Backward LD (sum of backward trajectory distances)

    Notes
    -----
    - LDs reveal phase space structures and invariant manifolds
    - Higher values indicate more "stretching" in phase space
    - For best results:
      - Use total_time >> 1 (typically 1000-10000)
      - Ensure mapping and backwards_mapping are exact inverses
    - Numba-optimized for performance

    Examples
    --------
    >>> # Basic usage
    >>> u0 = np.array([0.1, 0.2])
    >>> params = np.array([0.5, 1.0])
    >>> lds = lagrangian_descriptors(u0, params, 1000, fwd_map, bwd_map)
    >>> forward_ld, backward_ld = lds
    """
    # Initialize descriptors
    LDs = np.zeros(2)
    u_forward = u.copy()
    u_backward = u.copy()

    # Handle transient period
    if transient_time is not None:
        if transient_time >= total_time:
            return LDs  # Return zeros if no sample time remains

        # Evolve through transient
        for _ in range(transient_time):
            u_forward = mapping(u_forward, parameters)
            u_backward = backwards_mapping(u_backward, parameters)
        sample_size = total_time - transient_time
    else:
        sample_size = total_time

    # Main computation loop
    for _ in range(sample_size):
        # Forward evolution
        u_new_forward = mapping(u_forward, parameters)
        dx = abs(u_new_forward[0] - u_forward[0])
        if dx > mod / 2:
            dx = mod - dx
        dy = u_new_forward[1] - u_forward[1]
        LDs[0] += np.sqrt(dx**2 + dy**2)
        u_forward = u_new_forward

        # Backward evolution
        u_new_backward = backwards_mapping(u_backward, parameters)
        dx = abs(u_new_backward[0] - u_backward[0])
        if dx > mod / 2:
            dx = mod - dx
        dy = u_new_backward[1] - u_backward[1]
        LDs[1] += np.sqrt(dx**2 + dy**2)
        u_backward = u_new_backward

    return LDs


def RTE(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: int,
    mapping: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    transient_time: Optional[int] = None,
    **kwargs,
) -> Union[float, Tuple]:
    """
    Calculate Recurrence Time Entropy (RTE) for a dynamical system.

    RTE quantifies the complexity of a system by analyzing the distribution
    of white vertical lines, i.e., the gap between two diagonal lines.
    Higher entropy indicates more complex dynamics.

    Parameters
    ----------
    u : NDArray[np.float64]
        Initial state vector (shape: (neq,))
    parameters : NDArray[np.float64]
        System parameters passed to mapping function
    total_time : int
        Number of iterations to simulate
    mapping : Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]
        System evolution function: u_next = mapping(u, parameters)
    transient_time : Optional[int], default=None
        Time to wait before starting RTE calculation.
    **kwargs
        Configuration parameters (see RTEConfig)

    Returns
    -------
    Union[float, Tuple]
        - Base case: RTE value (float)
        - With optional returns: List containing [RTE, *requested_additional_data]

    Raises
    ------
    ValueError
        - If invalid metric specified
        - If trajectory generation fails

    Notes
    -----
    - Implements the method described in [1]
    - For optimal results:
        - Use total_time > 1000 for reliable statistics
        - Typical threshold values: 0.05-0.3
        - Set lmin=1 to include single-point recurrences

    References
    ----------
    [1] M. R. Sales, M. Mugnaine, J. Szezech, José D., R. L. Viana, I. L. Caldas, N. Marwan, and J. Kurths, Stickiness and recurrence plots: An entropy-based approach, Chaos: An Interdisciplinary Journal of Nonlinear Science 33, 033140 (2023)
    """

    u = u.copy()

    # Configuration handling
    config = RTEConfig(**kwargs)

    # Metric setup
    metric_map = {"supremum": np.inf, "euclidean": 2, "manhattan": 1}

    try:
        ord = metric_map[config.std_metric.lower()]
    except KeyError:
        raise ValueError(
            f"Invalid std_metric: {config.std_metric}. Must be {list(metric_map.keys())}"
        )

    if transient_time is not None:
        u = iterate_mapping(u, parameters, transient_time, mapping)
        total_time -= transient_time

    # Generate trajectory
    try:
        time_series = generate_trajectory(u, parameters, total_time, mapping)
    except Exception as e:
        raise ValueError(f"Trajectory generation failed: {str(e)}")

    # Threshold calculation
    if config.threshold_std:
        std = np.std(time_series, axis=0)
        eps = config.threshold * np.linalg.norm(std, ord=ord)
        if eps <= 0:
            eps = 0.1
    else:
        eps = config.threshold

    # Recurrence matrix calculation
    recmat = recurrence_matrix(time_series, float(eps), metric=config.metric)

    # White line distribution
    P = white_vertline_distr(recmat, wmin=config.lmin)
    P = P[P > 0]  # Remove zeros
    P /= P.sum()  # Normalize

    # Entropy calculation
    rte = -np.sum(P * np.log(P))

    # Prepare output
    result = [rte]
    if config.return_final_state:
        result.append(time_series[-1])
    if config.return_recmat:
        result.append(recmat)
    if config.return_p:
        result.append(P)

    return result[0] if len(result) == 1 else tuple(result)


def finite_time_RTE(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: int,
    finite_time: int,
    mapping: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    return_points: bool = False,
    **kwargs,
) -> Union[NDArray[np.float64], Tuple[NDArray[np.float64], NDArray[np.float64]]]:
    # Validate window size
    if finite_time > total_time:
        raise ValueError(
            f"finite_time ({finite_time}) exceeds available samples ({total_time})"
        )

    num_windows = total_time // finite_time
    RTE_values = np.zeros(num_windows)
    phase_space_points = np.zeros((num_windows, u.shape[0]))

    for i in range(num_windows):
        result = RTE(
            u, parameters, finite_time, mapping, return_final_state=True, **kwargs
        )
        if isinstance(result, tuple):
            RTE_values[i], u_new = result
            phase_space_points[i] = u
            u = u_new.copy()

    if return_points:
        return RTE_values, phase_space_points
    else:
        return RTE_values
