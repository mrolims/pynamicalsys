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

from typing import Callable, Optional, Tuple, Union, Any, Sequence

import numpy as np
from numba import njit
from numpy.typing import NDArray

from pynamicalsys.common.recurrence_quantification_analysis import (
    RTEConfig,
    build_recurrence_matrix,
    white_vertline_distr,
    calculate_threshold,
)
from pynamicalsys.common.time_series_metrics import hurst_exponent
from pynamicalsys.common.utils import (
    qr,
    wedge_norm,
    clv_col_normalize_inplace,
    clv_sanitize_inplace,
    clv_solve_upper_inplace,
)
from pynamicalsys.discrete_time.trajectory_analysis import (
    generate_trajectory,
    iterate_mapping,
)


@njit(error_model="numpy")
def compute_clvs(
    u,
    parameters,
    total_time,
    mapping,
    jacobian,
    num_clvs=None,
    transient_time=0,
    warmup_time=0,
    tail_time=0,
    seed=13,
    normalize_A=True,
    eps_norm=1e-300,
    rcond_guard=1e-14,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:

    np.random.seed(seed)

    u = np.asarray(u, dtype=np.float64).copy()
    parameters = np.asarray(parameters, dtype=np.float64)

    dim = u.size
    if num_clvs is None:
        num_clvs = dim
    if num_clvs < 1 or num_clvs > dim:
        raise ValueError("num_clvs must be in [1, dim]")

    # (A) Transient
    for _ in range(transient_time):
        u = mapping(u, parameters)

    # (B) Forward GS warm-up
    Q = np.eye(dim, num_clvs, dtype=np.float64)
    for _ in range(warmup_time):
        J = jacobian(u, parameters)
        Q_full, R_full = np.linalg.qr(J @ Q)
        Q = np.ascontiguousarray(Q_full[:, :num_clvs])
        u = mapping(u, parameters)

    # (C) Data collection window
    Q_store = np.zeros((total_time + 1, dim, num_clvs), dtype=np.float64)
    R_store = np.zeros((total_time, num_clvs, num_clvs), dtype=np.float64)
    Q_store[0] = Q

    traj = np.zeros((total_time + 1, dim), dtype=np.float64)
    traj[0] = u

    for i in range(total_time):
        J = jacobian(u, parameters)
        Q_full, R_full = np.linalg.qr(J @ Q)
        Q = np.ascontiguousarray(Q_full[:, :num_clvs])
        R = R_full[:num_clvs, :num_clvs]

        Q_store[i + 1] = Q
        R_store[i] = R

        u = mapping(u, parameters)
        traj[i + 1] = u

    # (D) Backward initialization (A_T -> A^-)
    A = np.triu(np.random.randn(num_clvs, num_clvs)).astype(np.float64)

    # Make sure A starts finite and reasonably scaled
    clv_sanitize_inplace(A)
    if normalize_A:
        clv_col_normalize_inplace(A, eps_norm)

    for _ in range(tail_time):
        J = jacobian(u, parameters)
        Q_full, R_full = np.linalg.qr(J @ Q)
        Q = np.ascontiguousarray(Q_full[:, :num_clvs])
        R = R_full[:num_clvs, :num_clvs]

        if normalize_A:
            clv_col_normalize_inplace(A, eps_norm)

        clv_solve_upper_inplace(R, A, rcond_guard)

        clv_sanitize_inplace(A)
        if normalize_A:
            clv_col_normalize_inplace(A, eps_norm)

        u = mapping(u, parameters)

    # (E) Backward recursion (CLVs)
    clvs = np.zeros((total_time + 1, dim, num_clvs), dtype=np.float64)

    # workspace
    V = np.empty((dim, num_clvs), dtype=np.float64)

    for t in range(total_time, -1, -1):
        if normalize_A:
            clv_col_normalize_inplace(A, eps_norm)

        V[:, :] = Q_store[t] @ A
        clv_sanitize_inplace(V)
        clv_col_normalize_inplace(V, eps_norm)
        clvs[t] = V

        if t > 0:
            clv_solve_upper_inplace(R_store[t - 1], A, rcond_guard)
            clv_sanitize_inplace(A)
            if normalize_A:
                clv_col_normalize_inplace(A, eps_norm)

    return clvs, traj


def _clv_angles(
    u: np.ndarray,
    parameters: np.ndarray,
    total_time: int,
    mapping: Callable[[np.ndarray, np.ndarray], np.ndarray],
    jacobian: Callable[[np.ndarray, np.ndarray], np.ndarray],
    warmup_time: int = 0,
    tail_time: int = 0,
    seed: int = 13,
    subspaces: Optional[Sequence[Tuple[Sequence[int], Sequence[int]]]] = None,
    pairs: Optional[Sequence[Tuple[int, int]]] = None,
    use_abs: bool = True,
    **clv_kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute CLV angle diagnostics.

    Computes:
    - minimum principal angles between user-defined subspaces
    - angles between user-defined CLV pairs

    At least one of `subspaces` or `pairs` must be provided.

    Returns
    -------
    angles : ndarray, shape (T, M)
        One column per requested angle:
        - first all subspace angles (in order given)
        - then all pairwise angles (in order given)
    traj : ndarray
        Trajectory returned by `compute_clvs`.
    """

    # -----------------------
    # Validate requests
    # -----------------------
    want_subspaces = subspaces is not None and len(subspaces) > 0
    want_pairs = pairs is not None and len(pairs) > 0

    if not want_subspaces and not want_pairs:
        raise ValueError("At least one of `subspaces` or `pairs` must be provided.")

    # -----------------------
    # Compute CLVs
    # -----------------------
    clvs, traj = compute_clvs(
        u=u,
        parameters=parameters,
        total_time=total_time,
        mapping=mapping,
        jacobian=jacobian,
        warmup_time=warmup_time,
        tail_time=tail_time,
        seed=seed,
        **clv_kwargs,
    )

    T, dim, num_clvs = clvs.shape

    # Normalize CLVs
    V = clvs / np.linalg.norm(clvs, axis=1, keepdims=True)

    n_sub = len(subspaces) if subspaces is not None else 0
    n_pairs = len(pairs) if pairs is not None else 0

    angles = np.empty((T, n_sub + n_pairs), dtype=np.float64)

    col = 0

    # -----------------------
    # Subspace angles
    # -----------------------
    if want_subspaces:
        for A_idx, B_idx in subspaces:
            for t in range(T):
                A = np.take(V[t], A_idx, axis=1)  # (dim, kA)
                B = np.take(V[t], B_idx, axis=1)  # (dim, kB)

                QA, _ = np.linalg.qr(A, mode="reduced")
                QB, _ = np.linalg.qr(B, mode="reduced")

                sigma_max = np.linalg.svd(QA.T @ QB, compute_uv=False)[0]
                if use_abs:
                    sigma_max = abs(sigma_max)
                sigma_max = np.clip(sigma_max, -1.0, 1.0)
                angles[t, col] = np.arccos(sigma_max)
            col += 1

    # -----------------------
    # Pairwise angles
    # -----------------------
    if want_pairs:
        for i, j in pairs:
            dots = np.einsum("td,td->t", V[:, :, i], V[:, :, j])
            if use_abs:
                dots = np.abs(dots)
            dots = np.clip(dots, -1.0, 1.0)
            angles[:, col] = np.arccos(dots)
            col += 1

    return angles, traj


def clv_angles(
    u: np.ndarray,
    parameters: np.ndarray,
    total_time: int,
    mapping: Callable[[np.ndarray, np.ndarray], np.ndarray],
    jacobian: Callable[[np.ndarray, np.ndarray], np.ndarray],
    subspaces: Optional[Sequence[Tuple[Sequence[int], Sequence[int]]]] = None,
    pairs: Optional[Sequence[Tuple[int, int]]] = None,
    window_time: Optional[int] = None,
    transient_time: int = 0,
    warmup_time: int = 0,
    tail_time: int = 0,
    seed: int = 13,
    use_abs: bool = True,
    **clv_kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Windowed or full-time CLV angle computation.

    Returns
    -------
    angles : ndarray
        If window_time is None:
            shape (T, M)
        Else:
            shape (num_windows, M + 1)
            first column is window center time index
    aux : ndarray
        If window_time is None:
            trajectory of the system
        Else:
            initial condition of each window
    """

    # -----------------------
    # Initial transient
    # -----------------------
    dim = u.shape[0]
    u = u.copy()

    for _ in range(transient_time):
        u = mapping(u, parameters)

    # -----------------------
    # No windowing
    # -----------------------
    if window_time is None:
        return _clv_angles(
            u=u,
            parameters=parameters,
            total_time=total_time,
            mapping=mapping,
            jacobian=jacobian,
            warmup_time=warmup_time,
            tail_time=tail_time,
            seed=seed,
            subspaces=subspaces,
            pairs=pairs,
            use_abs=use_abs,
            **clv_kwargs,
        )

    # -----------------------
    # Windowed computation
    # -----------------------
    num_windows = total_time // window_time

    # Determine number of angles M
    n_sub = 0 if subspaces is None else len(subspaces)
    n_pairs = 0 if pairs is None else len(pairs)

    if n_sub == 0 and n_pairs == 0:
        raise ValueError("At least one of `subspaces` or `pairs` must be provided.")

    M = n_sub + n_pairs

    # +1 column for window time index
    avg_angles = np.zeros((num_windows, M + 1), dtype=np.float64)
    initial_conditions = np.zeros((num_windows, dim), dtype=np.float64)

    # -----------------------
    # Window loop
    # -----------------------
    for i in range(num_windows):
        angles, traj = _clv_angles(
            u=u,
            parameters=parameters,
            total_time=window_time + tail_time,
            mapping=mapping,
            jacobian=jacobian,
            warmup_time=warmup_time,
            tail_time=tail_time,
            seed=seed,
            subspaces=subspaces,
            pairs=pairs,
            use_abs=use_abs,
            **clv_kwargs,
        )

        # Store IC of this window
        initial_conditions[i] = u.copy()

        # Window "time coordinate"
        avg_angles[i, 0] = i * window_time + 0.5 * (window_time - 1)

        # Average only over the well-conditioned part
        avg_angles[i, 1:] = angles[:window_time].mean(axis=0)

        # Advance IC
        u = traj[window_time].copy()

    return avg_angles, initial_conditions


def dig(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: int,
    mapping: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    transient_time: Optional[int] = None,
) -> float:
    """Compute the number of zeros after the decimal point (dig) of the weighted Birkhoff
    average convergence of a trajectory.

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
        dig value (higher values indicate better convergence)

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

    if transient_time is not None:
        u = iterate_mapping(u, parameters, transient_time, mapping)
        total_time -= transient_time

    # Generate trajectory
    try:
        time_series = generate_trajectory(u, parameters, total_time, mapping)
    except Exception as e:
        raise ValueError(f"Trajectory generation failed: {str(e)}")

    eps = calculate_threshold(time_series, config)

    # Recurrence matrix calculation
    recmat = build_recurrence_matrix(time_series, float(eps), metric=config.metric)

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
