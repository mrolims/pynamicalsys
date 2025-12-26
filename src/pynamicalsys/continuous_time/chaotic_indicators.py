# chaotic_indicators.py

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

from typing import Callable, Optional, Tuple, Union, Sequence, Any

import numpy as np
from numba import njit
from numpy.typing import NDArray

from pynamicalsys.common.recurrence_quantification_analysis import (
    RTEConfig,
    recurrence_matrix,
    white_vertline_distr,
)
from pynamicalsys.common.time_series_metrics import hurst_exponent
from pynamicalsys.common.utils import (
    fit_poly,
    qr,
    wedge_norm,
    clv_col_normalize_inplace,
    clv_sanitize_inplace,
    clv_solve_upper_inplace,
)
from pynamicalsys.continuous_time.numerical_integrators import rk4_step_wrapped
from pynamicalsys.continuous_time.trajectory_analysis import (
    evolve_system,
    generate_maxima_map,
    generate_poincare_section,
    generate_stroboscopic_map,
    step,
)


@njit
def lyapunov_exponents(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: float,
    equations_of_motion: Callable[
        [np.float64, NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]
    ],
    jacobian: Callable[
        [np.float64, NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]
    ],
    num_exponents: int,
    transient_time: Optional[float] = None,
    time_step: float = 0.01,
    atol: float = 1e-6,
    rtol: float = 1e-3,
    integrator=rk4_step_wrapped,
    return_history: bool = False,
    seed: int = 13,
    QR: Callable[
        [NDArray[np.float64]], Tuple[NDArray[np.float64], NDArray[np.float64]]
    ] = qr,
) -> NDArray[np.float64]:

    neq = len(u)  # Number of equations of the system
    nt = neq + neq * num_exponents  # system + variational equations

    u = u.copy()

    # Handle transient time
    if transient_time is not None:
        u = evolve_system(
            u,
            parameters,
            transient_time,
            equations_of_motion,
            time_step=time_step,
            atol=atol,
            rtol=rtol,
            integrator=integrator,
        )
        sample_time = total_time - transient_time
        time = transient_time
    else:
        sample_time = total_time
        time = 0

    # State + deviation vectors
    uv = np.zeros(nt)
    uv[:neq] = u.copy()

    # Randomly define the deviation vectors and orthonormalize them
    np.random.seed(seed)
    uv[neq:] = -1 + 2 * np.random.rand(nt - neq)
    v = uv[neq:].reshape(neq, num_exponents)
    v, _ = QR(v)
    uv[neq:] = v.reshape(neq * num_exponents)

    exponents = np.zeros(num_exponents, dtype=np.float64)
    history = []

    while time < total_time:
        if time + time_step > total_time:
            time_step = total_time - time

        uv, time, time_step = step(
            time,
            uv,
            parameters,
            equations_of_motion,
            jacobian=jacobian,
            time_step=time_step,
            atol=atol,
            rtol=rtol,
            integrator=integrator,
            number_of_deviation_vectors=num_exponents,
        )

        #  Reshape the deviation vectors into a neq x neq matrix
        v = uv[neq:].reshape(neq, num_exponents).copy()

        # Perform the QR decomposition
        v, R = QR(v)
        # Accumulate the log
        exponents += np.log(np.abs(np.diag(R)))

        if return_history:
            result = [time]
            for i in range(num_exponents):
                result.append(
                    exponents[i]
                    / (time - (transient_time if transient_time is not None else 0))
                )
            history.append(result)

        # Reshape v back to uv
        uv[neq:] = v.reshape(neq * num_exponents)

    if return_history:
        return history
    else:
        result = []
        for i in range(num_exponents):
            result.append(
                exponents[i]
                / (time - (transient_time if transient_time is not None else 0))
            )
        return [result]


@njit
def maximum_lyapunov_exponent(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: float,
    equations_of_motion: Callable[
        [np.float64, NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]
    ],
    jacobian: Callable[
        [np.float64, NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]
    ],
    transient_time: Optional[float] = None,
    time_step: float = 0.01,
    atol: float = 1e-6,
    rtol: float = 1e-3,
    integrator=rk4_step_wrapped,
    return_history: bool = False,
    seed: int = 13,
) -> NDArray[np.float64]:

    neq = len(u)  # Number of equations of the system
    nt = neq + neq  # system + variational equations

    u = u.copy()

    # Handle transient time
    if transient_time is not None:
        u = evolve_system(
            u,
            parameters,
            transient_time,
            equations_of_motion,
            time_step=time_step,
            atol=atol,
            rtol=rtol,
            integrator=integrator,
        )
        sample_time = total_time - transient_time
        time = transient_time
    else:
        sample_time = total_time
        time = 0

    # State + deviation vectors
    uv = np.zeros(nt)
    uv[:neq] = u.copy()

    # Randomly define the deviation vectors and orthonormalize them
    np.random.seed(seed)
    uv[neq:] = -1 + 2 * np.random.rand(nt - neq)
    norm = np.linalg.norm(uv[neq:])
    uv[neq:] /= norm

    exponent = 0.0
    history = []

    while time < total_time:
        if time + time_step > total_time:
            time_step = total_time - time

        uv, time, time_step = step(
            time,
            uv,
            parameters,
            equations_of_motion,
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
            result = [time]
            result.append(
                exponent
                / (time - (transient_time if transient_time is not None else 0))
            )
            history.append(result)

    if return_history:
        return history
    else:
        result = [
            exponent / (time - (transient_time if transient_time is not None else 0))
        ]
        return [result]


@njit(error_model="numpy")
def compute_clvs(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: float,
    equations_of_motion: Callable[
        [np.float64, NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]
    ],
    jacobian: Callable[
        [np.float64, NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]
    ],
    num_clvs: int,
    transient_time: Optional[float] = None,
    warmup_time: float = 0.0,
    tail_time: float = 0.0,
    time_step: float = 0.01,
    qr_time_step: float = 0.1,
    atol: float = 1e-6,
    rtol: float = 1e-3,
    integrator=rk4_step_wrapped,
    seed: int = 13,
    QR: Callable[
        [NDArray[np.float64]], Tuple[NDArray[np.float64], NDArray[np.float64]]
    ] = qr,
    normalize_A: bool = True,
    eps_norm: float = 1e-300,
    rcond_guard: float = 1e-14,
) -> Union[
    NDArray[np.float64],
    Tuple[NDArray[np.float64], NDArray[np.float64]],
]:
    """
    Continuous-time CLVs sampled every `qr_time_step`.
    (Docstring intentionally omitted here per your request—this is the low-level core.)
    """

    # -----------------------------
    # Basic sizes / sanity
    # -----------------------------
    u = u.copy()
    neq = u.shape[0]
    p = num_clvs
    if p < 1:
        p = 1
    if p > neq:
        p = neq

    # -----------------------------
    # Transient (state only)
    # -----------------------------
    if transient_time is not None:
        u = evolve_system(
            u,
            parameters,
            transient_time,
            equations_of_motion,
            time_step=time_step,
            atol=atol,
            rtol=rtol,
            integrator=integrator,
        )
        t0 = transient_time
    else:
        t0 = 0.0

    t_end = total_time

    # -----------------------------
    # Build state+tangent container uv
    # uv = [x ; vec(W)] where W is neq×p
    # -----------------------------
    nt = neq + neq * p
    uv = np.zeros(nt, dtype=np.float64)
    uv[:neq] = u

    # initial orthonormal tangent basis W = I[:, :p]
    W = np.eye(neq, p, dtype=np.float64)
    uv[neq:] = W.reshape(neq * p)

    # We keep `dt` mutable (like your LE routine)
    dt = time_step
    time = t0

    # -----------------------------
    # Choose QR sampling grid
    # -----------------------------
    if qr_time_step <= 0.0:
        qr_time_step = dt  # fallback

    # number of QR blocks for warmup / storage / tail
    # We sample at: time = t0 + k*qr_time_step
    warm_blocks = 0
    if warmup_time > 0.0:
        warm_blocks = int(np.floor(warmup_time / qr_time_step))
        if warm_blocks < 0:
            warm_blocks = 0

    total_blocks = int(np.floor((t_end - t0) / qr_time_step))
    if total_blocks < 1:
        total_blocks = 1

    tail_blocks = 0
    if tail_time > 0.0:
        tail_blocks = int(np.floor(tail_time / qr_time_step))
        if tail_blocks < 0:
            tail_blocks = 0

    # -----------------------------
    # Warm-up: do QR steps without storing
    # -----------------------------
    for _ in range(warm_blocks):
        t_target = time + qr_time_step
        if t_target > t_end:
            t_target = t_end

        # integrate uv to t_target
        while time < t_target:
            if time + dt > t_target:
                dt = t_target - time

            uv, time, dt = step(
                time,
                uv,
                parameters,
                equations_of_motion,
                jacobian=jacobian,
                time_step=dt,
                atol=atol,
                rtol=rtol,
                integrator=integrator,
                number_of_deviation_vectors=p,
            )

        # QR on the tangent block
        W = uv[neq:].reshape(neq, p).copy()
        Q, _R = QR(W)
        Q = np.ascontiguousarray(Q[:, :p])
        uv[neq:] = Q.reshape(neq * p)

        if time >= t_end:
            break

    # -----------------------------
    # Storage window: allocate and store Q, R (and trajectory optionally)
    # We store at k=0..total_blocks, inclusive => total_blocks+1 Q's
    # and R at k=0..total_blocks-1 => total_blocks R's
    # -----------------------------
    Q_store = np.zeros((total_blocks + 1, neq, p), dtype=np.float64)
    R_store = np.zeros((total_blocks, p, p), dtype=np.float64)

    traj = np.zeros((total_blocks + 1, neq + 1), dtype=np.float64)

    # store initial sample at current time (k=0)
    Q_store[0] = uv[neq:].reshape(neq, p)
    traj[0, 0] = time
    traj[0, 1:] = uv[:neq]

    # forward through storage blocks
    time_eps = 10.0 * np.finfo(np.float64).eps  # ~2e-15 scale
    dt_min = 100.0 * np.finfo(np.float64).eps  # or a user-set value

    for k in range(total_blocks):
        t_target = t0 + (k + 1) * qr_time_step
        if t_target > t_end:
            t_target = t_end

        # integrate until (almost) t_target
        while time < t_target - time_eps:
            dt_rem = t_target - time

            # If we're extremely close, snap and stop
            if dt_rem <= dt_min:
                time = t_target
                break

            # Don't step past the target
            if dt > dt_rem:
                dt = dt_rem

            uv, time, dt = step(
                time,
                uv,
                parameters,
                equations_of_motion,
                jacobian=jacobian,
                time_step=dt,
                atol=atol,
                rtol=rtol,
                integrator=integrator,
                number_of_deviation_vectors=p,
            )

        # SNAP (optional but recommended): avoid drift accumulation
        time = t_target

        # IMPORTANT: reset dt so we don't carry a tiny dt into the next block
        dt = time_step

        W = uv[neq:].reshape(neq, p).copy()
        Q, R_full = QR(W)
        Q = np.ascontiguousarray(Q[:, :p])
        R = R_full[:p, :p]

        Q_store[k + 1] = Q
        R_store[k] = R

        uv[neq:] = Q.reshape(neq * p)
        traj[k + 1, 0] = time
        traj[k + 1, 1:] = uv[:neq]

        if time >= t_end:
            # if we hit end early, the remaining stores are left as zeros;
            # but total_blocks computed from floor should avoid this in practice.
            break

    # -----------------------------
    # Tail: integrate further and store R_tail (for backward init)
    # -----------------------------
    R_tail = np.zeros((tail_blocks, p, p), dtype=np.float64)

    for k in range(tail_blocks):
        t_target = time + qr_time_step

        while time < t_target:
            if time + dt > t_target:
                dt = t_target - time

            uv, time, dt = step(
                time,
                uv,
                parameters,
                equations_of_motion,
                jacobian=jacobian,
                time_step=dt,
                atol=atol,
                rtol=rtol,
                integrator=integrator,
                number_of_deviation_vectors=p,
            )

        W = uv[neq:].reshape(neq, p).copy()
        Q, R_full = QR(W)
        Q = np.ascontiguousarray(Q[:, :p])
        R = R_full[:p, :p]
        R_tail[k] = R

        uv[neq:] = Q.reshape(neq * p)

    # -----------------------------
    # Backward init: build A^- using reversed tail R's
    # -----------------------------
    np.random.seed(seed)
    A = np.triu(np.random.randn(p, p)).astype(np.float64)

    for k in range(tail_blocks - 1, -1, -1):
        if normalize_A:
            clv_col_normalize_inplace(A, eps_norm)
        clv_solve_upper_inplace(R_tail[k], A, rcond_guard)

    # -----------------------------
    # Backward recursion: reconstruct CLVs on stored grid
    # -----------------------------
    clvs = np.zeros((total_blocks + 1, neq, p), dtype=np.float64)

    for k in range(total_blocks, -1, -1):
        if normalize_A:
            clv_col_normalize_inplace(A, eps_norm)

        V = Q_store[k] @ A
        clv_col_normalize_inplace(V, eps_norm)
        clvs[k] = V

        if k > 0:
            clv_solve_upper_inplace(R_store[k - 1], A, rcond_guard)

    return clvs, traj


def clv_angles(
    u: np.ndarray,
    parameters: np.ndarray,
    total_time: int,
    equations_of_motion: Callable[
        [np.float64, NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]
    ],
    jacobian: Callable[
        [np.float64, NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]
    ],
    transient_time: int = 0,
    warmup_time: int = 0,
    tail_time: int = 0,
    time_step: float = 0.01,
    qr_time_step: float = 0.1,
    atol: float = 1e-6,
    rtol: float = 1e-3,
    integrator=rk4_step_wrapped,
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

    dim = len(u)

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
        equations_of_motion=equations_of_motion,
        jacobian=jacobian,
        num_clvs=dim,
        transient_time=transient_time,
        warmup_time=warmup_time,
        tail_time=tail_time,
        time_step=time_step,
        qr_time_step=qr_time_step,
        atol=atol,
        rtol=rtol,
        integrator=integrator,
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


@njit
def SALI(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: float,
    equations_of_motion: Callable[
        [np.float64, NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]
    ],
    jacobian: Callable[
        [np.float64, NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]
    ],
    transient_time: Optional[float] = None,
    time_step: float = 0.01,
    atol: float = 1e-6,
    rtol: float = 1e-3,
    integrator=rk4_step_wrapped,
    return_history: bool = False,
    seed: int = 13,
    threshold: float = 1e-16,
) -> NDArray[np.float64]:

    neq = len(u)  # Number of equations of the system
    ndv = 2  # Number of deviation vectors
    nt = neq + neq * ndv  # Total number of equations including variational equations

    u = u.copy()

    # Handle transient time
    if transient_time is not None:
        u = evolve_system(
            u,
            parameters,
            transient_time,
            equations_of_motion,
            time_step=time_step,
            atol=atol,
            rtol=rtol,
            integrator=integrator,
        )
        time = transient_time
    else:
        time = 0

    # State + deviation vectors
    uv = np.zeros(nt)
    uv[:neq] = u.copy()

    # Randomly define the deviation vectors and orthonormalize them
    np.random.seed(seed)
    uv[neq:] = -1 + 2 * np.random.rand(nt - neq)
    v = uv[neq:].reshape(neq, ndv)
    v, _ = qr(v)
    uv[neq:] = v.reshape(neq * ndv)

    history = []

    while time < total_time:
        if time + time_step > total_time:
            time_step = total_time - time

        uv, time, time_step = step(
            time,
            uv,
            parameters,
            equations_of_motion,
            jacobian=jacobian,
            time_step=time_step,
            atol=atol,
            rtol=rtol,
            integrator=integrator,
            number_of_deviation_vectors=ndv,
        )

        # Reshape the deviation vectors into a neq x ndv matrix
        v = uv[neq:].reshape(neq, ndv)

        # Normalize the deviation vectors
        v[:, 0] /= np.linalg.norm(v[:, 0])
        v[:, 1] /= np.linalg.norm(v[:, 1])

        # Calculate the aligment indexes and SALI
        PAI = np.linalg.norm(v[:, 0] + v[:, 1])
        AAI = np.linalg.norm(v[:, 0] - v[:, 1])
        sali = min(PAI, AAI)

        if return_history:
            result = [time, sali]
            history.append(result)

        # Early termination
        if sali <= threshold:
            break

        # Reshape v back to uv
        uv[neq:] = v.reshape(neq * ndv)

    if return_history:
        return history
    else:
        return [[time, sali]]


def LDI(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: float,
    equations_of_motion: Callable[
        [np.float64, NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]
    ],
    jacobian: Callable[
        [np.float64, NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]
    ],
    number_deviation_vectors: int,
    transient_time: Optional[float] = None,
    time_step: float = 0.01,
    atol: float = 1e-6,
    rtol: float = 1e-3,
    integrator=rk4_step_wrapped,
    return_history: bool = False,
    seed: int = 13,
    threshold: float = 1e-16,
) -> NDArray[np.float64]:

    neq = len(u)  # Number of equations of the system
    ndv = number_deviation_vectors  # Number of deviation vectors
    nt = neq + neq * ndv  # Total number of equations including variational equations

    u = u.copy()

    # Handle transient time
    if transient_time is not None:
        u = evolve_system(
            u,
            parameters,
            transient_time,
            equations_of_motion,
            time_step=time_step,
            atol=atol,
            rtol=rtol,
            integrator=integrator,
        )
        time = transient_time
    else:
        time = 0

    # State + deviation vectors
    uv = np.zeros(nt)
    uv[:neq] = u.copy()

    # Randomly define the deviation vectors and orthonormalize them
    np.random.seed(seed)
    uv[neq:] = -1 + 2 * np.random.rand(nt - neq)
    v = uv[neq:].reshape(neq, ndv)
    v, _ = qr(v)
    uv[neq:] = v.reshape(neq * ndv)

    history = []

    while time < total_time:
        if time + time_step > total_time:
            time_step = total_time - time

        uv, time, time_step = step(
            time,
            uv,
            parameters,
            equations_of_motion,
            jacobian=jacobian,
            time_step=time_step,
            atol=atol,
            rtol=rtol,
            integrator=integrator,
            number_of_deviation_vectors=ndv,
        )

        # Reshape the deviation vectors into a neq x ndv matrix
        v = uv[neq:].reshape(neq, ndv)

        # Normalize the deviation vectors
        for i in range(ndv):
            v[:, i] /= np.linalg.norm(v[:, i])

        # Calculate the singular values
        S = np.linalg.svd(v, full_matrices=False, compute_uv=False)
        ldi = np.exp(np.sum(np.log(S)))  # LDI is the product of all singular values
        # Instead of computing prod(S) directly, which could lead to underflows
        # or overflows, we compute the sum_{i=1}^k log(S_i) and then take the
        # exponential of this sum.

        if return_history:
            result = [time, ldi]
            history.append(result)

        # Early termination
        if ldi <= threshold:
            break

        # Reshape v back to uv
        uv[neq:] = v.reshape(neq * ndv)

    if return_history:
        return history
    else:
        return [[time, ldi]]


def GALI(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: float,
    equations_of_motion: Callable[
        [np.float64, NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]
    ],
    jacobian: Callable[
        [np.float64, NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]
    ],
    number_deviation_vectors: int,
    transient_time: Optional[float] = None,
    time_step: float = 0.01,
    atol: float = 1e-6,
    rtol: float = 1e-3,
    integrator=rk4_step_wrapped,
    return_history: bool = False,
    seed: int = 13,
    threshold: float = 1e-16,
) -> NDArray[np.float64]:

    neq = len(u)  # Number of equations of the system
    ndv = number_deviation_vectors  # Number of deviation vectors
    nt = neq + neq * ndv  # Total number of equations including variational equations

    u = u.copy()

    # Handle transient time
    if transient_time is not None:
        u = evolve_system(
            u,
            parameters,
            transient_time,
            equations_of_motion,
            time_step=time_step,
            atol=atol,
            rtol=rtol,
            integrator=integrator,
        )
        time = transient_time
    else:
        time = 0

    # State + deviation vectors
    uv = np.zeros(nt)
    uv[:neq] = u.copy()

    # Randomly define the deviation vectors and orthonormalize them
    np.random.seed(seed)
    uv[neq:] = -1 + 2 * np.random.rand(nt - neq)
    v = uv[neq:].reshape(neq, ndv)
    v, _ = qr(v)
    uv[neq:] = v.reshape(neq * ndv)

    history = []

    while time < total_time:
        if time + time_step > total_time:
            time_step = total_time - time

        uv, time, time_step = step(
            time,
            uv,
            parameters,
            equations_of_motion,
            jacobian=jacobian,
            time_step=time_step,
            atol=atol,
            rtol=rtol,
            integrator=integrator,
            number_of_deviation_vectors=ndv,
        )

        # Reshape the deviation vectors into a neq x ndv matrix
        v = uv[neq:].reshape(neq, ndv)

        # Normalize the deviation vectors
        for i in range(ndv):
            v[:, i] /= np.linalg.norm(v[:, i])

        # Calculate GALI
        gali = wedge_norm(v)

        if return_history:
            result = [time, gali]
            history.append(result)

        # Early termination
        if gali <= threshold:
            break

        # Reshape v back to uv
        uv[neq:] = v.reshape(neq * ndv)

    if return_history:
        return history
    else:
        return [[time, gali]]


def recurrence_time_entropy(
    u,
    parameters,
    num_points,
    transient_time,
    equations_of_motion,
    time_step,
    atol,
    rtol,
    integrator,
    map_type,
    section_index,
    section_value,
    crossing,
    sampling_time,
    maxima_index,
    **kwargs,
):

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

    # Generate the Poincaré section or stroboscopic map
    if map_type == "PS":
        points = generate_poincare_section(
            u,
            parameters,
            num_points,
            equations_of_motion,
            transient_time,
            time_step,
            atol,
            rtol,
            integrator,
            section_index,
            section_value,
            crossing,
        )
        data = points[:, 1:]  # Remove time
        data = np.delete(data, section_index, axis=1)
    elif map_type == "SM":
        points = generate_stroboscopic_map(
            u,
            parameters,
            num_points,
            sampling_time,
            equations_of_motion,
            transient_time,
            time_step,
            atol,
            rtol,
            integrator,
        )

        data = points[:, 1:]  # Remove time
    else:
        points = generate_maxima_map(
            u,
            parameters,
            num_points,
            maxima_index,
            equations_of_motion,
            transient_time,
            time_step,
            atol,
            rtol,
            integrator,
        )

        data = points[:, 1:]  # Remove time

    # Threshold calculation
    if config.threshold_std:
        std = np.std(data, axis=0)
        eps = config.threshold * np.linalg.norm(std, ord=ord)
        if eps <= 0:
            eps = 0.1
    else:
        eps = config.threshold

    # Recurrence matrix calculation
    recmat = recurrence_matrix(data, float(eps), metric=config.metric)

    # White line distribution
    P = white_vertline_distr(recmat, wmin=config.lmin)
    P = P[P > 0]  # Remove zeros
    P /= P.sum()  # Normalize

    # Entropy calculation
    rte = -np.sum(P * np.log(P))

    # Prepare output
    result = [rte]
    if config.return_final_state:
        result.append(points[-1, 1:])
    if config.return_recmat:
        result.append(recmat)
    if config.return_p:
        result.append(P)

    return result[0] if len(result) == 1 else tuple(result)


def hurst_exponent_wrapped(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    num_points: int,
    equations_of_motion: Callable,
    time_step: float,
    atol: float,
    rtol: float,
    integrator: Callable,
    map_type: str,
    section_index: int,
    section_value: float,
    crossing: int,
    sampling_time: float,
    maxima_index: int,
    wmin: int = 2,
    transient_time: Optional[int] = None,
) -> NDArray[np.float64]:

    u = u.copy()

    # Generate the Poincaré section or stroboscopic map
    if map_type == "PS":
        points = generate_poincare_section(
            u,
            parameters,
            num_points,
            equations_of_motion,
            transient_time,
            time_step,
            atol,
            rtol,
            integrator,
            section_index,
            section_value,
            crossing,
        )
        data = points[:, 1:]  # Remove time
        data = np.delete(data, section_index, axis=1)
    elif map_type == "SM":
        points = generate_stroboscopic_map(
            u,
            parameters,
            num_points,
            sampling_time,
            equations_of_motion,
            transient_time,
            time_step,
            atol,
            rtol,
            integrator,
        )

        data = points[:, 1:]  # Remove time
    else:
        points = generate_maxima_map(
            u,
            parameters,
            num_points,
            maxima_index,
            equations_of_motion,
            transient_time,
            time_step,
            atol,
            rtol,
            integrator,
        )

        data = points[:, 1:]  # Remove time

    return hurst_exponent(data, wmin=wmin)
