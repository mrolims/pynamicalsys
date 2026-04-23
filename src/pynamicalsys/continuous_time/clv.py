# clv.py

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

from typing import Any

import numpy as np
from numba import njit
from numpy.typing import NDArray

from pynamicalsys.common.types import flow_t, flow_jacobian_t
from pynamicalsys.common.utils import (
    qr,
    clv_col_normalize_inplace,
    clv_solve_upper_inplace,
)
from pynamicalsys.continuous_time.step_methods import rk4_step_wrapped
from pynamicalsys.continuous_time.step import evolve_system, step


@njit(error_model="numpy")
def compute_clvs(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: np.float64,
    equations_of_motion: flow_t,
    jacobian: flow_jacobian_t,
    num_clvs: int,
    transient_time: np.float64 | None = None,
    warmup_time: np.float64 = np.float64(0.0),
    tail_time: np.float64 = np.float64(0.0),
    time_step: np.float64 = np.float64(0.01),
    qr_time_step: np.float64 = np.float64(0.1),
    atol: np.float64 = np.float64(1e-6),
    rtol: np.float64 = np.float64(1e-3),
    integrator=rk4_step_wrapped,
    seed: int = 13,
    QR=qr,
    normalize_A: bool = True,
    eps_norm: np.float64 = np.float64(1e-300),
    rcond_guard: np.float64 = np.float64(1e-14),
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute covariant Lyapunov vectors (CLVs) for a continuous-time dynamical
    system sampled at intervals of `qr_time_step`.

    Parameters
    ----------
    u : NDArray[np.float64]
        Initial state vector.
    parameters : NDArray[np.float64]
        System parameters passed to the equations of motion and Jacobian.
    total_time : np.float64
        Final integration time of the storage window.
    equations_of_motion : flow_t
        Continuous-time vector field with signature
        `(time, u, parameters) -> dudt`.
    jacobian : flow_jacobian_t
        Jacobian of the vector field with signature
        `(time, u, parameters) -> J`.
    num_clvs : int
        Number of CLVs to compute.
    transient_time : np.float64 | None, optional
        Initial integration time discarded before CLV sampling begins.
    warmup_time : np.float64, optional
        Additional forward warmup time before storing QR factors.
    tail_time : np.float64, optional
        Additional forward tail time used to initialize the backward recursion.
    time_step : np.float64, optional
        Initial integration step size.
    qr_time_step : np.float64, optional
        Time interval between successive QR samplings.
    atol : np.float64, optional
        Absolute tolerance used by adaptive integrators.
    rtol : np.float64, optional
        Relative tolerance used by adaptive integrators.
    integrator : callable, optional
        Step method with unified signature returning
        `(u_next, time_next, time_step_next, accept)`.
    seed : int, optional
        Random seed used to initialize the backward coefficient matrix.
    QR : callable, optional
        QR decomposition routine used during the forward orthonormalization.
    normalize_A : bool, optional
        Whether to normalize the columns of the backward coefficient matrix
        during the backward recursion.
    eps_norm : np.float64, optional
        Small cutoff used in column normalization.
    rcond_guard : np.float64, optional
        Guard threshold for the triangular backsolve.

    Returns
    -------
    tuple[NDArray[np.float64], NDArray[np.float64]]
        - `clvs`: array of shape `(num_samples, neq, p)` containing the CLVs
        - `traj`: array of shape `(num_samples, neq + 1)` containing time in the
          first column and the corresponding trajectory point in the remaining
          columns

    Notes
    -----
    The algorithm performs:
    - forward integration of the tangent dynamics with QR reorthonormalization
    - storage of the orthonormal bases and triangular factors
    - optional forward tail integration for backward initialization
    - backward recursion through the stored triangular factors to reconstruct
      the CLVs
    """
    u = u.copy()
    neq = u.shape[0]
    p = num_clvs
    if p < 1:
        p = 1
    if p > neq:
        p = neq

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
        t0 = transient_time
    else:
        t0 = np.float64(0.0)

    t_end = total_time

    nt = neq + neq * p
    uv = np.zeros(nt, dtype=np.float64)
    uv[:neq] = u

    W = np.eye(neq, p, dtype=np.float64)
    uv[neq:] = W.reshape(neq * p)

    dt = time_step
    time = t0

    if qr_time_step <= 0.0:
        qr_time_step = dt

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

    for _ in range(warm_blocks):
        t_target = time + qr_time_step
        if t_target > t_end:
            t_target = t_end

        while time < t_target:
            if time + dt > t_target:
                dt = t_target - time

            uv, time, dt = step(
                time=time,
                u=uv,
                parameters=parameters,
                equations_of_motion=equations_of_motion,
                jacobian=jacobian,
                time_step=dt,
                atol=atol,
                rtol=rtol,
                integrator=integrator,
                number_of_deviation_vectors=p,
            )

        W = uv[neq:].reshape(neq, p).copy()
        Q, _ = QR(W)
        Q = np.ascontiguousarray(Q[:, :p])
        uv[neq:] = Q.reshape(neq * p)

        if time >= t_end:
            break

    Q_store = np.zeros((total_blocks + 1, neq, p), dtype=np.float64)
    R_store = np.zeros((total_blocks, p, p), dtype=np.float64)
    traj = np.zeros((total_blocks + 1, neq + 1), dtype=np.float64)

    Q_store[0] = uv[neq:].reshape(neq, p)
    traj[0, 0] = time
    traj[0, 1:] = uv[:neq]

    time_eps = np.float64(10.0) * np.finfo(np.float64).eps
    dt_min = np.float64(100.0) * np.finfo(np.float64).eps

    for k in range(total_blocks):
        t_target = t0 + (k + 1) * qr_time_step
        if t_target > t_end:
            t_target = t_end

        while time < t_target - time_eps:
            dt_rem = t_target - time

            if dt_rem <= dt_min:
                time = t_target
                break

            if dt > dt_rem:
                dt = dt_rem

            uv, time, dt = step(
                time=time,
                u=uv,
                parameters=parameters,
                equations_of_motion=equations_of_motion,
                jacobian=jacobian,
                time_step=dt,
                atol=atol,
                rtol=rtol,
                integrator=integrator,
                number_of_deviation_vectors=p,
            )

        time = t_target
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
            break

    R_tail = np.zeros((tail_blocks, p, p), dtype=np.float64)

    for k in range(tail_blocks):
        t_target = time + qr_time_step

        while time < t_target:
            if time + dt > t_target:
                dt = t_target - time

            uv, time, dt = step(
                time=time,
                u=uv,
                parameters=parameters,
                equations_of_motion=equations_of_motion,
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

    np.random.seed(seed)
    A = np.triu(np.random.randn(p, p)).astype(np.float64)

    for k in range(tail_blocks - 1, -1, -1):
        if normalize_A:
            clv_col_normalize_inplace(A, eps_norm)
        clv_solve_upper_inplace(R_tail[k], A, rcond_guard)

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
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: np.float64,
    equations_of_motion: flow_t,
    jacobian: flow_jacobian_t,
    transient_time: np.float64 = np.float64(0.0),
    warmup_time: np.float64 = np.float64(0.0),
    tail_time: np.float64 = np.float64(0.0),
    time_step: np.float64 = np.float64(0.01),
    qr_time_step: np.float64 = np.float64(0.1),
    atol: np.float64 = np.float64(1e-6),
    rtol: np.float64 = np.float64(1e-3),
    integrator=rk4_step_wrapped,
    seed: int = 13,
    subspaces: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...] | None = None,
    pairs: tuple[tuple[int, int], ...] | None = None,
    use_abs: bool = True,
    **clv_kwargs: Any,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute CLV angle diagnostics from continuous-time covariant Lyapunov vectors.

    Parameters
    ----------
    u : NDArray[np.float64]
        Initial state vector.
    parameters : NDArray[np.float64]
        System parameters passed to the equations of motion and Jacobian.
    total_time : np.float64
        Final integration time of the storage window.
    equations_of_motion : flow_t
        Continuous-time vector field with signature
        `(time, u, parameters) -> dudt`.
    jacobian : flow_jacobian_t
        Jacobian of the vector field with signature
        `(time, u, parameters) -> J`.
    transient_time : np.float64, optional
        Initial integration time discarded before CLV sampling begins.
    warmup_time : np.float64, optional
        Additional forward warmup time before storing QR factors.
    tail_time : np.float64, optional
        Additional forward tail time used to initialize the backward recursion.
    time_step : np.float64, optional
        Initial integration step size.
    qr_time_step : np.float64, optional
        Time interval between successive QR samplings.
    atol : np.float64, optional
        Absolute tolerance used by adaptive integrators.
    rtol : np.float64, optional
        Relative tolerance used by adaptive integrators.
    integrator : callable, optional
        Step method with unified signature returning
        `(u_next, time_next, time_step_next, accept)`.
    seed : int, optional
        Random seed passed to `compute_clvs`.
    subspaces : tuple[tuple[tuple[int, ...], tuple[int, ...]], ...] | None, optional
        Pairs of CLV index sets defining subspaces whose minimum principal angle
        will be computed.
    pairs : tuple[tuple[int, int], ...] | None, optional
        Pairs of CLV indices whose mutual angle will be computed.
    use_abs : bool, optional
        If True, use absolute scalar products before applying arccos.
    **clv_kwargs : Any
        Additional keyword arguments forwarded to `compute_clvs`.

    Returns
    -------
    tuple[NDArray[np.float64], NDArray[np.float64]]
        - `angles`: array of shape `(T, M)` containing the requested angles
        - `traj`: trajectory array returned by `compute_clvs`

    Notes
    -----
    The output columns are ordered as:
    - first all subspace angles, in the order given by `subspaces`
    - then all pairwise CLV angles, in the order given by `pairs`
    """
    dim = len(u)

    want_subspaces = subspaces is not None and len(subspaces) > 0
    want_pairs = pairs is not None and len(pairs) > 0

    if not want_subspaces and not want_pairs:
        raise ValueError("At least one of `subspaces` or `pairs` must be provided.")

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
    V = clvs / np.linalg.norm(clvs, axis=1, keepdims=True)

    n_sub = len(subspaces) if subspaces is not None else 0
    n_pairs = len(pairs) if pairs is not None else 0
    angles = np.empty((T, n_sub + n_pairs), dtype=np.float64)

    col = 0

    if want_subspaces:
        assert subspaces is not None
        for A_idx, B_idx in subspaces:
            for t in range(T):
                A = np.take(V[t], A_idx, axis=1)
                B = np.take(V[t], B_idx, axis=1)

                QA, _ = np.linalg.qr(A, mode="reduced")
                QB, _ = np.linalg.qr(B, mode="reduced")

                sigma_max = np.linalg.svd(QA.T @ QB, compute_uv=False)[0]
                if use_abs:
                    sigma_max = abs(sigma_max)
                sigma_max = np.clip(sigma_max, -1.0, 1.0)
                angles[t, col] = np.arccos(sigma_max)
            col += 1

    if want_pairs:
        assert pairs is not None
        for i, j in pairs:
            dots = np.einsum("td,td->t", V[:, :, i], V[:, :, j])
            if use_abs:
                dots = np.abs(dots)
            dots = np.clip(dots, -1.0, 1.0)
            angles[:, col] = np.arccos(dots)
            col += 1

    return angles, traj
