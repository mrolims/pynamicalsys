# clv.py

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

from typing import Optional, Sequence

import numpy as np
from numba import njit
from numpy.typing import NDArray

from pynamicalsys.common.types import system_func_t, symplectic_tangent_step_t
from pynamicalsys.common.linalg import qr, qr_truncate
from pynamicalsys.common.clv import (
    clv_col_normalize_inplace,
    clv_solve_upper_inplace,
)
from pynamicalsys.hamiltonian_systems.poincare import (
    generate_poincare_section_from_traj_imp,
    generate_poincare_section_from_traj_sep,
)
from pynamicalsys.hamiltonian_systems.tangent import (
    advance_block_imp,
    advance_block_sep,
)


"""
TODO

- Factor out the common CLVs logic. The only difference
  between the integrators is the derivative callbacks
  and signatures: 
      (grad_T, grad_V, hess_T, hess_V) vs. (eom, hess_H).
"""


@njit(error_model="numpy")
def compute_clvs_sep(
    q: NDArray[np.float64],
    p: NDArray[np.float64],
    total_time: np.float64,
    time_step: np.float64,
    parameters: NDArray[np.float64],
    grad_T: system_func_t,
    grad_V: system_func_t,
    hess_T: system_func_t,
    hess_V: system_func_t,
    num_clvs: int,
    warmup_time: np.float64,
    tail_time: np.float64,
    qr_time_step: np.float64,
    seed: int,
    method: str,
    integrator_traj_tan: symplectic_tangent_step_t,
    poincare_section: bool,
    section_index: int,
    section_value: np.float64,
    crossing: int,
    normalize_A: bool = True,
    eps_norm: np.float64 = np.float64(1e-300),
    rcond_guard: np.float64 = np.float64(1e-14),
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute covariant Lyapunov vectors (CLVs) for a separable Hamiltonian
    system, H(q, p) = T(p) + V(q), integrated with an explicit symplectic
    stepper (velocity Verlet or fourth-order Yoshida).

    Requires grad_T, grad_V, hess_V, and hess_T to advance the trajectory
    and tangent vectors.

    Parameters
    ----------
    q : NDArray[np.float64]
        Initial generalized coordinates of shape `(dof,)`.
    p : NDArray[np.float64]
        Initial generalized momenta of shape `(dof,)`.
    total_time : np.float64
        Total integration time over the storage window.
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
    num_clvs : int
        Number of CLVs to compute.
    warmup_time : np.float64
        Forward warmup time used before storing QR factors.
    tail_time : np.float64
        Extra forward time used to initialize the backward recursion.
    qr_time_step : np.float64
        Time interval between successive QR factorizations.
    seed : int
        Random seed used to initialize the tangent basis and the backward matrix.
    method : str
        QR method used in the orthonormalization step. Supported options are:

        - `"QR"`:
          Use the internal reduced modified Gram-Schmidt QR routine `qr`.

        - `"QR_HH"`:
          Use `numpy.linalg.qr`, based on Householder reflections.

    integrator_traj_tan : symplectic_tangent_step_t
        Symplectic step for the trajectory and tangent dynamics.
    poincare_section : bool
        If True, return CLVs and trajectory sampled on the requested Poincaré section.
    section_index : int
        Index of the coordinate defining the Poincaré section.
    section_value : np.float64
        Value of the section coordinate.
    crossing : int
        Crossing rule for the Poincaré section.
    normalize_A : bool, optional
        Whether to normalize the columns of the backward coefficient matrix.
    eps_norm : np.float64, optional
        Small cutoff used in column normalization.
    rcond_guard : np.float64, optional
        Small cutoff used in the triangular solves.

    Returns
    -------
    tuple[NDArray[np.float64], NDArray[np.float64]]
        - `clvs`: array of shape `(n_samples, 2 * dof, num_clvs)`
        - `traj`: array of shape `(n_samples, 2 * dof + 1)` with time in the
          first column, coordinates next, and momenta last
    """
    q = q.copy()
    p = p.copy()

    dof = q.shape[0]
    neq = 2 * dof

    num_steps = round(total_time / time_step)
    qr_steps = round(qr_time_step / time_step)
    total_blocks = num_steps // qr_steps

    warm_blocks = 0
    if warmup_time > np.float64(0.0):
        warm_blocks = round(warmup_time / qr_time_step)

    tail_blocks = 0
    if tail_time > np.float64(0.0):
        tail_blocks = round(tail_time / qr_time_step)

    np.random.seed(seed)
    Q = -np.float64(1.0) + np.float64(2.0) * np.random.rand(neq, num_clvs)

    if method == "QR":
        Q, _ = qr(Q)
    elif method == "QR_HH":
        Q, _ = np.linalg.qr(Q)
    else:
        raise ValueError("method must be 'QR' or 'QR_HH'")

    Q = np.ascontiguousarray(Q[:, :num_clvs])

    for _ in range(warm_blocks):
        q, p, Q = advance_block_sep(
            q,
            p,
            Q,
            qr_steps,
            time_step,
            grad_T,
            grad_V,
            hess_T,
            hess_V,
            parameters,
            integrator_traj_tan,
        )

        if method == "QR":
            Q, _ = qr(Q)
        else:
            Q, _ = np.linalg.qr(Q)

        Q = np.ascontiguousarray(Q[:, :num_clvs])

    time = np.float64(warm_blocks * qr_steps) * time_step

    Q_store = np.zeros((total_blocks + 1, neq, num_clvs), dtype=np.float64)
    R_store = np.zeros((total_blocks, num_clvs, num_clvs), dtype=np.float64)
    times = np.zeros(total_blocks + 1, dtype=np.float64)
    q_history = np.zeros((total_blocks + 1, dof), dtype=np.float64)
    p_history = np.zeros((total_blocks + 1, dof), dtype=np.float64)

    Q_store[0] = Q
    times[0] = time
    q_history[0] = q
    p_history[0] = p

    for blk in range(total_blocks):
        q, p, Q = advance_block_sep(
            q,
            p,
            Q,
            qr_steps,
            time_step,
            grad_T,
            grad_V,
            hess_T,
            hess_V,
            parameters,
            integrator_traj_tan,
        )
        time += np.float64(qr_steps) * time_step

        if method == "QR":
            Q, R = qr_truncate(Q, num_clvs, qr)
        else:
            Q, R = qr_truncate(Q, num_clvs, np.linalg.qr)

        Q = np.ascontiguousarray(Q)
        R = np.ascontiguousarray(R)

        Q_store[blk + 1] = Q
        R_store[blk] = R
        times[blk + 1] = time
        q_history[blk + 1] = q
        p_history[blk + 1] = p

    R_tail = np.zeros((tail_blocks, num_clvs, num_clvs), dtype=np.float64)

    for blk in range(tail_blocks):
        q, p, Q = advance_block_sep(
            q,
            p,
            Q,
            qr_steps,
            time_step,
            grad_T,
            grad_V,
            hess_T,
            hess_V,
            parameters,
            integrator_traj_tan,
        )

        if method == "QR":
            Q, R = qr_truncate(Q, num_clvs, qr)
        else:
            Q, R = qr_truncate(Q, num_clvs, np.linalg.qr)

        Q = np.ascontiguousarray(Q)
        R = np.ascontiguousarray(R)
        R_tail[blk] = R

    np.random.seed(seed)
    A = np.triu(np.random.randn(num_clvs, num_clvs)).astype(np.float64)

    for k in range(tail_blocks - 1, -1, -1):
        if normalize_A:
            clv_col_normalize_inplace(A, eps_norm)
        clv_solve_upper_inplace(R_tail[k], A, rcond_guard)

    clvs = np.zeros((total_blocks + 1, neq, num_clvs), dtype=np.float64)

    for k in range(total_blocks, -1, -1):
        if normalize_A:
            clv_col_normalize_inplace(A, eps_norm)

        V = Q_store[k] @ A
        clv_col_normalize_inplace(V, eps_norm)
        clvs[k] = V

        if k > 0:
            clv_solve_upper_inplace(R_store[k - 1], A, rcond_guard)

    traj_size = total_blocks + 1

    if poincare_section:
        section_points, section_k = generate_poincare_section_from_traj_sep(
            q_history,
            p_history,
            parameters,
            grad_T,
            qr_time_step,
            section_index,
            section_value,
            crossing,
        )
        times = section_points[:, 0]
        q_history = section_points[:, 1 : dof + 1]
        p_history = section_points[:, dof + 1 :]
        traj_size = times.shape[0]
        clvs = clvs[section_k]

    traj = np.zeros((traj_size, 2 * dof + 1), dtype=np.float64)
    traj[:, 0] = times
    traj[:, 1 : dof + 1] = q_history
    traj[:, dof + 1 :] = p_history

    return clvs, traj


def clv_angles_sep(
    q: NDArray[np.float64],
    p: NDArray[np.float64],
    total_time: np.float64,
    time_step: np.float64,
    parameters: NDArray[np.float64],
    grad_T: system_func_t,
    grad_V: system_func_t,
    hess_T: system_func_t,
    hess_V: system_func_t,
    warmup_time: np.float64,
    tail_time: np.float64,
    qr_time_step: np.float64,
    seed: int,
    method: str,
    integrator_traj_tan: symplectic_tangent_step_t,
    poincare_section: bool,
    section_index: int,
    section_value: np.float64,
    crossing: int,
    subspaces: Optional[Sequence[tuple[Sequence[int], Sequence[int]]]] = None,
    pairs: Optional[Sequence[tuple[int, int]]] = None,
    normalize_A: bool = True,
    eps_norm: np.float64 = np.float64(1e-300),
    rcond_guard: np.float64 = np.float64(1e-14),
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute CLV-based angle diagnostics for a separable Hamiltonian
    system, H(q, p) = T(p) + V(q), integrated with an explicit symplectic
    stepper (velocity Verlet or fourth-order Yoshida).

    Requires grad_T, grad_V, hess_V, and hess_T to advance the trajectory
    and tangent vectors.

    Parameters
    ----------
    q : NDArray[np.float64]
        Initial generalized coordinates.
    p : NDArray[np.float64]
        Initial generalized momenta.
    total_time : np.float64
        Total integration time over the storage window.
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
    warmup_time : np.float64
        Forward warmup time used before storing QR factors.
    tail_time : np.float64
        Extra forward time used to initialize the backward recursion.
    qr_time_step : np.float64
        Time interval between successive QR factorizations.
    seed : int
        Random seed used to initialize the tangent basis and the backward matrix.
    method : str
        QR method used in the orthonormalization step. Must be `"QR"` or `"QR_HH"`.
    integrator_traj_tan : symplectic_tangent_step_t
        Symplectic step for the trajectory and tangent dynamics.
    poincare_section : bool
        If True, return CLV angles sampled on the requested Poincaré section.
    section_index : int
        Index of the coordinate defining the Poincaré section.
    section_value : np.float64
        Value of the section coordinate.
    crossing : int
        Crossing rule for the Poincaré section.
    subspaces : sequence of tuple[Sequence[int], Sequence[int]] or None, optional
        Subspace pairs used to compute minimum principal angles.
    pairs : sequence of tuple[int, int] or None, optional
        CLV index pairs used to compute pairwise angles.
    normalize_A : bool, optional
        Whether to normalize the columns of the backward coefficient matrix.
    eps_norm : np.float64, optional
        Small cutoff used in column normalization.
    rcond_guard : np.float64, optional
        Small cutoff used in the triangular solves.

    Returns
    -------
    tuple[NDArray[np.float64], NDArray[np.float64]]
        - `angles`: array of shape `(n_samples, n_angles)`
        - `traj`: trajectory array associated with the computed angles
    """
    want_subspaces = subspaces is not None and len(subspaces) > 0
    want_pairs = pairs is not None and len(pairs) > 0

    if not want_subspaces and not want_pairs:
        raise ValueError("At least one of `subspaces` or `pairs` must be provided.")

    dof = len(q)
    dim = 2 * dof

    clvs, traj = compute_clvs_sep(
        q=q,
        p=p,
        total_time=total_time,
        time_step=time_step,
        parameters=parameters,
        grad_T=grad_T,
        grad_V=grad_V,
        hess_T=hess_T,
        hess_V=hess_V,
        num_clvs=dim,
        warmup_time=warmup_time,
        tail_time=tail_time,
        qr_time_step=qr_time_step,
        seed=seed,
        method=method,
        integrator_traj_tan=integrator_traj_tan,
        poincare_section=poincare_section,
        section_index=section_index,
        section_value=section_value,
        crossing=crossing,
        normalize_A=normalize_A,
        eps_norm=eps_norm,
        rcond_guard=rcond_guard,
    )

    T, dim, _ = clvs.shape
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
                sigma_max = np.abs(sigma_max)
                sigma_max = np.clip(sigma_max, -1.0, 1.0)
                angles[t, col] = np.arccos(sigma_max)
            col += 1

    if want_pairs:
        assert pairs is not None
        for i, j in pairs:
            dots = np.einsum("td,td->t", V[:, :, i], V[:, :, j])
            dots = np.abs(dots)
            dots = np.clip(dots, -1.0, 1.0)
            angles[:, col] = np.arccos(dots)
            col += 1

    return angles, traj


@njit(error_model="numpy")
def compute_clvs_imp(
    q: NDArray[np.float64],
    p: NDArray[np.float64],
    total_time: np.float64,
    time_step: np.float64,
    parameters: NDArray[np.float64],
    eom: system_func_t,
    hess_H: system_func_t,
    num_clvs: int,
    warmup_time: np.float64,
    tail_time: np.float64,
    qr_time_step: np.float64,
    seed: int,
    method: str,
    tol: np.float64,
    max_iter: int,
    integrator_traj_tan: symplectic_tangent_step_t,
    poincare_section: bool,
    section_index: int,
    section_value: np.float64,
    crossing: int,
    normalize_A: bool = True,
    eps_norm: np.float64 = np.float64(1e-300),
    rcond_guard: np.float64 = np.float64(1e-14),
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute covariant Lyapunov vectors (CLVs) for a general (possible
    non-separable) Hamiltonian system H(q, p), integrated with the
    implicit midpoint method

    Requires eom and hess_H to advance the trajectory and tangent vectors.

    Parameters
    ----------
    q : NDArray[np.float64]
        Initial generalized coordinates of shape `(dof,)`.
    p : NDArray[np.float64]
        Initial generalized momenta of shape `(dof,)`.
    total_time : np.float64
        Total integration time over the storage window.
    time_step : np.float64
        Integration time step.
    parameters : NDArray[np.float64]
        Additional system parameters.
    eom : system_func_t
        Equations of motion of the system.
    hess_H : system_func_t
        Hessian of the Hamiltonian w.r.t. z = (q, p).
    num_clvs : int
        Number of CLVs to compute.
    warmup_time : np.float64
        Forward warmup time used before storing QR factors.
    tail_time : np.float64
        Extra forward time used to initialize the backward recursion.
    qr_time_step : np.float64
        Time interval between successive QR factorizations.
    seed : int
        Random seed used to initialize the tangent basis and the backward matrix.
    method : str
        QR method used in the orthonormalization step. Supported options are:

        - `"QR"`:
          Use the internal reduced modified Gram-Schmidt QR routine `qr`.

        - `"QR_HH"`:
          Use `numpy.linalg.qr`, based on Householder reflections.
    tol : np.float64
        Newton convergence tolerance on the residual norm.
    max_iter : int
        Maximum Newton iterations per step.
    integrator_traj_tan : symplectic_tangent_step_t
        Symplectic step for the trajectory and tangent dynamics.
    poincare_section : bool
        If True, return CLVs and trajectory sampled on the requested Poincaré section.
    section_index : int
        Index of the coordinate defining the Poincaré section.
    section_value : np.float64
        Value of the section coordinate.
    crossing : int
        Crossing rule for the Poincaré section.
    normalize_A : bool, optional
        Whether to normalize the columns of the backward coefficient matrix.
    eps_norm : np.float64, optional
        Small cutoff used in column normalization.
    rcond_guard : np.float64, optional
        Small cutoff used in the triangular solves.

    Returns
    -------
    tuple[NDArray[np.float64], NDArray[np.float64]]
        - `clvs`: array of shape `(n_samples, 2 * dof, num_clvs)`
        - `traj`: array of shape `(n_samples, 2 * dof + 1)` with time in the
          first column, coordinates next, and momenta last
    """
    q = q.copy()
    p = p.copy()

    dof = q.shape[0]
    neq = 2 * dof

    num_steps = round(total_time / time_step)
    qr_steps = round(qr_time_step / time_step)
    total_blocks = num_steps // qr_steps

    warm_blocks = 0
    if warmup_time > np.float64(0.0):
        warm_blocks = round(warmup_time / qr_time_step)

    tail_blocks = 0
    if tail_time > np.float64(0.0):
        tail_blocks = round(tail_time / qr_time_step)

    np.random.seed(seed)
    Q = -np.float64(1.0) + np.float64(2.0) * np.random.rand(neq, num_clvs)

    if method == "QR":
        Q, _ = qr(Q)
    elif method == "QR_HH":
        Q, _ = np.linalg.qr(Q)
    else:
        raise ValueError("method must be 'QR' or 'QR_HH'")

    Q = np.ascontiguousarray(Q[:, :num_clvs])

    for _ in range(warm_blocks):
        q, p, Q = advance_block_imp(
            q,
            p,
            Q,
            qr_steps,
            time_step,
            eom,
            hess_H,
            parameters,
            tol,
            max_iter,
            integrator_traj_tan,
        )

        if method == "QR":
            Q, _ = qr(Q)
        else:
            Q, _ = np.linalg.qr(Q)

        Q = np.ascontiguousarray(Q[:, :num_clvs])

    time = np.float64(warm_blocks * qr_steps) * time_step

    Q_store = np.zeros((total_blocks + 1, neq, num_clvs), dtype=np.float64)
    R_store = np.zeros((total_blocks, num_clvs, num_clvs), dtype=np.float64)
    times = np.zeros(total_blocks + 1, dtype=np.float64)
    q_history = np.zeros((total_blocks + 1, dof), dtype=np.float64)
    p_history = np.zeros((total_blocks + 1, dof), dtype=np.float64)

    Q_store[0] = Q
    times[0] = time
    q_history[0] = q
    p_history[0] = p

    for blk in range(total_blocks):
        q, p, Q = advance_block_imp(
            q,
            p,
            Q,
            qr_steps,
            time_step,
            eom,
            hess_H,
            parameters,
            tol,
            max_iter,
            integrator_traj_tan,
        )
        time += np.float64(qr_steps) * time_step

        if method == "QR":
            Q, R = qr_truncate(Q, num_clvs, qr)
        else:
            Q, R = qr_truncate(Q, num_clvs, np.linalg.qr)

        Q = np.ascontiguousarray(Q)
        R = np.ascontiguousarray(R)

        Q_store[blk + 1] = Q
        R_store[blk] = R
        times[blk + 1] = time
        q_history[blk + 1] = q
        p_history[blk + 1] = p

    R_tail = np.zeros((tail_blocks, num_clvs, num_clvs), dtype=np.float64)

    for blk in range(tail_blocks):
        q, p, Q = advance_block_imp(
            q,
            p,
            Q,
            qr_steps,
            time_step,
            eom,
            hess_H,
            parameters,
            tol,
            max_iter,
            integrator_traj_tan,
        )

        if method == "QR":
            Q, R = qr_truncate(Q, num_clvs, qr)
        else:
            Q, R = qr_truncate(Q, num_clvs, np.linalg.qr)

        Q = np.ascontiguousarray(Q)
        R = np.ascontiguousarray(R)
        R_tail[blk] = R

    np.random.seed(seed)
    A = np.triu(np.random.randn(num_clvs, num_clvs)).astype(np.float64)

    for k in range(tail_blocks - 1, -1, -1):
        if normalize_A:
            clv_col_normalize_inplace(A, eps_norm)
        clv_solve_upper_inplace(R_tail[k], A, rcond_guard)

    clvs = np.zeros((total_blocks + 1, neq, num_clvs), dtype=np.float64)

    for k in range(total_blocks, -1, -1):
        if normalize_A:
            clv_col_normalize_inplace(A, eps_norm)

        V = Q_store[k] @ A
        clv_col_normalize_inplace(V, eps_norm)
        clvs[k] = V

        if k > 0:
            clv_solve_upper_inplace(R_store[k - 1], A, rcond_guard)

    traj_size = total_blocks + 1

    if poincare_section:
        section_points, section_k = generate_poincare_section_from_traj_imp(
            q_history,
            p_history,
            parameters,
            eom,
            qr_time_step,
            section_index,
            section_value,
            crossing,
        )
        times = section_points[:, 0]
        q_history = section_points[:, 1 : dof + 1]
        p_history = section_points[:, dof + 1 :]
        traj_size = times.shape[0]
        clvs = clvs[section_k]

    traj = np.zeros((traj_size, 2 * dof + 1), dtype=np.float64)
    traj[:, 0] = times
    traj[:, 1 : dof + 1] = q_history
    traj[:, dof + 1 :] = p_history

    return clvs, traj


def clv_angles_imp(
    q: NDArray[np.float64],
    p: NDArray[np.float64],
    total_time: np.float64,
    time_step: np.float64,
    parameters: NDArray[np.float64],
    eom: system_func_t,
    hess_H: system_func_t,
    warmup_time: np.float64,
    tail_time: np.float64,
    qr_time_step: np.float64,
    seed: int,
    method: str,
    tol: np.float64,
    max_iter: int,
    integrator_traj_tan: symplectic_tangent_step_t,
    poincare_section: bool,
    section_index: int,
    section_value: np.float64,
    crossing: int,
    subspaces: Optional[Sequence[tuple[Sequence[int], Sequence[int]]]] = None,
    pairs: Optional[Sequence[tuple[int, int]]] = None,
    normalize_A: bool = True,
    eps_norm: np.float64 = np.float64(1e-300),
    rcond_guard: np.float64 = np.float64(1e-14),
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute CLV-based angle diagnostics for a separable Hamiltonian
    system, H(q, p) = T(p) + V(q), integrated with an explicit symplectic
    stepper (velocity Verlet or fourth-order Yoshida).

    Requires grad_T, grad_V, hess_V, and hess_T to advance the trajectory
    and tangent vectors.

    Parameters
    ----------
    q : NDArray[np.float64]
        Initial generalized coordinates.
    p : NDArray[np.float64]
        Initial generalized momenta.
    total_time : np.float64
        Total integration time over the storage window.
    time_step : np.float64
        Integration time step.
    parameters : NDArray[np.float64]
        Additional system parameters.
    eom : system_func_t
        Equations of motion of the system.
    hess_H : system_func_t
        Hessian of the Hamiltonian w.r.t. z = (q, p).
    warmup_time : np.float64
        Forward warmup time used before storing QR factors.
    tail_time : np.float64
        Extra forward time used to initialize the backward recursion.
    qr_time_step : np.float64
        Time interval between successive QR factorizations.
    seed : int
        Random seed used to initialize the tangent basis and the backward matrix.
    method : str
        QR method used in the orthonormalization step. Must be `"QR"` or `"QR_HH"`.
    tol : np.float64
        Newton convergence tolerance on the residual norm.
    max_iter : int
        Maximum Newton iterations per step.
    integrator_traj_tan : symplectic_tangent_step_t
        Symplectic step for the trajectory and tangent dynamics.
    poincare_section : bool
        If True, return CLV angles sampled on the requested Poincaré section.
    section_index : int
        Index of the coordinate defining the Poincaré section.
    section_value : np.float64
        Value of the section coordinate.
    crossing : int
        Crossing rule for the Poincaré section.
    subspaces : sequence of tuple[Sequence[int], Sequence[int]] or None, optional
        Subspace pairs used to compute minimum principal angles.
    pairs : sequence of tuple[int, int] or None, optional
        CLV index pairs used to compute pairwise angles.
    normalize_A : bool, optional
        Whether to normalize the columns of the backward coefficient matrix.
    eps_norm : np.float64, optional
        Small cutoff used in column normalization.
    rcond_guard : np.float64, optional
        Small cutoff used in the triangular solves.

    Returns
    -------
    tuple[NDArray[np.float64], NDArray[np.float64]]
        - `angles`: array of shape `(n_samples, n_angles)`
        - `traj`: trajectory array associated with the computed angles
    """
    want_subspaces = subspaces is not None and len(subspaces) > 0
    want_pairs = pairs is not None and len(pairs) > 0

    if not want_subspaces and not want_pairs:
        raise ValueError("At least one of `subspaces` or `pairs` must be provided.")

    dof = len(q)
    dim = 2 * dof

    clvs, traj = compute_clvs_imp(
        q=q,
        p=p,
        total_time=total_time,
        time_step=time_step,
        parameters=parameters,
        eom=eom,
        hess_H=hess_H,
        num_clvs=dim,
        warmup_time=warmup_time,
        tail_time=tail_time,
        qr_time_step=qr_time_step,
        seed=seed,
        method=method,
        tol=tol,
        max_iter=max_iter,
        integrator_traj_tan=integrator_traj_tan,
        poincare_section=poincare_section,
        section_index=section_index,
        section_value=section_value,
        crossing=crossing,
        normalize_A=normalize_A,
        eps_norm=eps_norm,
        rcond_guard=rcond_guard,
    )

    T, dim, _ = clvs.shape
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
                sigma_max = np.abs(sigma_max)
                sigma_max = np.clip(sigma_max, -1.0, 1.0)
                angles[t, col] = np.arccos(sigma_max)
            col += 1

    if want_pairs:
        assert pairs is not None
        for i, j in pairs:
            dots = np.einsum("td,td->t", V[:, :, i], V[:, :, j])
            dots = np.abs(dots)
            dots = np.clip(dots, -1.0, 1.0)
            angles[:, col] = np.arccos(dots)
            col += 1

    return angles, traj
