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


from typing import Any, Sequence, Tuple
import numpy as np
from numba import njit
from numpy.typing import NDArray

from pynamicalsys.common.types import int_t, numeric_t, map_t, jacobian_t
from pynamicalsys.common.clv import (
    clv_col_normalize_inplace,
    clv_sanitize_inplace,
    clv_solve_upper_inplace,
)


@njit(error_model="numpy")
def compute_clvs(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: int_t,
    mapping: map_t,
    jacobian: jacobian_t,
    num_clvs: int | None = None,
    transient_time: int_t = 0,
    warmup_time: int_t = 0,
    tail_time: int_t = 0,
    seed: int = 13,
    normalize_A: bool = True,
    eps_norm: numeric_t = 1e-300,
    rcond_guard: numeric_t = 1e-14,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute covariant Lyapunov vectors (CLVs) for a discrete dynamical system.

    Parameters
    ----------
    u : NDArray[np.float64]
        Initial condition of shape `(dim,)`.
    parameters : NDArray[np.float64]
        Parameter array passed to `mapping` and `jacobian`.
    total_time : int_t
        Number of forward steps stored in the CLV computation window.
    mapping : map_t
        Function defining the time evolution of the system.
    jacobian : jacobian_t
        Function returning the Jacobian matrix of the map.
    num_clvs : int | None, optional
        Number of CLVs to compute. If None, all `dim` CLVs are computed.
    transient_time : int_t, optional
        Number of initial iterations discarded before the CLV computation.
    warmup_time : int_t, optional
        Number of forward QR warm-up iterations used before storing the
        orthonormal bases.
    tail_time : int_t, optional
        Number of additional forward steps used to initialize the backward
        recursion.
    seed : int, optional
        Seed used to initialize the backward coefficient matrix.
    normalize_A : bool, optional
        If True, normalize the columns of the backward coefficient matrix
        during the recursion.
    eps_norm : numeric_t, optional
        Minimum column norm used in the normalization safeguards.
    rcond_guard : numeric_t, optional
        Minimum absolute value allowed for diagonal entries of the triangular
        factors used in the backward solve.

    Returns
    -------
    Tuple[NDArray[np.float64], NDArray[np.float64]]
        - `clvs`: array of shape `(total_time + 1, dim, num_clvs)` containing
          the CLVs along the stored trajectory
        - `traj`: array of shape `(total_time + 1, dim)` containing the stored
          trajectory

    Notes
    -----
    The computation consists of five stages:

    1. transient evolution
    2. forward QR warm-up
    3. forward storage of orthonormal bases and triangular factors
    4. backward initialization of the coefficient matrix
    5. backward recursion yielding the CLVs

    This function assumes that all input validation has already been performed
    by the wrapper.
    """
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
        J = jacobian(u, parameters, mapping)
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
        J = jacobian(u, parameters, mapping)
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
        J = jacobian(u, parameters, mapping)
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
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: int_t,
    mapping: map_t,
    jacobian: jacobian_t,
    warmup_time: int_t = 0,
    tail_time: int_t = 0,
    seed: int = 13,
    subspaces: Sequence[Tuple[Sequence[int], Sequence[int]]] | None = None,
    pairs: Sequence[Tuple[int, int]] | None = None,
    use_abs: bool = True,
    **clv_kwargs: Any,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute CLV angle diagnostics along a trajectory.

    Parameters
    ----------
    u : NDArray[np.float64]
        Initial condition of shape `(dim,)`.
    parameters : NDArray[np.float64]
        Parameter array passed to `mapping` and `jacobian`.
    total_time : int_t
        Number of time steps used in the CLV computation window.
    mapping : map_t
        Function defining the time evolution of the system.
    jacobian : jacobian_t
        Function returning the Jacobian matrix of the map.
    warmup_time : int_t, optional
        Number of QR warm-up steps passed to `compute_clvs`.
    tail_time : int_t, optional
        Number of tail steps passed to `compute_clvs`.
    seed : int, optional
        Seed passed to `compute_clvs`.
    subspaces : Sequence[Tuple[Sequence[int], Sequence[int]]] | None, optional
        Pairs of CLV index sets defining subspaces whose minimum principal
        angles are to be computed.
    pairs : Sequence[Tuple[int, int]] | None, optional
        Pairs of CLV indices whose mutual angles are to be computed.
    use_abs : bool, optional
        If True, use the absolute value of the cosine before applying
        `arccos`.
    **clv_kwargs : Any
        Additional keyword arguments forwarded to `compute_clvs`.

    Returns
    -------
    Tuple[NDArray[np.float64], NDArray[np.float64]]
        - `angles`: array of shape `(T, M)`, where `T = total_time + 1` and
          `M` is the number of requested angles
        - `traj`: trajectory returned by `compute_clvs`

    Notes
    -----
    The output columns are ordered as follows:
    first all requested subspace angles, then all requested pairwise angles.

    This helper assumes that request validation is handled by the wrapper.
    """
    want_subspaces = subspaces is not None and len(subspaces) > 0
    want_pairs = pairs is not None and len(pairs) > 0

    if not want_subspaces and not want_pairs:
        raise ValueError("At least one of `subspaces` or `pairs` must be provided.")

    subspaces_seq: Sequence[Tuple[Sequence[int], Sequence[int]]] = (
        subspaces if subspaces is not None else ()
    )
    pairs_seq: Sequence[Tuple[int, int]] = pairs if pairs is not None else ()

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

    T, _, _ = clvs.shape

    V = clvs / np.linalg.norm(clvs, axis=1, keepdims=True)

    n_sub = len(subspaces_seq)
    n_pairs = len(pairs_seq)

    angles = np.empty((T, n_sub + n_pairs), dtype=np.float64)

    col = 0

    if n_sub > 0:
        for A_idx, B_idx in subspaces_seq:
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

    if n_pairs > 0:
        for i, j in pairs_seq:
            dots = np.einsum("td,td->t", V[:, :, i], V[:, :, j])
            if use_abs:
                dots = np.abs(dots)
            dots = np.clip(dots, -1.0, 1.0)
            angles[:, col] = np.arccos(dots)
            col += 1

    return angles, traj


def clv_angles(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: int_t,
    mapping: map_t,
    jacobian: jacobian_t,
    subspaces: Sequence[Tuple[Sequence[int], Sequence[int]]] | None = None,
    pairs: Sequence[Tuple[int, int]] | None = None,
    window_time: int_t | None = None,
    transient_time: int_t = 0,
    warmup_time: int_t = 0,
    tail_time: int_t = 0,
    seed: int = 13,
    use_abs: bool = True,
    **clv_kwargs: Any,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute full-time or windowed CLV angle diagnostics.

    Parameters
    ----------
    u : NDArray[np.float64]
        Initial condition of shape `(dim,)`.
    parameters : NDArray[np.float64]
        Parameter array passed to `mapping` and `jacobian`.
    total_time : int_t
        Total number of iterations used in the computation.
    mapping : map_t
        Function defining the time evolution of the system.
    jacobian : jacobian_t
        Function returning the Jacobian matrix of the map.
    subspaces : Sequence[Tuple[Sequence[int], Sequence[int]]] | None, optional
        Pairs of CLV index sets defining subspaces whose minimum principal
        angles are to be computed.
    pairs : Sequence[Tuple[int, int]] | None, optional
        Pairs of CLV indices whose mutual angles are to be computed.
    window_time : int_t | None, optional
        Length of the averaging window. If None, no windowing is performed.
    transient_time : int_t, optional
        Number of initial iterations discarded before the computation.
    warmup_time : int_t, optional
        Number of QR warm-up steps passed to `compute_clvs`.
    tail_time : int_t, optional
        Number of tail steps passed to `compute_clvs`.
    seed : int, optional
        Seed passed to `compute_clvs`.
    use_abs : bool, optional
        If True, use the absolute value of the cosine before applying
        `arccos`.
    **clv_kwargs : Any
        Additional keyword arguments forwarded to `compute_clvs`.

    Returns
    -------
    Tuple[NDArray[np.float64], NDArray[np.float64]]
        If `window_time is None`:
            - `angles`: array of shape `(T, M)`
            - `traj`: trajectory of shape `(T, dim)`

        If `window_time is not None`:
            - `avg_angles`: array of shape `(num_windows, M + 1)`, where the
              first column contains the window center time index
            - `initial_conditions`: array of shape `(num_windows, dim)`

    Notes
    -----
    This function assumes that all validation of requests and time scales is
    performed by the wrapper.
    """

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
