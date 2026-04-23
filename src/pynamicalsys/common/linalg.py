# linalg.py

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

import numpy as np
from typing import Callable, Tuple
from numpy.typing import NDArray
from numba import njit


@njit
def qr(
    M: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute the reduced QR decomposition using modified Gram-Schmidt.

    Parameters
    ----------
    M : NDArray[np.float64]
        Input matrix of shape `(m, n)` with `m >= n`.

    Returns
    -------
    Tuple[NDArray[np.float64], NDArray[np.float64]]
        - `Q`: Orthonormal matrix of shape `(m, n)`
        - `R`: Upper triangular matrix of shape `(n, n)`

    Raises
    ------
    ValueError
        If `m < n`.

    Notes
    -----
    This implementation uses modified Gram-Schmidt, which is more stable than
    classical Gram-Schmidt, but generally less stable than Householder QR.
    """
    m, n = M.shape
    if m < n:
        raise ValueError("Input matrix must have m >= n for QR decomposition")

    Q = M.copy().astype(np.float64)
    R = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(i):
            s = 0.0
            for row in range(m):
                s += Q[row, j] * Q[row, i]
            R[j, i] = s

            for row in range(m):
                Q[row, i] -= R[j, i] * Q[row, j]

        norm_sq = 0.0
        for row in range(m):
            norm_sq += Q[row, i] * Q[row, i]
        R[i, i] = np.sqrt(norm_sq)

        if R[i, i] == 0.0:
            continue

        inv_norm = 1.0 / R[i, i]
        for row in range(m):
            Q[row, i] *= inv_norm

    return Q, R


@njit
def householder_qr(
    M: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    m, n = M.shape
    if m < n:
        raise ValueError("Input matrix must have m >= n for QR decomposition")

    R = M.copy().astype(np.float64)
    Q = np.eye(m, dtype=np.float64)

    for k in range(n):
        x = R[k:, k]
        norm_x = np.linalg.norm(x)

        if norm_x == 0.0:
            continue

        alpha = -np.copysign(norm_x, x[0])
        v = x.copy()
        v[0] -= alpha
        v_norm = np.linalg.norm(v)

        if v_norm == 0.0:
            continue

        v /= v_norm

        # Apply reflector to R[k:, k:]
        R[k:, k:] -= 2.0 * np.outer(v, v @ R[k:, k:])

        # Apply reflector to Q[:, k:]
        Q[:, k:] -= 2.0 * np.outer(Q[:, k:] @ v, v)

    return Q, R


@njit
def qr_truncate(
    Q: NDArray[np.float64],
    k: int,
    QR: Callable[
        [NDArray[np.float64]],
        tuple[NDArray[np.float64], NDArray[np.float64]],
    ],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute a QR decomposition and retain only the first `k` columns of `Q`
    and the leading `k x k` block of `R`.

    Parameters
    ----------
    Q : NDArray[np.float64]
        Input matrix of shape `(m, n)`.
    k : int
        Number of columns and rows to retain after the QR decomposition.
    QR : callable
        QR decomposition routine returning `(Q_full, R_full)`.

    Returns
    -------
    tuple[NDArray[np.float64], NDArray[np.float64]]
        - `Q_trunc`: array of shape `(m, k)` containing the first `k` columns
          of the orthonormal factor
        - `R_trunc`: array of shape `(k, k)` containing the leading block of
          the upper-triangular factor
    """
    Q_full, R_full = QR(Q)
    Q_trunc = np.ascontiguousarray(Q_full[:, :k])
    R_trunc = R_full[:k, :k]
    return Q_trunc, R_trunc
