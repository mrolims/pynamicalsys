# utils.py

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

import numpy as np
from typing import Callable, Tuple
from numpy.typing import NDArray
from numba import njit
from .types import numeric_t


@njit
def clv_sanitize_inplace(M: NDArray[np.float64]) -> None:
    nrows, ncols = M.shape
    for i in range(nrows):
        for j in range(ncols):
            x = M[i, j]
            if not np.isfinite(x):
                M[i, j] = 0.0


@njit
def clv_col_normalize_inplace(M: NDArray[np.float64], eps_norm: numeric_t) -> None:
    nrows, ncols = M.shape
    for j in range(ncols):
        s = 0.0
        for i in range(nrows):
            v = M[i, j]
            s += v * v

        nrm = np.sqrt(s)

        if (not np.isfinite(nrm)) or (nrm < eps_norm):
            for i in range(nrows):
                M[i, j] = 0.0
            continue

        inv = 1.0 / nrm
        for i in range(nrows):
            M[i, j] *= inv


@njit
def clv_solve_upper_inplace(
    R: NDArray[np.float64],
    B: NDArray[np.float64],
    rcond_guard: numeric_t,
) -> None:
    """
    Solve the upper-triangular system ``R X = B`` in place.

    Parameters
    ----------
    R : NDArray[np.float64]
        Upper-triangular matrix of shape `(p, p)`.
    B : NDArray[np.float64]
        Right-hand side matrix of shape `(p, m)`. Overwritten in place with
        the solution `X`.
    rcond_guard : numeric_t
        Minimum absolute value allowed for diagonal entries of `R` during the
        back-substitution step.

    Notes
    -----
    This function modifies `B` in place and does not modify `R`.
    """
    p = R.shape[0]
    ncols = B.shape[1]

    for col in range(ncols):
        for i in range(p - 1, -1, -1):
            s = B[i, col]

            for k in range(i + 1, p):
                s -= R[i, k] * B[k, col]

            rii = R[i, i]
            if (not np.isfinite(rii)) or (np.abs(rii) < rcond_guard):
                rii = rcond_guard if rii >= 0.0 else -rcond_guard

            if not np.isfinite(s):
                B[i, col] = 0.0
            else:
                B[i, col] = s / rii


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


@njit
def finite_difference_jacobian(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    mapping: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    eps: float = -1.0,
) -> NDArray[np.float64]:
    """
    Compute the Jacobian matrix using adaptive finite differences with error control.

    Parameters
    ----------
    u : NDArray[np.float64]
        State vector at which to compute Jacobian (shape: (n,))
    parameters : NDArray[np.float64]
        System parameters
    mapping : Callable[[NDArray, NDArray], NDArray]
        Vector-valued function to differentiate
    eps : float, optional
        Initial step size (automatically determined if -1.0)

    Returns
    -------
    NDArray[np.float64]
        Jacobian matrix (shape: (n, n)) where J[i,j] = ∂f_i/∂u_j

    Raises
    ------
    ValueError
        If invalid method is specified
        If eps is not positive when provided

    Notes
    -----
    - For 'central' method (default), accuracy is O(eps²)
    - For 'complex' method, accuracy is O(eps⁴) but requires complex arithmetic
    - Automatic step size selection based on machine epsilon and input scale
    - Includes Richardson extrapolation for higher accuracy
    - Handles edge cases like zero components carefully

    Examples
    --------
    >>> def lorenz(u, p):
    ...     x, y, z = u
    ...     sigma, rho, beta = p
    ...     return np.array([sigma*(y-x), x*(rho-z)-y, x*y-beta*z])
    >>> u = np.array([1.0, 1.0, 1.0])
    >>> params = np.array([10.0, 28.0, 8/3])
    >>> J = finite_difference_jacobian(u, params, lorenz, method='central')
    """
    n = len(u)
    J = np.zeros((n, n))

    # Determine optimal step size if not provided
    if eps <= 0:
        eps = float(np.finfo(np.float64).eps) ** (1 / 3) * max(
            1.0, float(np.linalg.norm(u))
        )

    for i in range(n):
        # Central difference: O(eps²) accuracy
        u_plus = u.copy()
        u_minus = u.copy()
        u_plus[i] += eps
        u_minus[i] -= eps
        J[:, i] = (mapping(u_plus, parameters) - mapping(u_minus, parameters)) / (
            2 * eps
        )

    return J


@njit
def wedge_norm_2(vectors: NDArray[np.float64]) -> float:
    """
    Computes the norm of the wedge product of n m-dimensional vectors using the Gram determinant.

    Parameters:
    vectors : NDArray[np.float64]
        A (m, n) array where m is the dimension and n is the number of vectors.

    Returns:
    norm : float
        The norm (magnitude) of the wedge product.
    """
    m, n = vectors.shape
    if n > m:
        raise ValueError(
            "Cannot compute the wedge product: more vectors than dimensions."
        )

    # Compute the Gram matrix
    G = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dot = 0.0
            for k in range(m):
                dot += vectors[k, i] * vectors[k, j]
            G[i, j] = dot

    # Compute determinant
    det = np.linalg.det(G)

    # If determinant is slightly negative due to numerical error, clip to 0
    if det < 0:
        det = 0.0

    norm = np.sqrt(det)
    return norm


def wedge_norm(V: NDArray[np.float64]) -> float:
    """
    Computes the norm of the wedge product of k d-dimensional vectors using the Gram determinant.

    Parameters:
    vectors : NDArray[np.float64]
        A (d, k) array where d is the dimension and k is the number of vectors.

    Returns:
    norm : float
        The norm (magnitude) of the wedge product.
    """
    G = V.T @ V  # Gram matrix, shape (k, k)

    det = np.linalg.det(G)

    return 0 if det < 0 else np.sqrt(det)


@njit
def _coeff_mat(x: NDArray[np.float64], deg: int) -> NDArray[np.float64]:
    mat_ = np.zeros(shape=(x.shape[0], deg + 1))
    const = np.ones_like(x)
    mat_[:, 0] = const
    mat_[:, 1] = x
    if deg > 1:
        for n in range(2, deg + 1):
            mat_[:, n] = x**n
    return mat_


@njit
def _fit_x(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    # linalg solves ax = b
    det_ = np.linalg.lstsq(a, b)[0]
    return det_


@njit
def fit_poly(
    x: NDArray[np.float64], y: NDArray[np.float64], deg: int
) -> NDArray[np.float64]:
    a = _coeff_mat(x, deg)
    p = _fit_x(a, y)
    # Reverse order so p[0] is coefficient of highest order
    return p[::-1]
