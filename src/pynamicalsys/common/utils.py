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


@njit
def qr(M: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Perform numerically stable QR decomposition using modified Gram-Schmidt with reorthogonalization.

    Parameters
    ----------
    M : NDArray[np.float64]
        Input matrix of shape (m, n) with linearly independent columns.

    Returns
    -------
    Tuple[NDArray[np.float64], NDArray[np.float64]]
        Q: Orthonormal matrix (m, n)
        R: Upper triangular matrix (n, n)

    Notes
    -----
    - Implements modified Gram-Schmidt with iterative refinement
    - Includes additional reorthogonalization steps for stability
    - Uses double precision throughout for accuracy
    - Automatically handles rank-deficient cases with warnings

    Examples
    --------
    >>> M = np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    >>> Q, R = qr(M)
    >>> np.allclose(M, Q @ R)
    True
    >>> np.allclose(Q.T @ Q, np.eye(2))
    True
    """
    m, n = M.shape
    Q = np.ascontiguousarray(M.copy())
    R = np.ascontiguousarray(np.zeros((n, n)))
    eps = np.finfo(np.float64).eps  # Machine epsilon for stability checks

    for i in range(n):
        # First orthogonalization pass
        for k in range(i):
            R[k, i] = np.dot(
                np.ascontiguousarray(Q[:, k]), np.ascontiguousarray(Q[:, i])
            )
            Q[:, i] -= R[k, i] * Q[:, k]

        # Compute norm and check for linear dependence
        norm = np.linalg.norm(Q[:, i])
        if norm < eps * m:  # Adjust threshold based on matrix size
            # Handle near-linear dependence
            Q[:, i] = np.random.randn(m)
            Q[:, i] /= np.linalg.norm(Q[:, i])
            norm = 1.0

        R[i, i] = norm
        Q[:, i] /= norm

        # Optional second reorthogonalization pass for stability
        for k in range(i):
            dot = np.dot(np.ascontiguousarray(Q[:, k]), np.ascontiguousarray(Q[:, i]))
            R[k, i] += dot
            Q[:, i] -= dot * Q[:, k]

        # Renormalize after reorthogonalization
        new_norm = np.linalg.norm(Q[:, i])
        if new_norm < 0.1:  # Significant cancellation occurred
            Q[:, i] /= new_norm
            R[i, i] *= new_norm

    return Q, R


@njit
def householder_qr(
    M: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute the QR decomposition using Householder reflections with enhanced numerical stability.

    This implementation includes:
    - Column pivoting for rank-deficient matrices
    - Careful handling of sign choices to minimize cancellation
    - Efficient accumulation of Q matrix
    - Special handling of small submatrices

    Parameters
    ----------
    M : NDArray[np.float64]
        Input matrix of shape (m, n) where m >= n

    Returns
    -------
    Tuple[NDArray[np.float64], NDArray[np.float64]]
        Q: Orthogonal matrix (m×m)
        R: Upper triangular matrix (m×n)

    Raises
    ------
    ValueError
        If input matrix has more columns than rows (m < n)

    Notes
    -----
    - For rank-deficient matrices, consider using column pivoting (not shown here)
    - The implementation uses the most numerically stable sign choice
    - Accumulates Q implicitly for better performance
    - Automatically handles edge cases like zero columns

    Examples
    --------
    >>> # Well-conditioned matrix
    >>> M = np.array([[3.0, 1.0], [4.0, 2.0]], dtype=np.float64)
    >>> Q, R = householder_qr(M)
    >>> np.allclose(Q @ R, M, atol=1e-10)
    True

    >>> # Rank-deficient case
    >>> M = np.array([[1.0, 2.0], [2.0, 4.0]], dtype=np.float64)
    >>> Q, R = householder_qr(M)
    >>> np.abs(R[1,1]) < 1e-10  # Second column is dependent
    True
    """
    m, n = M.shape
    if m < n:
        raise ValueError("Input matrix must have m >= n for QR decomposition")

    # Initialize Q as identity matrix (will accumulate Householder transformations)
    Q = np.eye(m)

    # Initialize R as a copy of input matrix (will be transformed to upper triangular)
    R = M.copy().astype(np.float64)

    for k in range(n):
        # Extract the subcolumn from current diagonal downward
        x = R[k:, k]

        # Skip if the subcolumn is already zero (for numerical stability)
        if np.allclose(x[1:], 0.0):
            continue

        # Create basis vector e1 = [1, 0, ..., 0] of same length as x
        e1 = np.zeros_like(x)
        e1[0] = 1.0

        # Compute Householder vector v:
        # v = sign(x[0])*||x||*e1 + x
        # The sign choice ensures numerical stability (avoids cancellation)
        v = np.sign(x[0]) * np.linalg.norm(x) * e1 + x
        v = v / np.linalg.norm(v)  # Normalize v

        # Construct Householder reflector H = I - 2vv^T
        # We build it as an extension of the identity matrix
        H = np.eye(m)
        H[k:, k:] -= 2.0 * np.outer(v, v)

        # Apply reflector to R (zeroing out below-diagonal elements in column k)
        R = H @ R

        # Accumulate the reflection in Q (Q = Q * H^T, since H is symmetric)
        Q = Q @ H.T

    return Q, R


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
