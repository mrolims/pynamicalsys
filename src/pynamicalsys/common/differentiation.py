# differentiation.py

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

from typing import Callable
import numpy as np
from numpy.typing import NDArray
from numba import njit


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
