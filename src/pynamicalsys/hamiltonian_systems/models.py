# models.py

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
from numba import njit
from numpy.typing import NDArray
from typing import Union, Sequence


@njit
def henon_heiles_grad_T(
    p: NDArray[np.float64],
    parameters: Union[None, Sequence[float], NDArray[np.float64]] = None,
) -> NDArray[np.float64]:
    """Gradient of T(p)=0.5*(p0^2+p1^2). Returns [dT/dp0, dT/dp1]."""
    p0, p1 = p[0], p[1]
    return np.array([p0, p1])


@njit
def henon_heiles_hess_T(
    p=None,
    parameters: Union[None, Sequence[float], NDArray[np.float64]] = None,
) -> NDArray[np.float64]:
    """Hessian of T (unit-mass) - constant 2x2 identity matrix.
    p argument unused, kept for API symmetry with other functions."""
    return np.array([[1.0, 0.0], [0.0, 1.0]])


@njit
def henon_heiles_grad_V(
    q,
    parameters: Union[None, Sequence[float], NDArray[np.float64]] = None,
) -> NDArray[np.float64]:
    """Gradient of Hénon–Heiles potential V at q = [q0, q1].
    Returns [dV/dq0, dV/dq1]."""
    q0, q1 = q[0], q[1]
    dV_dq0 = q0 * (1.0 + 2.0 * q1)
    dV_dq1 = q1 + q0 * q0 - q1 * q1
    return np.array([dV_dq0, dV_dq1])


@njit
def henon_heiles_hess_V(
    q,
    parameters: Union[None, Sequence[float], NDArray[np.float64]] = None,
) -> NDArray[np.float64]:
    """Hessian of Hénon–Heiles potential V at q = [q0, q1].
    Returns a 2x2 nested list [[H00, H01], [H10, H11]]."""
    q0, q1 = q[0], q[1]
    H00 = 1.0 + 2.0 * q1
    H01 = 2.0 * q0
    H11 = 1.0 - 2.0 * q1
    return np.array([[H00, H01], [H01, H11]])
