# models.py

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


@njit
def henon_heiles_eom(
    q: NDArray[np.float64],
    p: NDArray[np.float64],
    parameters: Union[None, Sequence[float], NDArray[np.float64]] = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Equations of motion for the Hénon-Heiles Hamiltonian.
    Returns (dq/dt, dp/dt) = (grad_T(p), -grad_V(q)).
    """
    qdot = henon_heiles_grad_T(p, parameters)
    pdot = -henon_heiles_grad_V(q, parameters)
    return qdot, pdot


@njit
def henon_heiles_hess_H(
    q: NDArray[np.float64],
    p: NDArray[np.float64],
    parameters: Union[None, Sequence[float], NDArray[np.float64]] = None,
) -> NDArray[np.float64]:
    """
    Full Hessian of the Hénon-Heiles Hamiltonian H(q,p) = T(p) + V(q)
    w.r.t. the combined state z = (q, p), shape (4, 4):

        [[ Hqq,  0  ],
         [  0,  Hpp ]]

    Since H is separable here, the q-p cross block is identically zero.
    """
    Hqq = henon_heiles_hess_V(q, parameters)
    Hpp = henon_heiles_hess_T(p, parameters)

    n = q.shape[0]
    H = np.zeros((2 * n, 2 * n))
    H[:n, :n] = Hqq
    H[n:, n:] = Hpp
    return H
