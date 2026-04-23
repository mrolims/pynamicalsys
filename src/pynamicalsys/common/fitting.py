# fitting.py

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
from numpy.typing import NDArray
from numba import njit


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
