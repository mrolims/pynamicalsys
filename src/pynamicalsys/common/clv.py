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


import numpy as np
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
