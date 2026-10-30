# buffers.py

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

# Number of rows allocated for a new buffer
_INITIAL_ROWS = 64


@njit
def new_buffer(num_columns: int) -> NDArray[np.float64]:
    """
    Allocate an empty buffer of rows, to be filled with `ensure_row`.

    The number of rows is not known in advance with an adaptive integrator,
    so the buffer grows geometrically, which costs far less than appending
    each row to a list of lists.
    """
    return np.empty((_INITIAL_ROWS, num_columns), dtype=np.float64)


@njit
def ensure_row(buffer: NDArray[np.float64], count: int) -> NDArray[np.float64]:
    """
    Return a buffer with room for row `count`, doubling it when it is full.

    The first `count` rows are kept.
    """
    if count < buffer.shape[0]:
        return buffer
    grown = np.empty((2 * buffer.shape[0], buffer.shape[1]), dtype=np.float64)
    # An explicit loop compiles much faster than a 2D slice assignment.
    for i in range(count):
        for j in range(buffer.shape[1]):
            grown[i, j] = buffer[i, j]
    return grown
