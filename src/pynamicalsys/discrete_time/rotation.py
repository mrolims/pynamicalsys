# rotation.py

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

from pynamicalsys.common.types import map_t, numeric_t, int_t


@njit
def rotation_number(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: int_t,
    mapping: map_t,
    mod: numeric_t = 1.0,
) -> float:
    """
    Compute the rotation number of a trajectory.

    The rotation number is estimated as the time average of the wrapped increment
    of the first coordinate,
    ``[(x_{n+1} - x_n) mod mod]``,
    along a trajectory of length `total_time`.

    Parameters
    ----------
    u : NDArray[np.float64]
        Initial condition of shape `(system_dimension,)`.
    parameters : NDArray[np.float64]
        System parameters passed to the mapping function.
    total_time : time_t
        Number of iterations used in the average.
    mapping : map_t
        System mapping function.
    mod : numeric_t, optional
        Period used to wrap the increment of the first coordinate.
        The default is `1.0`.

    Returns
    -------
    float
        Estimated rotation number.

    Notes
    -----
    This low-level routine assumes all inputs were already validated by the wrapper.
    In particular:
    - `u` must be a 1D array,
    - `total_time` must be positive,
    - `mod` must be positive.
    """
    state_old = u.copy()
    rotation = 0.0

    for _ in range(total_time):
        state_new = mapping(state_old, parameters)
        rotation += (state_new[0] - state_old[0]) % mod
        state_old = state_new

    return float(rotation / total_time)
