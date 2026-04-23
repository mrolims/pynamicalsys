# symmetry.py

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

from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from pynamicalsys.common.types import int_t


def generate_symmetry_points(
    array: NDArray[np.float64],
    func: Callable[..., NDArray[np.float64]],
    axis: int_t,
    *args: Any,
    **kwargs: Any,
) -> NDArray[np.float64]:
    """
    Generate phase-space points along a symmetry line or symmetry curve.

    Parameters
    ----------
    array : NDArray[np.float64]
        One-dimensional array used to parametrize the symmetry line.
        If `axis == 0`, `array` is interpreted as x-values and the function
        generates points on `y = f(x)`.
        If `axis == 1`, `array` is interpreted as y-values and the function
        generates points on `x = g(y)`.
    func : Callable[..., NDArray[np.float64]]
        Callable that defines the symmetry line or curve.
        It must accept the coordinate array as its first argument and return
        a one-dimensional array with the same length.
    axis : int_t
        Symmetry-line parametrization axis:
        - `0` for `y = f(x)`
        - `1` for `x = g(y)`
    *args : Any
        Additional positional arguments passed to `func`.
    **kwargs : Any
        Additional keyword arguments passed to `func`.

    Returns
    -------
    NDArray[np.float64]
        Array of shape `(n_points, 2)` containing the generated points.

    Notes
    -----
    Validation is handled in the wrapper.
    """
    if axis == 0:
        x_array = array.copy()
        y_array = np.asarray(func(x_array, *args, **kwargs), dtype=np.float64)
    else:
        y_array = array.copy()
        x_array = np.asarray(func(y_array, *args, **kwargs), dtype=np.float64)

    return np.column_stack((x_array, y_array))
