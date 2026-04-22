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
