# validators.py

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
from numpy.typing import NDArray


def validate_finite_time(finite_time, total_time) -> None:
    if finite_time > total_time // 2:
        raise ValueError(f"finite_time must be less than or equal to {total_time // 2}")


def validate_and_convert_param_range(param_range) -> NDArray[np.float64]:
    """
    Validate and convert `param_range` input to a 1D numpy array.

    Accepts either:
    - A tuple (start, stop, num_points) for linspace generation
    - A 1D array-like of precomputed values

    Returns
    -------
    param_values : np.ndarray
        1D array of parameter values.

    Raises
    ------
    ValueError
        If `param_range` is a tuple but does not have exactly 3 elements.
        If `param_range` is a tuple but elements are not numbers or if the
        precomputed array is not 1D.
        If `param_range` cannot be converted to a 1D numpy array.
        If `param_range` is a tuple but the elements are not numbers.
    TypeError
        If `param_range` is not a tuple or a 1D array-like.

    """
    if isinstance(param_range, tuple):
        if len(param_range) != 3:
            raise ValueError("param_range tuple must have (start, stop, num_points)")
        try:
            param_values = np.linspace(*param_range)
        except TypeError as e:
            raise ValueError("param_range tuple elements must be numbers") from e
    else:
        try:
            param_values = np.asarray(param_range, dtype=np.float64)
            if param_values.ndim != 1:
                raise ValueError("Precomputed param_range must be 1D array")
        except (TypeError, ValueError) as e:
            raise TypeError(
                "param_range must be a 1D array-like or a (start, stop, num_points) tuple"
            ) from e

    return param_values


def validate_sample_times(sample_times, total_time):
    """
    Validate and convert sample_times to a 1D int32 array within [0, total_time].

    Parameters
    ----------
    sample_times : array-like or None
        Optional time indices to sample from, must be non-negative and ≤ total_time.
    total_time : int
        Maximum valid time index.

    Returns
    -------
    sample_times_arr : np.ndarray or None
        Validated 1D array of sample times, or None if input was None.

    Raises
    ------
    TypeError
        If sample_times is not convertible to a 1D int32 array.
    ValueError
        If sample_times is not 1D or contains out-of-bound values.
    """
    if sample_times is None:
        return None

    try:
        sample_times_arr = np.asarray(sample_times, dtype=np.int64)
        if sample_times_arr.ndim != 1:
            raise ValueError("sample_times must be a 1D array")
        if np.any(sample_times_arr < 0) or np.any(sample_times_arr > total_time):
            raise ValueError("sample_times must be in the range [0, total_time]")
    except (TypeError, ValueError) as e:
        raise TypeError("sample_times must be convertible to a 1D int32 array") from e

    return sample_times_arr
