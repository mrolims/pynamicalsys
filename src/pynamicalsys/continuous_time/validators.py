# validators.py

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

from numbers import Real
import numpy as np
from pynamicalsys.common.types import numeric_t


def validate_times(
    transient_time: numeric_t | None,
    total_time: numeric_t,
    type_: type = Real,
) -> tuple[np.float64 | None, np.float64]:
    """
    Validate continuous-time simulation times.

    Parameters
    ----------
    transient_time : numeric_t | None
        Initial time to discard before measurements. If not None, it must be a
        non-negative real number strictly smaller than `total_time`.
    total_time : numeric_t
        Total integration time. Must be a non-negative real number.
    type_ : type, optional
        Expected numeric type category, default is `numbers.Real`.

    Returns
    -------
    tuple[float | None, float]
        Validated `(transient_time, total_time)` converted to floats.

    Raises
    ------
    TypeError
        If `total_time` or `transient_time` is not of the expected numeric type.
    ValueError
        If `total_time` is negative.
        If `transient_time` is negative.
        If `transient_time >= total_time`.
    """
    type_name = getattr(type_, "__name__", str(type_))

    if isinstance(total_time, bool) or not isinstance(total_time, type_):
        raise TypeError(f"total_time must be of type {type_name}")

    total_time = np.float64(total_time)

    if total_time < 0:
        raise ValueError("total_time must be non-negative")

    if transient_time is not None:
        if isinstance(transient_time, bool) or not isinstance(transient_time, type_):
            raise TypeError(f"transient_time must be of type {type_name}")

        transient_time = np.float64(transient_time)

        if transient_time < 0:
            raise ValueError("transient_time must be non-negative")

        if transient_time >= total_time:
            raise ValueError("transient_time must be less than total_time")
    else:
        transient_time = np.float64(0.0)

    return transient_time, total_time


def validate_sample_times(
    sample_times,
    transient_time: np.float64,
    total_time: np.float64,
) -> np.ndarray:
    """
    Validate continuous-time sample times.

    Parameters
    ----------
    sample_times : array-like
        Times at which the history is sampled. They must be finite and lie in
        the interval `(transient_time, total_time]`.
    transient_time : np.float64
        Time at which the accumulation starts.
    total_time : np.float64
        Final integration time.

    Returns
    -------
    np.ndarray
        Sorted 1D array of unique sample times, of dtype `float64`.

    Raises
    ------
    TypeError
        If `sample_times` cannot be converted to a 1D array of real numbers.
    ValueError
        If `sample_times` is not 1D, is empty, contains non-finite values, or
        contains values outside `(transient_time, total_time]`.
    """
    try:
        sample_times_arr = np.asarray(sample_times, dtype=np.float64)
    except (TypeError, ValueError) as e:
        raise TypeError(
            "sample_times must be convertible to a 1D array of real numbers"
        ) from e

    if sample_times_arr.ndim != 1:
        raise ValueError("sample_times must be a 1D array")

    if sample_times_arr.size == 0:
        raise ValueError("sample_times must not be empty")

    if not np.all(np.isfinite(sample_times_arr)):
        raise ValueError("sample_times must contain only finite values")

    if np.any(sample_times_arr <= transient_time) or np.any(
        sample_times_arr > total_time
    ):
        raise ValueError(
            "sample_times must be in the interval (transient_time, total_time]"
        )

    return np.unique(sample_times_arr)
