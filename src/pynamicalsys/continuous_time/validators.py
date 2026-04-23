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
