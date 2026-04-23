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

from numbers import Real

import numpy as np
from numpy.typing import NDArray

from pynamicalsys.common.types import int_t, numeric_like_t, numeric_t


def validate_times(
    transient_time: numeric_t | None,
    total_time: numeric_t,
) -> tuple[np.float64 | None, np.float64]:
    """
    Validate Hamiltonian-system time parameters.

    Parameters
    ----------
    transient_time : numeric_t | None
        Initial integration time to discard. If not `None`, it must be a
        non-negative real number strictly smaller than `total_time`.
    total_time : numeric_t
        Total integration time. It must be a non-negative real number.

    Returns
    -------
    tuple[np.float64 | None, np.float64]
        Validated `(transient_time, total_time)`.

    Raises
    ------
    TypeError
        If `total_time` or `transient_time` is not a valid real number.
    ValueError
        If `total_time` is negative.
        If `transient_time` is negative.
        If `transient_time >= total_time`.
    """
    if isinstance(total_time, bool) or not isinstance(total_time, Real):
        raise TypeError("total_time must be a valid real number")

    total_time = np.float64(total_time)

    if total_time < np.float64(0.0):
        raise ValueError("total_time must be non-negative")

    if transient_time is not None:
        if isinstance(transient_time, bool) or not isinstance(transient_time, Real):
            raise TypeError("transient_time must be a valid real number")

        transient_time = np.float64(transient_time)

        if transient_time < np.float64(0.0):
            raise ValueError("transient_time must be non-negative")

        if transient_time >= total_time:
            raise ValueError("transient_time must be less than total_time")

    return transient_time, total_time


def validate_initial_conditions(
    u: numeric_like_t,
    degrees_of_freedom: int_t,
    allow_ensemble: bool = True,
) -> NDArray[np.float64]:
    """
    Validate and standardize Hamiltonian initial conditions.

    Parameters
    ----------
    u : numeric_like_t
        Initial condition(s). It may define either one initial condition of
        shape `(dof,)` or an ensemble of shape `(num_ic, dof)`.
    degrees_of_freedom : int_t
        Expected number of degrees of freedom.
    allow_ensemble : bool, optional
        Whether a 2D ensemble of initial conditions is allowed.

    Returns
    -------
    NDArray[np.float64]
        Validated contiguous array of initial conditions.

    Raises
    ------
    TypeError
        If `degrees_of_freedom` is not an integer.
    ValueError
        If the shape of `u` is invalid.
        If `u` does not match the expected number of degrees of freedom.
        If an ensemble is provided when `allow_ensemble=False`.
    """
    if isinstance(degrees_of_freedom, bool) or not isinstance(
        degrees_of_freedom, (int, np.integer)
    ):
        raise TypeError("degrees_of_freedom must be an integer")

    if np.isscalar(u):
        u = np.array([u], dtype=np.float64)
    else:
        u = np.asarray(u, dtype=np.float64)

        if u.ndim not in (1, 2):
            raise ValueError("Initial condition must be a 1D or 2D array")

    u = np.ascontiguousarray(u).copy()

    if u.ndim == 1:
        if u.shape[0] != degrees_of_freedom:
            raise ValueError(
                f"1D initial condition must have length {degrees_of_freedom}"
            )

    else:
        if not allow_ensemble:
            raise ValueError(
                "Ensemble of initial conditions not allowed in this context"
            )

        if u.shape[1] != degrees_of_freedom:
            raise ValueError(
                f"Each initial condition must have length {degrees_of_freedom}"
            )

    return u
