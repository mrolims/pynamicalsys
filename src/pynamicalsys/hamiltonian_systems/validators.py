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

from numbers import Integral, Real
from typing import Type, Union

import numpy as np
from numpy.typing import NDArray


def validate_non_negative(
    value: Union[Integral, int, Real, float],
    name: str,
    type_: Type[Union[Integral, int, Real, float]] = Integral,
) -> None:
    if not isinstance(value, type_):
        raise TypeError(f"{name} must be of type {type_.__name__}")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def validate_times(transient_time, total_time) -> tuple[float, float]:

    if isinstance(total_time, (Integral, Real)):
        total_time = float(total_time)
    else:
        raise ValueError("total_time must be a valid number")
    if total_time < 0:
        raise ValueError("total_time must be non-negative")

    if transient_time is not None:

        if isinstance(transient_time, (Integral, Real)):
            transient_time = float(transient_time)
        else:
            raise ValueError("transient_time must be a valid number")
        if transient_time < 0:
            raise ValueError("transient_time must be non-negative")

        if transient_time >= total_time:
            raise ValueError("transient_time must be less than total_time")

    return transient_time, total_time


def validate_initial_conditions(
    u, degrees_of_freedom, allow_ensemble=True
) -> NDArray[np.float64]:
    if np.isscalar(u):
        u = np.array([u], dtype=np.float64)
    else:
        u = np.asarray(u, dtype=np.float64)
        if u.ndim not in (1, 2):
            raise ValueError("Initial condition must be 1D or 2D array")

    u = np.ascontiguousarray(u).copy()

    if u.ndim == 1:
        if len(u) != degrees_of_freedom:
            raise ValueError(
                f"1D initial condition must have length {degrees_of_freedom}"
            )
    elif u.ndim == 2:
        if not allow_ensemble:
            raise ValueError(
                "Ensemble of initial conditions not allowed in this context"
            )
        if u.shape[1] != degrees_of_freedom:
            raise ValueError(
                f"Each initial condition must have length {degrees_of_freedom}"
            )
    return u


def validate_parameters(parameters, number_of_parameters) -> NDArray[np.float64]:
    if number_of_parameters == 0:
        if parameters is not None:
            raise ValueError("This system does not expect any parameters.")
        return np.array([0], dtype=np.float64)

    if parameters is None:
        raise ValueError(
            f"This system expects {number_of_parameters} parameter(s), but got None."
        )

    if np.isscalar(parameters):
        parameters = np.array([parameters], dtype=np.float64)
    else:
        parameters = np.asarray(parameters, dtype=np.float64)
        if parameters.ndim != 1:
            raise ValueError(
                f"`parameters` must be a 1D array or scalar. Got shape {parameters.shape}."
            )

    if parameters.size != number_of_parameters:
        raise ValueError(
            f"Expected {number_of_parameters} parameter(s), but got {parameters.size}."
        )

    return parameters
