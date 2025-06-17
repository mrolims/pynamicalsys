# numerical_integrators.py

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

from typing import Optional, Callable  # , Union, Tuple, Dict, List, Any, Sequence
from numpy.typing import NDArray
import numpy as np
from numba import njit, prange

from pynamicalsys.continuous_time.models import variational_equations


@njit
def rk4_step(
    t: float,
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    equations_of_motion: Callable[
        [float, NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]
    ],
    time_step: float = 0.01,
) -> NDArray[np.float64]:

    k1 = equations_of_motion(t, u, parameters)
    k2 = equations_of_motion(t + 0.5 * time_step, u + 0.5 * time_step * k1, parameters)
    k3 = equations_of_motion(t + 0.5 * time_step, u + 0.5 * time_step * k2, parameters)
    k4 = equations_of_motion(t + time_step, u + time_step * k3, parameters)

    u_next = u + (time_step / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return u_next


@njit
def variational_rk4_step(
    t: float,
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    equations_of_motion: Callable[
        [float, NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]
    ],
    jacobian: Callable[
        [float, NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]
    ],
    time_step: float = 0.01,
    number_of_deviation_vectors: Optional[int] = None,
) -> NDArray[np.float64]:

    k1 = variational_equations(
        t,
        u,
        parameters,
        equations_of_motion,
        jacobian,
        number_of_deviation_vectors=number_of_deviation_vectors,
    )

    k2 = variational_equations(
        t + 0.5 * time_step,
        u + 0.5 * time_step * k1,
        parameters,
        equations_of_motion,
        jacobian,
        number_of_deviation_vectors=number_of_deviation_vectors,
    )
    k3 = variational_equations(
        t + 0.5 * time_step,
        u + 0.5 * time_step * k2,
        parameters,
        equations_of_motion,
        jacobian,
        number_of_deviation_vectors=number_of_deviation_vectors,
    )
    k4 = variational_equations(
        t + time_step,
        u + time_step * k3,
        parameters,
        equations_of_motion,
        jacobian,
        number_of_deviation_vectors=number_of_deviation_vectors,
    )

    u_next = u + (time_step / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return u_next
