# trajectory_analysis.py

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

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

from pynamicalsys.continuous_time.numerical_integrators import rk4_step_wrapped


@njit
def step(
    time: np.float64,
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    equations_of_motion: Callable[
        [NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]
    ],
    jacobian: Optional[
        Callable[
            [np.float64, NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]
        ]
    ] = None,
    time_step: float = 0.01,
    atol: float = 1e-6,
    rtol: float = 1e-3,
    integrator=rk4_step_wrapped,
    number_of_deviation_vectors: Optional[int] = None,
) -> NDArray[np.float64]:

    u = u.copy()
    accept = False

    while not accept:
        u_new, time_new, time_step_new, accept = integrator(
            time,
            u,
            parameters,
            equations_of_motion,
            jacobian=jacobian,
            time_step=time_step,
            atol=atol,
            rtol=rtol,
            number_of_deviation_vectors=number_of_deviation_vectors,
        )
        if accept:
            time = time_new
            u = u_new.copy()

        time_step = time_step_new

    return u_new, time_new, time_step_new


@njit
def evolve_system(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: float,
    equations_of_motion: Callable[
        [NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]
    ],
    time_step: float = 0.01,
    atol: float = 1e-6,
    rtol: float = 1e-3,
    integrator=rk4_step_wrapped,
) -> NDArray[np.float64]:

    u = u.copy()

    time = 0
    while time < total_time:
        u, time, time_step = step(
            time,
            u,
            parameters,
            equations_of_motion,
            time_step=time_step,
            atol=atol,
            rtol=rtol,
            integrator=integrator,
        )
        if time + time_step > total_time:
            time_step = total_time - time

    return u


@njit
def generate_trajectory(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: float,
    equations_of_motion: Callable[
        [np.float64, NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]
    ],
    transient_time: Optional[float] = None,
    time_step: float = 0.01,
    atol: float = 1e-6,
    rtol: float = 1e-3,
    integrator=rk4_step_wrapped,
) -> NDArray[np.float64]:

    u = u.copy()
    if transient_time is not None:
        u = evolve_system(
            u,
            parameters,
            transient_time,
            equations_of_motion,
            time_step=time_step,
            atol=atol,
            rtol=rtol,
            integrator=integrator,
        )
        time = transient_time
    else:
        time = 0

    neq = len(u)
    result = np.zeros(neq + 1)
    trajectory = []

    while time < total_time:
        if time + time_step > total_time:
            time_step = total_time - time

        u, time, time_step = step(
            time,
            u,
            parameters,
            equations_of_motion,
            time_step=time_step,
            atol=atol,
            rtol=rtol,
            integrator=integrator,
        )

        result = [time]
        for i in range(neq):
            result.append(u[i])
        trajectory.append(result)

    return trajectory


@njit(cache=True, parallel=True)
def ensemble_trajectories(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: float,
    equations_of_motion: Callable[
        [np.float64, NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]
    ],
    transient_time: Optional[float] = None,
    time_step: float = 0.01,
    atol: float = 1e-6,
    rtol: float = 1e-3,
    integrator=rk4_step_wrapped,
) -> NDArray[np.float64]:

    if u.ndim != 2:
        raise ValueError("Initial conditions must be 2D array (num_ic, neq)")

    num_ic, neq = u.shape

    trajectories = []

    for i in prange(num_ic):
        trajectory = generate_trajectory(
            u[i],
            parameters,
            total_time,
            equations_of_motion,
            transient_time=transient_time,
            time_step=time_step,
            atol=atol,
            rtol=rtol,
            integrator=integrator,
        )
        trajectories.append(np.array(trajectory))

    return trajectories
