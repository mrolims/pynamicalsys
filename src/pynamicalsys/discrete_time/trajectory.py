# trajectory.py

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

from typing import Optional

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

from pynamicalsys.common.types import int_t, map_t


@njit
def iterate_mapping(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: int_t,
    mapping: map_t,
) -> NDArray[np.float64]:
    """
    Iterate a discrete-time mapping for a fixed number of steps.

    Parameters
    ----------
    u : NDArray[np.float64]
        Initial condition of shape `(system_dimension,)`.
    parameters : NDArray[np.float64]
        System parameters passed to the mapping.
    total_time : int_t
        Number of iterations to apply.
    mapping : map_t
        Mapping function with signature
        `u_next = mapping(u, parameters)`.

    Returns
    -------
    NDArray[np.float64]
        Final state after `total_time` iterations.

    Notes
    -----
    This is a low-level routine and assumes inputs have already been validated
    by the wrapper.
    """
    state = u.copy()

    for _ in range(total_time):
        state = mapping(state, parameters)

    return state


@njit
def generate_trajectory(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: int_t,
    mapping: map_t,
    transient_time: Optional[int_t] = None,
) -> NDArray[np.float64]:
    """
    Generate a trajectory from a single initial condition.

    Parameters
    ----------
    u : NDArray[np.float64]
        Initial condition of shape `(system_dimension,)`.
    parameters : NDArray[np.float64]
        System parameters passed to the mapping.
    total_time : int_t
        Total number of iterations used in the computation.
    mapping : map_t
        Mapping function with signature
        `u_next = mapping(u, parameters)`.
    transient_time : Optional[int_t], optional
        Number of initial iterations discarded before storing the trajectory.
        If `None`, no transient is removed.

    Returns
    -------
    NDArray[np.float64]
        Trajectory array of shape `(sample_size, system_dimension)`, where
        `sample_size = total_time - transient_time` if a transient is used,
        and `sample_size = total_time` otherwise.

    Notes
    -----
    The returned trajectory does not include the initial condition. The first row
    corresponds to the first stored iterate after the transient.
    This is a low-level routine and assumes inputs have already been validated
    by the wrapper.
    """
    state = u.copy()

    if transient_time is None:
        sample_size = total_time
    else:
        state = iterate_mapping(state, parameters, transient_time, mapping)
        sample_size = total_time - transient_time

    system_dimension = state.shape[0]
    trajectory = np.empty((sample_size, system_dimension), dtype=np.float64)

    for i in range(sample_size):
        state = mapping(state, parameters)
        trajectory[i] = state

    return trajectory


@njit(parallel=True)
def ensemble_trajectories(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: int_t,
    mapping: map_t,
    transient_time: Optional[int_t] = None,
) -> NDArray[np.float64]:
    """
    Generate trajectories for an ensemble of initial conditions.

    Parameters
    ----------
    u : NDArray[np.float64]
        Initial conditions of shape `(num_initial_conditions, system_dimension)`.
    parameters : NDArray[np.float64]
        System parameters passed to the mapping.
    total_time : int_t
        Total number of iterations used for each trajectory.
    mapping : map_t
        Mapping function with signature
        `u_next = mapping(u, parameters)`.
    transient_time : Optional[int_t], optional
        Number of initial iterations discarded before storing each trajectory.
        If `None`, no transient is removed.

    Returns
    -------
    NDArray[np.float64]
        Concatenated trajectory array of shape
        `(num_initial_conditions * sample_size, system_dimension)`, where
        `sample_size = total_time - transient_time` if a transient is used,
        and `sample_size = total_time` otherwise.

    Notes
    -----
    Trajectories are stacked in the same order as the input initial conditions.
    This is a low-level routine and assumes inputs have already been validated
    by the wrapper.
    """
    num_initial_conditions, system_dimension = u.shape

    if transient_time is None:
        sample_size = total_time
    else:
        sample_size = total_time - transient_time

    trajectories = np.empty(
        (num_initial_conditions * sample_size, system_dimension),
        dtype=np.float64,
    )

    for i in prange(num_initial_conditions):
        trajectory = generate_trajectory(
            u=u[i],
            parameters=parameters,
            total_time=total_time,
            mapping=mapping,
            transient_time=transient_time,
        )
        trajectories[i * sample_size : (i + 1) * sample_size] = trajectory

    return trajectories
