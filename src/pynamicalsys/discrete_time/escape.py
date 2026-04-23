# escape.py

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

from typing import Tuple

import numpy as np
from numba import njit
from numpy.typing import NDArray

from pynamicalsys.common.types import int_t, map_t


@njit
def escape_basin_and_time_entering(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    mapping: map_t,
    max_time: int_t,
    exits: NDArray[np.float64],
) -> Tuple[int, int_t]:
    """
    Track the trajectory until it enters one of a set of exit regions.

    Parameters
    ----------
    u : NDArray[np.float64]
        Initial condition of shape `(system_dimension,)`.
    parameters : NDArray[np.float64]
        System parameters passed to the mapping function.
    mapping : map_t
        System mapping function.
    max_time : int_t
        Maximum number of iterations.
    exits : NDArray[np.float64]
        Exit regions of shape `(n_exits, system_dimension, 2)`, where
        `exits[k, i, 0]` and `exits[k, i, 1]` are the lower and upper bounds
        of the `i`-th coordinate for the `k`-th exit region.

    Returns
    -------
    Tuple[int, int]
        Tuple `(exit_index, escape_time)`, where:

        - `exit_index` is the index of the first exit region entered,
        - `escape_time` is the iteration at which the escape occurred.

        If no escape occurs within `max_time`, returns `(-1, max_time)`.

    Raises
    ------
    ValueError
        - If `max_time <= 0`.
        - If `exits` does not have shape `(n_exits, system_dimension, 2)`.

    Notes
    -----
    Each exit region is interpreted as a hyperrectangle in phase space.
    """
    if max_time <= 0:
        raise ValueError("max_time must be positive")

    if exits.ndim != 3 or exits.shape[2] != 2:
        raise ValueError("exits must have shape (n_exits, system_dimension, 2)")

    n_exits = exits.shape[0]
    n_dim = exits.shape[1]
    state = u.copy()

    for time in range(1, max_time + 1):
        state = mapping(state, parameters)

        for exit_idx in range(n_exits):
            in_exit = True
            for dim in range(n_dim):
                lower = exits[exit_idx, dim, 0]
                upper = exits[exit_idx, dim, 1]
                if state[dim] < lower or state[dim] > upper:
                    in_exit = False
                    break

            if in_exit:
                return exit_idx, time

    return -1, max_time


@njit
def escape_time_exiting(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    mapping: map_t,
    max_time: int_t,
    region_limits: NDArray[np.float64],
) -> Tuple[int, int_t]:
    """
    Track the trajectory until it exits a bounded region.

    Parameters
    ----------
    u : NDArray[np.float64]
        Initial condition of shape `(system_dimension,)`.
    parameters : NDArray[np.float64]
        System parameters passed to the mapping function.
    mapping : map_t
        System mapping function.
    max_time : int_t
        Maximum number of iterations.
    region_limits : NDArray[np.float64]
        Region boundaries of shape `(system_dimension, 2)`, where
        `region_limits[i, 0]` is the lower bound and
        `region_limits[i, 1]` is the upper bound of coordinate `i`.

    Returns
    -------
    Tuple[int, int]
        Tuple `(face_index, escape_time)`, where:

        - `face_index` identifies the boundary through which the trajectory escaped,
        - `escape_time` is the iteration at which the escape occurred.

        Face indexing follows:
        - `2*i`   -> lower boundary of coordinate `i`
        - `2*i+1` -> upper boundary of coordinate `i`

        If no escape occurs within `max_time`, returns `(-1, max_time)`.

    Raises
    ------
    ValueError
        - If `max_time <= 0`.
        - If `region_limits` does not have shape `(system_dimension, 2)`.
    """
    if max_time <= 0:
        raise ValueError("max_time must be positive")

    if region_limits.ndim != 2 or region_limits.shape[1] != 2:
        raise ValueError("region_limits must have shape (system_dimension, 2)")

    n_dim = region_limits.shape[0]
    state = u.copy()

    for time in range(1, max_time + 1):
        state = mapping(state, parameters)

        for dim in range(n_dim):
            if state[dim] < region_limits[dim, 0]:
                return 2 * dim, time

            if state[dim] > region_limits[dim, 1]:
                return 2 * dim + 1, time

    return -1, max_time


@njit
def survival_probability_core(
    escape_times: NDArray[np.int64],
    max_time: int_t,
    min_time: int_t = 1,
    time_step: int_t = 1,
) -> Tuple[NDArray[np.int64], NDArray[np.float64]]:
    """
    Compute the survival probability from a set of escape times.

    The survival probability `S(t)` is the fraction of trajectories whose
    escape time is strictly greater than `t`.

    Parameters
    ----------
    escape_times : NDArray[np.int64]
        Escape times for an ensemble of trajectories.
    max_time : int_t
        Maximum time at which the survival probability is evaluated.
    min_time : int_t, optional
        Minimum evaluation time. Default is `1`.
    time_step : int_t, optional
        Step between consecutive evaluation times. Default is `1`.

    Returns
    -------
    Tuple[NDArray[np.int64], NDArray[np.float64]]
        Tuple `(t_values, survival_probs)`, where:

        - `t_values` contains the evaluation times,
        - `survival_probs` contains the corresponding survival probabilities.

    Raises
    ------
    ValueError
        - If `max_time <= min_time`.
        - If `time_step <= 0`.
        - If any value in `escape_times` is smaller than `1`.

    Notes
    -----
    This implementation evaluates

    `S(t) = P(T_escape > t)`

    using sorted escape times and `np.searchsorted`.
    """
    if max_time <= min_time:
        raise ValueError("max_time must be greater than min_time")

    if time_step <= 0:
        raise ValueError("time_step must be positive")

    if np.any(escape_times < 1):
        raise ValueError("all escape_times must be >= 1")

    t_values = np.arange(min_time, max_time + 1, time_step, dtype=np.int64)
    sorted_times = np.sort(escape_times)
    n_samples = sorted_times.size

    if n_samples == 0:
        return t_values, np.empty(0, dtype=np.float64)

    escaped_by_t = np.searchsorted(sorted_times, t_values, side="right")
    survival_probs = 1.0 - escaped_by_t / n_samples

    return t_values, survival_probs
