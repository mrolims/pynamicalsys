# transport.py

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


import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

from pynamicalsys.common.types import int_t, map_t, numeric_t
from pynamicalsys.discrete_time.trajectory import iterate_mapping


@njit(parallel=True)
def diffusion_coefficient(
    u0: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: int_t,
    mapping: map_t,
    axis: int = 1,
) -> np.float64:
    """
    Compute the diffusion coefficient from an ensemble of trajectories.

    The diffusion coefficient is estimated through the Einstein relation

    .. math::
        D \\approx \\frac{\\langle (x(t)-x(0))^2 \\rangle}{2t},

    where the average is taken over the ensemble of initial conditions.

    Parameters
    ----------
    u0 : NDArray[np.float64]
        Ensemble of initial conditions with shape ``(num_ic, system_dimension)``.
    parameters : NDArray[np.float64]
        System parameters passed to ``mapping``.
    total_time : int_t
        Number of iterations used in the transport estimate.
    mapping : map_t
        System mapping function.
    axis : int, optional
        Coordinate index used in the displacement calculation.

    Returns
    -------
    np.float64
        Estimated diffusion coefficient.
    """
    num_ic = u0.shape[0]
    u_final = np.empty_like(u0)

    for i in prange(num_ic):
        u_final[i] = iterate_mapping(u0[i], parameters, total_time, mapping)

    msd = np.mean((u_final[:, axis] - u0[:, axis]) ** 2)

    return np.float64(msd / (2.0 * total_time))


@njit(parallel=True)
def average_vs_time(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: int_t,
    mapping: map_t,
    sample_times: NDArray[np.integer] | None = None,
    axis: int = 1,
) -> NDArray[np.float64]:
    """
    Compute the ensemble average of one coordinate as a function of time.

    Parameters
    ----------
    u : NDArray[np.float64]
        Ensemble of initial conditions with shape ``(num_ic, system_dimension)``.
    parameters : NDArray[np.float64]
        System parameters passed to ``mapping``.
    total_time : int_t
        Total number of iterations.
    mapping : map_t
        System mapping function.
    sample_times : Optional[NDArray[np.int32]], optional
        Array of sampling times. If ``None``, all times from ``1`` to
        ``total_time`` are used.
    axis : int, optional
        Coordinate index whose ensemble average is computed.

    Returns
    -------
    NDArray[np.float64]
        Ensemble-average time series evaluated at the requested sample times.
    """
    num_ic = u.shape[0]
    u_current = u.copy()

    if sample_times is None:
        sample_times_arr = np.arange(1, total_time + 1, dtype=np.int64)
        output = np.empty(total_time, dtype=np.float64)
    else:
        sample_times_arr = sample_times
        output = np.empty(sample_times_arr.shape[0], dtype=np.float64)

    prev_t = 0
    for sample_idx in range(sample_times_arr.shape[0]):
        st = sample_times_arr[sample_idx]
        steps = st - prev_t

        for _ in range(steps):
            for i in prange(num_ic):
                u_current[i] = mapping(u_current[i], parameters)

        output[sample_idx] = np.mean(u_current[:, axis])
        prev_t = st

    return output


@njit(parallel=True)
def cumulative_average_vs_time(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: int_t,
    mapping: map_t,
    sample_times: NDArray[np.integer] | None = None,
    axis: int = 1,
) -> NDArray[np.float64]:
    """
    Compute the cumulative ensemble average of one coordinate as a function of time.

    Parameters
    ----------
    u : NDArray[np.float64]
        Ensemble of initial conditions with shape ``(num_ic, system_dimension)``.
    parameters : NDArray[np.float64]
        System parameters passed to ``mapping``.
    total_time : int_t
        Total number of iterations.
    mapping : map_t
        System mapping function.
    sample_times : Optional[NDArray[np.int32]], optional
        Array of sampling times. If ``None``, all times from ``1`` to
        ``total_time`` are used.
    axis : int, optional
        Coordinate index whose cumulative ensemble average is computed.

    Returns
    -------
    NDArray[np.float64]
        Cumulative ensemble-average time series evaluated at the requested
        sample times.
    """
    num_ic = u.shape[0]
    u_current = u.copy()
    sum_values = np.zeros(num_ic, dtype=np.float64)

    if sample_times is None:
        sample_times_arr = np.arange(1, total_time + 1, dtype=np.int64)
        output = np.empty(total_time, dtype=np.float64)
    else:
        sample_times_arr = sample_times
        output = np.empty(sample_times_arr.shape[0], dtype=np.float64)

    prev_t = 0
    for sample_idx in range(sample_times_arr.shape[0]):
        st = sample_times_arr[sample_idx]
        steps = st - prev_t

        for _ in range(steps):
            for i in prange(num_ic):
                u_current[i] = mapping(u_current[i], parameters)
            sum_values += u_current[:, axis]

        output[sample_idx] = np.mean(sum_values / st)
        prev_t = st

    return output


@njit(parallel=True)
def root_mean_squared(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: int_t,
    mapping: map_t,
    sample_times: NDArray[np.integer] | None = None,
    axis: int = 1,
) -> NDArray[np.float64]:
    """
    Compute the root-mean-squared value of one coordinate as a function of time.

    Parameters
    ----------
    u : NDArray[np.float64]
        Ensemble of initial conditions with shape ``(num_ic, system_dimension)``.
    parameters : NDArray[np.float64]
        System parameters passed to ``mapping``.
    total_time : int_t
        Total number of iterations.
    mapping : map_t
        System mapping function.
    sample_times : Optional[NDArray[np.int32]], optional
        Array of sampling times. If ``None``, all times from ``1`` to
        ``total_time`` are used.
    axis : int, optional
        Coordinate index whose RMS value is computed.

    Returns
    -------
    NDArray[np.float64]
        RMS time series evaluated at the requested sample times.
    """
    num_ic = u.shape[0]
    u_current = u.copy()
    sum_squares = np.zeros(num_ic, dtype=np.float64)

    if sample_times is None:
        sample_times_arr = np.arange(1, total_time + 1, dtype=np.int64)
        output = np.empty(total_time, dtype=np.float64)
    else:
        sample_times_arr = sample_times
        output = np.empty(sample_times_arr.shape[0], dtype=np.float64)

    prev_t = 0
    for sample_idx in range(sample_times_arr.shape[0]):
        st = sample_times_arr[sample_idx]
        steps = st - prev_t

        for _ in range(steps):
            for i in prange(num_ic):
                u_current[i] = mapping(u_current[i], parameters)
            sum_squares += u_current[:, axis] ** 2

        output[sample_idx] = np.sqrt(np.mean(sum_squares / st))
        prev_t = st

    return output


@njit(parallel=True)
def mean_squared_displacement(
    u0: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: int_t,
    mapping: map_t,
    sample_times: NDArray[np.integer] | None = None,
    axis: int = 1,
) -> NDArray[np.float64]:
    """
    Compute the mean squared displacement of one coordinate as a function of time.

    Parameters
    ----------
    u0 : NDArray[np.float64]
        Ensemble of initial conditions with shape ``(num_ic, system_dimension)``.
    parameters : NDArray[np.float64]
        System parameters passed to ``mapping``.
    total_time : int_t
        Total number of iterations.
    mapping : map_t
        System mapping function.
    sample_times : Optional[NDArray[np.int32]], optional
        Array of sampling times. If ``None``, all times from ``1`` to
        ``total_time`` are used.
    axis : int, optional
        Coordinate index used in the displacement calculation.

    Returns
    -------
    NDArray[np.float64]
        Mean-squared-displacement time series evaluated at the requested sample
        times.
    """
    num_ic = u0.shape[0]
    u = u0.copy()
    initial_values = u0[:, axis].copy()

    if sample_times is None:
        sample_times_arr = np.arange(1, total_time + 1, dtype=np.int64)
        output = np.empty(total_time, dtype=np.float64)
    else:
        sample_times_arr = sample_times
        output = np.empty(sample_times_arr.shape[0], dtype=np.float64)

    prev_t = 0
    for sample_idx in range(sample_times_arr.shape[0]):
        st = sample_times_arr[sample_idx]
        steps = st - prev_t

        for _ in range(steps):
            for i in prange(num_ic):
                u[i] = mapping(u[i], parameters)

        displacements = u[:, axis] - initial_values
        output[sample_idx] = np.mean(displacements**2)
        prev_t = st

    return output


@njit
def recurrence_times(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: int_t,
    mapping: map_t,
    eps: numeric_t,
    transient_time: int_t | None = None,
) -> NDArray[np.float64]:
    """
    Compute recurrence times to an ``eps``-neighborhood of the reference point.

    Parameters
    ----------
    u : NDArray[np.float64]
        Initial condition of shape ``(system_dimension,)``.
    parameters : NDArray[np.float64]
        System parameters passed to ``mapping``.
    total_time : int_t
        Total number of iterations used to detect recurrences.
    mapping : map_t
        System mapping function.
    eps : numeric_t
        Side length of the recurrence neighborhood.
    transient_time : Optional[int_t], optional
        Number of initial iterations discarded before defining the recurrence box.

    Returns
    -------
    NDArray[np.float64]
        Array containing the recurrence times between successive returns to the
        ``eps``-neighborhood.
    """
    state = u.copy()

    if transient_time is not None:
        state = iterate_mapping(state, parameters, transient_time, mapping)

    lower_bound = state - eps / 2.0
    upper_bound = state + eps / 2.0

    rt = 0
    rts = []

    for _ in range(total_time):
        state = mapping(state, parameters)
        rt += 1

        if np.all(state >= lower_bound) and np.all(state <= upper_bound):
            rts.append(rt)
            rt = 0

    return np.asarray(rts, dtype=np.float64)
