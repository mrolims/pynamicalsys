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


# Quantities recorded per initial condition by `_ensemble_series`
_AVERAGE = 0
_CUMULATIVE_AVERAGE = 1
_ROOT_MEAN_SQUARED = 2
_MEAN_SQUARED_DISPLACEMENT = 3

# Number of values buffered per chunk of sample times, which bounds the memory
# used by `_ensemble_series`
_CHUNK_VALUES = 2**20


@njit(parallel=True)
def _advance_ensemble(
    states: NDArray[np.float64],
    sums: NDArray[np.float64],
    parameters: NDArray[np.float64],
    mapping: map_t,
    sample_times: NDArray[np.integer],
    start_time: int_t,
    axis: int,
    kind: int,
) -> NDArray[np.float64]:
    """
    Advance every initial condition through a chunk of sample times.

    Each initial condition is iterated from `start_time` to the last sample
    time in its own thread, so the parallel region is entered once per chunk
    instead of once per iteration. `states` and `sums` are updated in place.

    Returns
    -------
    NDArray[np.float64]
        Values of shape `(len(sample_times), num_ic)`: the coordinate `axis`
        at each sample time, or its running sum (`_CUMULATIVE_AVERAGE`) or
        running sum of squares (`_ROOT_MEAN_SQUARED`) up to it.
    """
    num_ic = states.shape[0]
    num_samples = sample_times.shape[0]
    values = np.empty((num_samples, num_ic), dtype=np.float64)

    for i in prange(num_ic):
        state = states[i].copy()
        total = sums[i]
        time = start_time
        for k in range(num_samples):
            while time < sample_times[k]:
                state = mapping(state, parameters)
                time += 1
                if kind == _CUMULATIVE_AVERAGE:
                    total += state[axis]
                elif kind == _ROOT_MEAN_SQUARED:
                    total += state[axis] ** 2
            if kind == _CUMULATIVE_AVERAGE or kind == _ROOT_MEAN_SQUARED:
                values[k, i] = total
            else:
                values[k, i] = state[axis]
        states[i] = state
        sums[i] = total

    return values


@njit
def _ensemble_series(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: int_t,
    mapping: map_t,
    sample_times: NDArray[np.integer] | None,
    axis: int,
    kind: int,
) -> NDArray[np.float64]:
    """
    Compute an ensemble statistic of one coordinate at each sample time.

    The sample times are processed in chunks of at most `_CHUNK_VALUES`
    values. For each sample, the statistic is reduced over the initial
    conditions with the same expression as a direct computation, so the
    result does not depend on the chunking or on the number of threads.
    """
    num_ic = u.shape[0]

    if sample_times is None:
        sample_times_arr = np.arange(1, total_time + 1, dtype=np.int64)
    else:
        sample_times_arr = sample_times

    num_samples = sample_times_arr.shape[0]
    output = np.empty(num_samples, dtype=np.float64)
    states = u.copy()
    sums = np.zeros(num_ic, dtype=np.float64)
    initial_values = u[:, axis].copy()
    chunk_size = max(1, _CHUNK_VALUES // max(1, num_ic))

    start_time = 0
    for chunk_start in range(0, num_samples, chunk_size):
        chunk_times = sample_times_arr[chunk_start : chunk_start + chunk_size]
        values = _advance_ensemble(
            states, sums, parameters, mapping, chunk_times, start_time, axis, kind
        )
        for k in range(chunk_times.shape[0]):
            st = chunk_times[k]
            row = values[k]
            if kind == _AVERAGE:
                output[chunk_start + k] = np.mean(row)
            elif kind == _CUMULATIVE_AVERAGE:
                output[chunk_start + k] = np.mean(row / st)
            elif kind == _ROOT_MEAN_SQUARED:
                output[chunk_start + k] = np.sqrt(np.mean(row / st))
            else:
                output[chunk_start + k] = np.mean((row - initial_values) ** 2)
        start_time = chunk_times[-1]

    return output


@njit
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
    return _ensemble_series(
        u, parameters, total_time, mapping, sample_times, axis, _AVERAGE
    )


@njit
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
    return _ensemble_series(
        u, parameters, total_time, mapping, sample_times, axis, _CUMULATIVE_AVERAGE
    )


@njit
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
    return _ensemble_series(
        u, parameters, total_time, mapping, sample_times, axis, _ROOT_MEAN_SQUARED
    )


@njit
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
    return _ensemble_series(
        u0, parameters, total_time, mapping, sample_times, axis, _MEAN_SQUARED_DISPLACEMENT
    )


@njit
def recurrence_times(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: int_t,
    mapping: map_t,
    eps: numeric_t,
    periods: NDArray[np.float64],
    transient_time: int_t | None = None,
) -> NDArray[np.int64]:
    """
    Compute recurrence times to an ``eps``-neighborhood of the reference point.

    Parameters
    ----------
    u : NDArray[np.float64]
        Initial condition of shape ``(system_dimension,)``.
    parameters : NDArray[np.float64]
        System parameters passed to ``mapping``.
    total_time : int_t
        Total number of iterations, counted from the initial condition, so it
        includes ``transient_time``. Recurrences are detected during the last
        ``total_time - transient_time`` iterations.
    mapping : map_t
        System mapping function.
    eps : numeric_t
        Side length of the recurrence neighborhood.
    periods : NDArray[np.float64]
        Wrapping period of each coordinate, of shape ``(system_dimension,)``,
        with ``np.inf`` for coordinates that do not wrap. The distance to the
        reference point is measured modulo the period, so the box wraps around
        the boundaries of periodic coordinates.
    transient_time : Optional[int_t], optional
        Number of initial iterations discarded before defining the recurrence box.

    Returns
    -------
    NDArray[np.int64]
        Array containing the recurrence times between successive returns to the
        ``eps``-neighborhood. The iterations after the last return are not
        included, since they do not form a complete recurrence time.
    """
    state = u.copy()

    sample_size = total_time
    if transient_time is not None:
        state = iterate_mapping(state, parameters, transient_time, mapping)
        sample_size = total_time - transient_time

    reference = state.copy()
    half_eps = eps / 2.0
    neq = len(state)

    rt = 0
    rts = []

    for _ in range(sample_size):
        state = mapping(state, parameters)
        rt += 1

        # Distance to the reference point, measured modulo the period of each
        # periodic coordinate so that the box wraps around its boundaries.
        inside = True
        for j in range(neq):
            difference = state[j] - reference[j]
            if np.isfinite(periods[j]):
                half_period = 0.5 * periods[j]
                difference = (difference + half_period) % periods[j] - half_period
            if abs(difference) > half_eps:
                inside = False
                break

        if inside:
            rts.append(rt)
            rt = 0

    return np.asarray(rts, dtype=np.int64)
