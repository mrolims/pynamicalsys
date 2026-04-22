import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

from pynamicalsys.common.types import int_t, map_t


@njit(parallel=True)
def ensemble_time_average(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    mapping: map_t,
    total_time: int_t,
    axis: int = 1,
) -> NDArray[np.float64]:
    """
    Compute the centered time average of a coordinate for an ensemble of initial conditions.

    For each initial condition, this function computes the time average of the
    selected coordinate over `total_time` iterations. It then subtracts the
    ensemble mean of these time averages, returning the fluctuation of each
    trajectory around the ensemble-averaged time mean.

    Parameters
    ----------
    u : NDArray[np.float64]
        Array of initial conditions of shape `(num_ic, system_dimension)`.
    parameters : NDArray[np.float64]
        System parameters passed to `mapping`.
    mapping : map_t
        Mapping function of the dynamical system.
    total_time : int_t
        Number of iterations used to compute the time averages.
    axis : int, optional
        Coordinate index used in the averaging. Default is `1`.

    Returns
    -------
    NDArray[np.float64]
        Array of shape `(num_ic,)` containing the centered time average for each
        initial condition.

    Notes
    -----
    The returned quantity is

    `A_i - <A>`

    where `A_i` is the time average of trajectory `i` and `<A>` is the ensemble
    average of all trajectory time averages.
    """
    u_current = u.copy()
    num_ic = u_current.shape[0]
    sum_values = np.zeros(num_ic, dtype=np.float64)

    for i in prange(num_ic):
        for _ in range(total_time):
            u_current[i] = mapping(u_current[i], parameters)
            sum_values[i] += u_current[i, axis]

    time_averages = sum_values / total_time
    ensemble_average = np.sum(time_averages) / num_ic

    return time_averages - ensemble_average
