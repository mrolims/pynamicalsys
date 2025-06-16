# transport.py

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

from typing import Optional, Callable
from numpy.typing import NDArray
import numpy as np
from numba import njit, prange
from .trajectory_analysis import iterate_mapping


@njit(cache=True, parallel=True)
def diffusion_coefficient(
    u0: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: int,
    mapping: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    axis: int = 1,
) -> np.float64:
    """
    Compute the diffusion coefficient for an ensemble of trajectories.

    The diffusion coefficient D is estimated using the Einstein relation:
    D = lim_{t→∞} ⟨(x(t) - x(0))²⟩ / (2t)
    where ⟨·⟩ denotes ensemble averaging.

    Parameters
    ----------
    u0 : NDArray[np.float64]
        Array of initial conditions (shape: (num_ic, neq))
    parameters : NDArray[np.float64]
        System parameters passed to mapping function
    total_time : int
        Total evolution time (must be > transient_time)
    mapping : Callable[[NDArray, NDArray], NDArray]
        System evolution function: u_next = mapping(u, parameters)
    axis : int, optional
        axis index to analyze (default: 1)

    Returns
    -------
    float
        Estimated diffusion coefficient D

    Raises
    ------
    ValueError
        If total_time ≤ transient_time
        If axis index is invalid

    Notes
    -----
    - Assumes normal diffusion (linear mean squared displacement growth)
    - For anisotropic systems, analyze each axis separately
    - Parallelized over initial conditions for large ensembles
    """
    # Input validation
    if axis < 0 or axis >= u0.shape[1]:
        raise ValueError(f"axis must be in [0, {u0.shape[1]-1}]")

    num_ic = u0.shape[0]
    u_final = np.empty_like(u0)

    # Parallel evolution of trajectories
    for i in prange(num_ic):
        # Evolve each initial condition
        u_final[i] = iterate_mapping(u0[i], parameters, total_time, mapping)

    # Compute mean squared displacement
    msd = np.mean((u_final[:, axis] - u0[:, axis]) ** 2)

    return msd / (2 * (total_time))


@njit(cache=True, parallel=True)
def average_vs_time(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: int,
    mapping: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    sample_times: Optional[NDArray[np.int32]] = None,
    axis: int = 1,
    transient_time: int = 0,
) -> NDArray[np.float64]:
    """
    Compute the time evolution of ensemble averages for a dynamical system.

    Tracks the average value of a specified coordinate across multiple trajectories,
    with options for downsampling and transient removal. Useful for studying
    convergence to equilibrium or statistical properties.

    Parameters
    ----------
    u : NDArray[np.float64]
        Array of initial conditions (shape: (num_ic, num_dim))
    parameters : NDArray[np.float64]
        System parameters passed to mapping function
    total_time : int
        Total number of iterations (must be > transient_time)
    mapping : Callable[[NDArray, NDArray], NDArray]
        System evolution function: u_next = mapping(u, parameters)
    sample_times : Optional[NDArray[np.int64]], optional
        Specific time steps to record (default: record all steps)
    axis : int, optional
        Coordinate index to analyze (default: 1)
    transient_time : int, optional
        Initial iterations to discard (default: 0)

    Returns
    -------
    NDArray[np.float64]
        Array of average values at requested times

    Raises
    ------
    ValueError
        If total_time ≤ transient_time
        If sample_times contains values > total_time
        If axis is invalid

    Notes
    -----
    - Uses parallel processing over initial conditions
    - For large ensembles, consider using sample_times to reduce memory
    - The output length matches len(sample_times) if provided, else (total_time - transient_time)
    """
    # Input validation
    if total_time <= transient_time:
        raise ValueError("total_time must be > transient_time")
    if axis < 0 or axis >= u.shape[1]:
        raise ValueError(f"axis must be in [0, {u.shape[1]-1}]")
    if sample_times is not None:
        if np.any(sample_times >= total_time):
            raise ValueError("All sample_times must be < total_time")

    # Initialize tracking
    num_ic = u.shape[0]
    effective_time = total_time - transient_time
    u_current = u.copy()

    # Handle output array
    if sample_times is not None:
        output = np.empty(len(sample_times))
    else:
        output = np.empty(effective_time)

    output_idx = 0

    # Main evolution loop
    for t in range(total_time):
        # Parallel evolution
        for i in prange(num_ic):
            u_current[i] = mapping(u_current[i], parameters)

        # Record if past transient and matches sampling
        if t >= transient_time:
            output[output_idx] = np.mean(u_current[:, axis])
            if sample_times is None:
                output_idx += 1
            elif t in sample_times:
                output_idx += 1

    return output


@njit(cache=True, parallel=True)
def cumulative_average_vs_time(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: int,
    mapping: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    sample_times: Optional[NDArray[np.int32]] = None,
    axis: int = 1,
    transient_time: int = 0,
) -> NDArray[np.float64]:
    """
    Compute the time evolution of the cumulative average of a coordinate across trajectories.

    Parameters
    ----------
    u : NDArray[np.float64]
        Array of initial conditions (shape: (num_ic, num_dim))
    parameters : NDArray[np.float64]
        System parameters passed to mapping function
    total_time : int
        Total number of iterations (must be > transient_time)
    mapping : Callable[[NDArray, NDArray], NDArray]
        System evolution function: u_next = mapping(u, parameters)
    sample_times : Optional[NDArray[np.int64]], optional
        Specific time steps to record (default: record all steps)
    axis : int, optional
        Coordinate index to analyze (default: 1)
    transient_time : int, optional
        Initial iterations to discard (default: 0)

    Returns
    -------
    NDArray[np.float64]
        Array of cumulative average values at requested times

    Raises
    ------
    ValueError
        If total_time ≤ transient_time
        If sample_times contains invalid values
        If axis is invalid

    Notes
    -----
    - Uses parallel processing over initial conditions
    - For large total_time, use sample_times to reduce memory usage
    - The cumulative average is computed over the ensemble after removing transient
    """

    num_ic = u.shape[0]
    u_current = u.copy()
    sum_values = np.zeros(num_ic)

    # Initialize output array
    if sample_times is not None:
        output_size = len(sample_times)
    else:
        output_size = total_time - transient_time

    cumul_average = np.zeros(output_size)
    output_idx = 0

    # Main evolution loop
    for t in range(1, total_time + 1):
        # Parallel evolution
        for i in prange(num_ic):
            u_current[i] = mapping(u_current[i], parameters)

        # Record if past transient and matches sampling
        if t > transient_time:
            # Update running sum of squares
            sum_values += u_current[:, axis]
            cumul_average[output_idx] = np.mean(sum_values / t)
            if sample_times is None:
                output_idx += 1
            elif t in sample_times:
                output_idx += 1

    return cumul_average


@njit(cache=True, parallel=True)
def root_mean_squared(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: int,
    mapping: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    sample_times: Optional[NDArray[np.int32]] = None,
    axis: int = 1,
    transient_time: int = 0,
) -> NDArray[np.float64]:
    """
    Compute the time evolution of the root mean square (RMS) of a coordinate across trajectories.

    The RMS is calculated as:
    RMS(t) = sqrt(∑(x_i(t)²)/N)
    where N is the number of trajectories and x_i are the coordinate values.

    Parameters
    ----------
    u : NDArray[np.float64]
        Array of initial conditions (shape: (num_ic, num_dim))
    parameters : NDArray[np.float64]
        System parameters passed to mapping function
    total_time : int
        Total number of iterations (must be > transient_time)
    mapping : Callable[[NDArray, NDArray], NDArray]
        System evolution function: u_next = mapping(u, parameters)
    sample_times : Optional[NDArray[np.int64]], optional
        Specific time steps to record (default: record all steps)
    axis : int, optional
        Coordinate index to analyze (default: 1)
    transient_time : int, optional
        Initial iterations to discard (default: 0)

    Returns
    -------
    NDArray[np.float64]
        Array of RMS values at requested times

    Raises
    ------
    ValueError
        If total_time ≤ transient_time
        If sample_times contains invalid values
        If axis is invalid

    Notes
    -----
    - Uses parallel processing over initial conditions
    - For large total_time, use sample_times to reduce memory usage
    - The RMS is computed over the ensemble after removing transient
    """

    num_ic = u.shape[0]
    u_current = u.copy()
    sum_squares = np.zeros(num_ic)

    # Initialize output array
    if sample_times is not None:
        output_size = len(sample_times)
    else:
        output_size = total_time - transient_time

    rms = np.zeros(output_size)
    output_idx = 0

    # Main evolution loop
    for t in range(1, total_time + 1):
        # Parallel evolution
        for i in prange(num_ic):
            u_current[i] = mapping(u_current[i], parameters)

        # Record if past transient and matches sampling
        if t > transient_time:
            # Update running sum of squares
            sum_squares += u_current[:, axis] ** 2
            rms[output_idx] = np.sqrt(np.mean(sum_squares / t))
            if sample_times is None:
                output_idx += 1
            elif t in sample_times:
                output_idx += 1

    return rms


@njit(cache=True, parallel=True)
def mean_squared_displacement(
    u0: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: int,
    mapping: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    sample_times: Optional[NDArray[np.int32]] = None,
    axis: int = 1,
    transient_time: int = 0,
) -> NDArray[np.float64]:
    """
    Compute the mean squared displacement (MSD) of a coordinate across multiple trajectories.

    The MSD is calculated as:
    MSD(t) = ⟨(x_i(t) - x_i(0))²⟩
    where ⟨·⟩ denotes the average over all trajectories.

    Parameters
    ----------
    u0 : NDArray[np.float64]
        Array of initial conditions (shape: (num_ic, num_dim))
    parameters : NDArray[np.float64]
        System parameters passed to mapping function
    total_time : int
        Total number of iterations (must be > transient_time)
    mapping : Callable[[NDArray, NDArray], NDArray]
        System evolution function: u_next = mapping(u, parameters)
    sample_times : Optional[NDArray[np.int64]], optional
        Specific time steps to record (default: record all steps)
    axis : int, optional
        Coordinate index to analyze (default: 1)
    transient_time : int, optional
        Initial iterations to discard (default: 0)

    Returns
    -------
    NDArray[np.float64]
        Array of MSD values at requested times

    Raises
    ------
    ValueError
        If total_time ≤ transient_time
        If sample_times contains invalid values
        If axis is invalid

    Notes
    -----
    - Uses parallel processing over initial conditions
    - For normal diffusion, MSD grows linearly with time
    - The output length matches len(sample_times) if provided, else (total_time - transient_time)
    """
    # Input validation

    num_ic = u0.shape[0]
    u = u0.copy()
    # Store initial values for MSD calculation
    initial_values = u0[:, axis].copy()

    # Initialize output array
    if sample_times is not None:
        output_size = len(sample_times)
    else:
        output_size = total_time - transient_time

    msd = np.zeros(output_size)
    output_idx = 0

    # Main evolution loop
    for t in range(1, total_time + 1):
        # Parallel evolution
        for i in prange(num_ic):
            u[i] = mapping(u[i], parameters)

        # Calculate and store MSD if past transient
        if t > transient_time:
            displacements = u[:, axis] - initial_values
            msd[output_idx] = np.mean(displacements**2)
            if sample_times is None:
                output_idx += 1
            elif t in sample_times:
                output_idx += 1

    return msd


@njit(cache=True)
def recurrence_times(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: int,
    mapping: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    eps: float,
    transient_time: Optional[int] = None,
) -> NDArray[np.float64]:
    """Compute recurrence times to a neighborhood of the initial condition.

    Parameters
    ----------
    u : NDArray[np.float64]
        Initial state vector (shape: `(neq,)`).
    parameters : NDArray[np.float64]
        System parameters passed to `mapping` and `jacobian`.
    total_time : int
        Total number of iterations to compute (must be > 0)
    mapping : Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]
        System mapping function (must be Numba-compatible)
    eps : float
        Size of the neighborhood (must be > 0)
    transient_time : Optional[int], optional
        Number of initial iterations to discard (default: None, no transient removal)

    Returns
    -------
    NDArray[np.float64]
        Array of recurrence times where:
        - Each element is the time between returns to the eps-neighborhood
        - Empty array if no recurrences occur

    Notes
    -----
    - A recurrence occurs when the trajectory enters the hypercube:
      [u-eps/2, u+eps/2]^d
    - Useful for analyzing:
      - Stickiness in Hamiltonian systems
      - Chaotic vs regular orbits
    - For meaningful results:
      - eps should be small but not smaller than numerical precision
      - total_time should be >> expected recurrence times
    """

    u = u.copy()

    if transient_time is not None:
        u = iterate_mapping(u, parameters, transient_time, mapping)

    lower_bound = u - eps / 2
    upper_bound = u + eps / 2

    # Initialize recurrence time and list
    rt = 0
    rts = []

    # Iterate over the total time
    for t in range(total_time):
        # Evolve the system
        u = mapping(u, parameters)

        # Increment the recurrence time
        rt += 1

        # Check if the state has entered the box
        if np.all(u >= lower_bound) and np.all(u <= upper_bound):
            rts.append(rt)
            rt = 0

    return np.array(rts)
