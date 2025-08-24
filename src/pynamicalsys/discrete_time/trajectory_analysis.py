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


@njit
def iterate_mapping(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: int,
    mapping: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    transient_time: Optional[int] = None,
) -> NDArray[np.float64]:
    """
    Iterate a dynamical system mapping function with optional transient handling.

    This function evolves a state vector through repeated application of a mapping function,
    with Numba-optimized performance. Useful for both simulation and transient removal.

    Parameters
    ----------
    u : NDArray[np.float64]
        Initial state vector of shape (neq,), where neq is the system dimension
    parameters : NDArray[np.float64]
        System parameters passed to the mapping function
    total_time : int
        Total number of iterations to perform (after any transient)
    mapping : Callable[[NDArray, NDArray], NDArray]
        System mapping function: u_next = mapping(u, parameters)
    transient_time : Optional[int], optional
        Number of initial iterations to discard as transient (default: None)

    Returns
    -------
    NDArray[np.float64]
        Final state vector after all iterations (shape: (neq,))

    Raises
    ------
    ValueError
        If total_time is not positive
        If transient_time is negative
    """
    # Input validation
    if total_time <= 0:
        raise ValueError("total_time must be positive")
    if transient_time is not None and transient_time < 0:
        raise ValueError("transient_time must be non-negative")

    # Handle transient
    if transient_time is not None:
        for _ in range(transient_time):
            u = mapping(u, parameters)

    # Main iteration
    for _ in range(total_time):
        u = mapping(u, parameters)

    return u


@njit
def generate_trajectory(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: int,
    mapping: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    transient_time: Optional[int] = None,
) -> NDArray[np.float64]:
    """
    Generate a trajectory for a dynamical system from a single initial condition.

    This Numba-optimized function efficiently computes the system's evolution while
    optionally discarding an initial transient period. The implementation minimizes
    memory allocations and maximizes computational performance.

    Parameters
    ----------
    u : NDArray[np.float64]
        Initial state vector (shape: (neq,)), where neq is the system dimension
    parameters : NDArray[np.float64]
        System parameters passed to the mapping function
    total_time : int
        Total number of iterations to compute (including transient if specified)
    mapping : Callable[[NDArray, NDArray], NDArray]
        System evolution function: u_next = mapping(u, parameters)
    transient_time : Optional[int], optional
        Number of initial iterations to discard (default: None)

    Returns
    -------
    NDArray[np.float64]
        Time series array of shape (sample_size, neq), where:
        - sample_size = total_time (if no transient)
        - sample_size = total_time - transient_time (with transient)

    Raises
    ------
    ValueError
        If total_time is not positive
        If transient_time exceeds total_time

    Notes
    -----
    - Memory efficient: Pre-allocates output array
    - Numerically stable: Works with both discrete and continuous systems
    - For continuous systems, ensure proper time scaling in the mapping function
    """
    # Input validation
    if total_time <= 0:
        raise ValueError("total_time must be positive")
    if transient_time is not None:
        if transient_time < 0:
            raise ValueError("transient_time must be non-negative")
        if transient_time >= total_time:
            raise ValueError("transient_time must be less than total_time")

    # Handle transient
    state = u.copy()
    if transient_time is not None:
        state = iterate_mapping(state, parameters, transient_time, mapping)
        sample_size = total_time - transient_time
    else:
        sample_size = total_time

    # Pre-allocate trajectory array
    neq = len(state)
    trajectory = np.empty((sample_size, neq))

    # Generate trajectory
    for i in range(sample_size):
        state = mapping(state, parameters)
        trajectory[i] = state

    return trajectory


@njit(cache=True, parallel=True)
def ensemble_trajectories(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: int,
    mapping: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    transient_time: Optional[int] = None,
) -> NDArray[np.float64]:
    """
    Generate parallelized ensemble trajectories for multiple initial conditions.

    This function efficiently computes trajectories for an ensemble of initial conditions
    using Numba's parallel processing capabilities. Each trajectory is computed independently,
    making it ideal for large ensembles or parameter studies.

    Parameters
    ----------
    u : NDArray[np.float64]
        Array of initial conditions with shape (num_ic, neq), where:
        - num_ic: number of initial conditions
        - neq: system dimension (number of equations)
    parameters : NDArray[np.float64]
        System parameters (shape: arbitrary, passed to mapping)
    total_time : int
        Total iterations per trajectory (including transient if specified)
    mapping : Callable[[NDArray, NDArray], NDArray]
        System evolution function: u_next = mapping(u, parameters)
    transient_time : Optional[int], optional
        Initial iterations to discard per trajectory (default: None)

    Returns
    -------
    NDArray[np.float64]
        Concatenated trajectories of shape (num_ic * sample_size, neq), where:
        sample_size = total_time - (transient_time or 0)
        Trajectories are stacked in input order [IC1_t0..tN, IC2_t0..tN, ...]

    Raises
    ------
    ValueError
        If total_time ≤ transient_time
        If u is not 2D
        If parameters are incompatible with mapping

    Notes
    -----
    - Parallelization: Each IC processed independently using prange
    - Memory: Pre-allocates output array for optimal performance
    - Performance: ~10-100x faster than sequential for large ensembles
    - Post-processing: Use .reshape(num_ic, sample_size, neq) to separate trajectories
    """
    # Input validation
    if u.ndim != 2:
        raise ValueError("Initial conditions must be 2D array (num_ic, neq)")
    if transient_time is not None and transient_time >= total_time:
        raise ValueError("transient_time must be < total_time")

    num_ic, neq = u.shape
    sample_size = total_time - (transient_time if transient_time else 0)

    # Pre-allocate output array
    ensemble_ts = np.empty((num_ic * sample_size, neq))

    # Parallel trajectory generation
    for i in prange(num_ic):  # Parallel loop over initial conditions
        # Generate trajectory for i-th initial condition
        traj = generate_trajectory(
            u[i], parameters, total_time, mapping, transient_time
        )
        # Store in pre-allocated array
        ensemble_ts[i * sample_size : (i + 1) * sample_size] = traj

    return ensemble_ts


def bifurcation_diagram(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    param_index: int,
    param_range: Union[NDArray[np.float64], Tuple[float, float, int]],
    total_time: int,
    mapping: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    transient_time: Optional[int] = None,
    continuation: bool = False,
    return_last_state: bool = False,
    observable_fn: Optional[Callable[[NDArray[np.float64]], float]] = None,
) -> Union[
    Tuple[NDArray[np.float64], NDArray[np.float64]],
    Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
]:
    """
    Generate a bifurcation diagram by varying a system parameter and recording system states.

    Parameters
    ----------
    u : NDArray[np.float64]
        Initial state vector (shape: (neq,))
    parameters : NDArray[np.float64]
        System parameters (will be modified during sweep)
    param_index : int
        Index of parameter to vary in parameters array
    param_range : Union[NDArray[np.float64], Tuple[float, float, int]]
        Either:
        - Precomputed array of parameter values, or
        - Tuple of (start, stop, num_points) for linspace generation
    total_time : int
        Total iterations per parameter value (including transient)
    mapping : Callable[[NDArray, NDArray], NDArray]
        System evolution function: u_next = mapping(u, parameters)
    transient_time : Optional[int], optional
        Initial iterations to discard (default: total_time//10)
    observable_fn : Optional[Callable[[NDArray], float]], optional
        Function mapping state vector to plottable value (default: first coordinate)

    Returns
    -------
    Tuple[NDArray[np.float64], NDArray[np.float64]]
        - param_values: Array of parameter values used
        - observations: Array of shape (num_params, sample_size) containing observed values

    Notes
    -----
    - For periodic windows, increase total_time to capture full cycles
    - The default 10% transient discard is often sufficient for most systems
    - For higher-dimensional observations, provide a custom observable_fn
    """

    u = u.copy()

    # Process parameter range
    if isinstance(param_range, tuple):
        param_values = np.linspace(param_range[0], param_range[1], param_range[2])
    else:
        param_values = np.ascontiguousarray(param_range)

    # Set default transient time
    if transient_time is None:
        transient_time = total_time // 10
    sample_size = total_time - transient_time

    # Set default observable
    if observable_fn is None:

        def observable_fn(x):
            return x[0]

    # Pre-allocate results array
    num_points = len(param_values)
    results = np.empty((num_points, sample_size))
    current_params = parameters.copy()

    trajectory: NDArray[np.float64] = np.empty(
        (total_time - transient_time, u.shape[0])
    )

    # Main parameter sweep loop
    for i in range(num_points):
        current_params[param_index] = param_values[i]

        # Generate and process trajectory
        trajectory = generate_trajectory(
            u, current_params, total_time, mapping, transient_time
        )

        # Store observable values
        for j in range(sample_size):
            results[i, j] = observable_fn(trajectory[j])

        if continuation:
            u = trajectory[-1]  # Update state for next iteration

    if return_last_state:
        return param_values, results, trajectory[-1]
    else:
        return param_values, results


@njit
def period_counter(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    mapping: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    total_time: int = 5000,
    transient_time: Optional[int] = None,
    tolerance: float = 1e-10,
    min_period: int = 1,
    max_period: int = 1000,
    stability_checks: int = 3,
) -> int:
    """Detects the period of a dynamical system by analyzing state recurrence.

    This function determines the smallest period p where the system satisfies:
    ||x_{n+p} - x_n|| < tolerance for consecutive states after transients.

    Parameters
    ----------
    u : NDArray[np.float64]
        Initial state vector (shape: (neq,))
    parameters : NDArray[np.float64]
        System parameters passed to mapping function
    mapping : Callable[[NDArray, NDArray], NDArray]
        System evolution function: x_next = mapping(x, parameters)
    total_time : int, optional
        Maximum iterations to analyze (default: 5000)
    transient_time : Optional[int], optional
        Initial iterations to discard (default: None)
    tolerance : float, optional
        Numerical tolerance for period detection (default: 1e-10)
    min_period : int, optional
        Minimum period to consider (default: 1)
    max_period : int, optional
        Maximum period to consider (default: 1000)
    stability_checks : int, optional
        Number of consecutive period matches required (default: 3)

    Returns
    -------
    int
        Detected period, or -1 if no period found
    """

    # Make a copy of the provided initial condition to avoid modifying the original state
    state = u.copy()

    # Handle transient period
    if transient_time is not None:
        if transient_time >= total_time:
            return -1
        state = iterate_mapping(state, parameters, transient_time, mapping)
        sample_size = total_time - transient_time
    else:
        sample_size = total_time

    state_ini = state.copy()
    p = 1
    period = np.full(stability_checks, -1)  # Ring buffer for stability check
    idx = 0

    for _ in range(sample_size):
        state = mapping(state, parameters)

        if np.allclose(state, state_ini, atol=tolerance):
            period[idx % stability_checks] = p
            idx += 1

            # Check if last 'stability_checks' periods are equal and valid
            if idx >= stability_checks:
                same = True
                for i in range(1, stability_checks):
                    if period[i] != period[0]:
                        same = False
                        break
                if same and min_period <= period[0] <= max_period:
                    return period[0]
            p = 0  # reset period counter after a match

        p += 1

    return -1


@njit
def rotation_number(
    u: Union[NDArray[np.float64], Sequence[float], float],
    parameters: Union[NDArray[np.float64], Sequence[float], float],
    total_time: int,
    mapping: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    mod: float = 1.0,
) -> float:

    u_old = u.copy()

    rn = 0

    for i in range(total_time):
        u_new = mapping(u_old, parameters)
        rn += (u_new[0] - u_old[0]) % mod
        u_old = u_new.copy()

    rn /= total_time

    return rn


@njit
def escape_basin_and_time_entering(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    mapping: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    max_time: int,
    exits: NDArray[np.float64],
) -> Tuple[int, int]:
    """
    Track system evolution until it escapes through predefined exit regions.

    This function simulates a dynamical system until its state enters one of
    the specified exit regions or until max_time is reached. Useful for studying
    basin boundaries and escape dynamics.

    Parameters
    ----------
    u : NDArray[np.float64]
        Initial state vector (shape: (n_dim,))
    parameters : NDArray[np.float64]
        System parameters passed to mapping function
    mapping : Callable[[NDArray, NDArray], NDArray]
        System evolution function: u_next = mapping(u, parameters)
    max_time : int
        Maximum iterations to simulate (must be positive)
    exits : NDArray[np.float64]
        Center of the holes or exit regions, shape (n_exits, n_dim):
    tolerance : float, optional
        Numerical tolerance for boundary checks (default: 1e-12)

    Returns
    -------
    Tuple[int, int]
        - exit_index: 0-based exit region index (-1 if no escape)
        - escape_time: Iteration when escape occurred (max_time if no escape)

    Raises
    ------
    ValueError
        If max_time is not positive
        If exits array has invalid shape

    Notes
    -----
    - Uses Numba optimization for fast iteration
    - Exit checks are performed using vectorized comparisons
    - For conservative systems, consider larger max_time values
    """
    # Input validation
    if max_time <= 0:
        raise ValueError("max_time must be positive")
    if exits.ndim != 3 or exits.shape[2] != 2:
        raise ValueError("exits must have shape (n_exits, n_dim, 2)")

    n_exits = exits.shape[0]
    n_dim = exits.shape[1]
    u_current = u.copy()

    for time in range(1, max_time + 1):
        u_current = mapping(u_current, parameters)

        # Check all exit regions
        for exit_idx in range(n_exits):
            in_exit = True
            for dim in range(n_dim):
                lower = exits[exit_idx, dim, 0]
                upper = exits[exit_idx, dim, 1]
                if not (lower <= u_current[dim] <= upper):
                    in_exit = False
                    break

            if in_exit:
                return exit_idx, time  # 0-based indexing

    return -1, max_time


@njit
def escape_time_exiting(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    mapping: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    max_time: int,
    region_limits: NDArray[np.float64],
) -> Tuple[int, int]:
    """
    Track system evolution until it escapes a defined region through any boundary face.

    This function simulates a dynamical system until its state exits a specified
    hyperrectangular region or until max_time is reached. The escape face is
    identified for boundary analysis.

    Parameters
    ----------
    u : NDArray[np.float64]
        Initial state vector (shape: (n_dim,))
    parameters : NDArray[np.float64]
        System parameters passed to mapping function
    mapping : Callable[[NDArray, NDArray], NDArray]
        System evolution function: u_next = mapping(u, parameters)
    max_time : int
        Maximum iterations to simulate (must be positive)
    region_limits : NDArray[np.float64]
        Region boundaries of shape (n_dim, 2) where:
        region_limits[i,0] = lower bound in dimension i
        region_limits[i,1] = upper bound in dimension i

    Returns
    -------
    Tuple[int, int]
        - escape_time: Iteration when escape occurred (max_time if no escape)
        - face_index: Escaped face index (0 to 2*n_dim-1), or -1 if no escape
                     Faces are ordered as [dim0_lower, dim0_upper, dim1_lower,...]

    Raises
    ------
    ValueError
        If max_time is not positive
        If region_limits has invalid shape

    Notes
    -----
    - Face indexing: For dimension i, face 2*i is lower bound, 2*i+1 is upper
    - Uses Numba optimization for fast iteration
    - For conservative systems, consider larger max_time values
    """
    # Input validation
    if max_time <= 0:
        raise ValueError("max_time must be positive")
    if region_limits.ndim != 2 or region_limits.shape[1] != 2:
        raise ValueError("region_limits must have shape (n_dim, 2)")

    n_dim = region_limits.shape[0]
    u_current = u.copy()
    for time in range(1, max_time + 1):
        u_current = mapping(u_current, parameters)
        # Check all dimensions for boundary crossing
        for dim in range(n_dim):
            if u_current[dim] < region_limits[dim, 0]:
                return 2 * dim, time  # lower face escape
            if u_current[dim] > region_limits[dim, 1]:
                return 2 * dim + 1, time  # upper face escape

    return -1, max_time  # No escape


@njit
def survival_probability(
    escape_times: NDArray[np.int32],
    max_time: np.int32,
    min_time: int = 1,
    time_step: int = 1,
) -> Tuple[NDArray[np.int64], NDArray[np.float64]]:
    """
    Calculate the survival probability function S(t) from observed escape times.

    The survival probability S(t) represents the probability that a system remains
    in a given region beyond time t. This implementation uses efficient sorting
    and searching algorithms for optimal performance with large datasets.

    Parameters
    ----------
    escape_times : NDArray[np.int64]
        Array of escape times for each trajectory (must be ≥ 1)
    max_time : int
        Maximum time to evaluate (must be > min_time)
    min_time : int, optional
        Minimum time to evaluate (default: 1)
    time_step : int, optional
        Time resolution for evaluation (default: 1)

    Returns
    -------
    Tuple[NDArray[np.int64], NDArray[np.float64]]
        - t_values: Array of evaluation times
        - survival_probs: Corresponding survival probabilities S(t)

    Raises
    ------
    ValueError
        If max_time ≤ min_time
        If time_step ≤ 0
        If escape_times contains values < 1

    Notes
    -----
    - Implementation uses numpy's searchsorted for O(n log n) performance
    - Handles right-censored data (escape_times > max_time are treated as censored)
    - For smooth results with few samples, consider kernel density methods
    """
    # Input validation
    if max_time <= min_time:
        raise ValueError("max_time must be > min_time")
    if time_step <= 0:
        raise ValueError("time_step must be positive")
    if np.any(escape_times < 1):
        raise ValueError("All escape_times must be ≥ 1")

    # Filter and sort escape times
    valid_times = escape_times[(escape_times >= min_time) & (escape_times <= max_time)]
    valid_times = np.sort(valid_times)
    n_samples = len(escape_times)
    n_valid = len(valid_times)

    # Handle case where all times exceed max_time
    if n_valid == 0:
        t_values = np.arange(min_time, max_time + 1, time_step)
        return t_values, np.ones_like(t_values, dtype=np.float64)

    # Create evaluation points
    t_values = np.arange(min_time, max_time + 1, time_step)

    # Find insertion indices for each t in sorted escape_times
    indices = np.searchsorted(valid_times, t_values, side="right")

    # Compute Kaplan-Meier survival probability
    survival_probs = 1.0 - indices / n_samples

    return t_values, survival_probs


@njit
def is_periodic(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    mapping: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    period: int,
    tolerance: float = 1e-10,
    transient_time: Optional[int] = None,
) -> bool:
    """Check if a point is periodic with given period under the system mapping.

    Parameters
    ----------
    u : NDArray[np.float64]
        Initial condition of shape (d,)
    parameters : NDArray[np.float64]
        System parameters of shape (p,)
    mapping : Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]
        System mapping function (must be Numba-compatible)
    period : int
        Period to check (must be > 0)
    tolerance : float, optional
        Tolerance for periodicity check (default: 1e-10)
    transient_time : Optional[int], optional
        Initial iterations to discard (default: None)

    Returns
    -------
    bool
        True if f^period(u) ≈ u within tolerance
        False otherwise

    Notes
    -----
    - Checks if mapping^period(u) ≈ u
    - For fixed points, use period=1
    - The check is performed component-wise
    - Lower tolerance gives stricter periodicity check
    - For reliable results:
      - tolerance should be > numerical error accumulation
      - period should be < system's expected maximum period
    """

    # Compute mapped point
    u_periodic = u.copy()

    if transient_time is not None:
        # Apply transient mapping
        u_periodic = iterate_mapping(
            u_periodic,
            parameters,
            transient_time,
            mapping,
            transient_time=transient_time,
        )

    u_periodic = iterate_mapping(u_periodic, parameters, period, mapping)

    # Check periodicity component-wise
    periodic = True
    for i in range(u.shape[0]):
        if abs(u[i] - u_periodic[i]) > tolerance:
            periodic = False
            break

    return periodic


@njit(cache=True, parallel=True)
def scan_phase_space(
    grid_points: NDArray[np.float64],
    parameters: NDArray[np.float64],
    mapping: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    period: int,
    tolerance: float = 1e-10,
    transient_time: Optional[int] = None,
) -> NDArray[np.float64]:
    """Scan phase space grid for periodic orbits of specified period.

    Parameters
    ----------
    grid_points : NDArray[np.float64]
        3D array of initial conditions with shape (nx, ny, d) where:
        - nx: number of x-axis grid points
        - ny: number of y-axis grid points
        - d: system dimension (must be ≥ 2)
    parameters : NDArray[np.float64]
        System parameters of shape (p,)
    mapping : Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]
        Numba-compatible system mapping function
    period : int
        Period to search for (must be ≥ 1)
    tolerance : float, optional
        Tolerance for periodicity check (default: 1e-10)
    transient_time : Optional[int], optional
        Initial iterations to discard (default: None)

    Returns
    -------
    NDArray[np.float64]
        Array of periodic points found, shape (n_found, d)

    Raises
    ------
    ValueError
        If grid_points has invalid dimensions
        If period is not positive
        If tolerance is not positive

    Notes
    -----
    - Uses parallel processing over grid points
    - Typical workflow:
      1. Create phase space grid with np.meshgrid
      2. Reshape into (nx, ny, d) array
      3. Call this function
    - Memory efficient - returns only found points
    """

    # Input validation
    if grid_points.ndim != 3:
        raise ValueError("grid_points must be 3D array (nx, ny, d)")

    nx = grid_points.shape[0]
    ny = grid_points.shape[1]
    n_dim = grid_points.shape[2]

    result = np.zeros((nx * ny, n_dim), dtype=np.float64)

    # Iterate over grid points
    for i in prange(nx):
        for j in range(ny):
            k = i * ny + j
            u = np.empty(n_dim)
            u[0] = grid_points[i, j, 0]
            u[1] = grid_points[i, j, 1]
            # Check if periodic
            if is_periodic(
                u,
                parameters,
                mapping,
                period,
                tolerance=tolerance,
                transient_time=transient_time,
            ):
                # Store periodic point
                result[k, :] = grid_points[i, j, :]
                # number_of_periodic_points += 1

    return result


@njit
def scan_symmetry_line(
    points: NDArray[np.float64],
    parameters: NDArray[np.float64],
    mapping: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    period: int,
    tolerance: float = 1e-10,
    transient_time: Optional[int] = None,
) -> NDArray[np.float64]:
    n_points = points.shape[0]
    n_dim = points.shape[1]

    periodic_points = np.empty((n_points, n_dim), dtype=np.float64)
    number_of_periodic_points = 0

    for i in range(n_points):
        u = np.empty(n_dim)
        u[0] = points[i, 0]
        u[1] = points[i, 1]
        # Check if periodic
        if is_periodic(
            u,
            parameters,
            mapping,
            period,
            tolerance=tolerance,
            transient_time=transient_time,
        ):
            # Store periodic point
            periodic_points[number_of_periodic_points, :] = points[i, :]
            number_of_periodic_points += 1

    # If no periodic points found, return empty array
    if number_of_periodic_points == 0:
        return np.empty((0, n_dim), dtype=np.float64)
    # Resize result to only include found periodic points
    return periodic_points[:number_of_periodic_points, :]


def find_periodic_orbit_symmetry_line(
    points: NDArray[np.float64],
    parameters: NDArray[np.float64],
    mapping: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    period: int,
    func: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    axis: int,
    tolerance: float = 1e-10,
    max_iter: int = 1000,
    convergence_threshold: float = 1e-15,
    tolerance_decay_factor: float = 1 / 4,
    verbose: bool = False,
    transient_time: Optional[int] = None,
) -> NDArray[np.float64]:

    #  Make a copy of the points to avoid modifying the original
    points = points.copy()
    points = generate_symmetry_points(points, func, axis, parameters)
    n_points = points.shape[0]
    n_dim = points.shape[1]

    # Initialize periodic orbit
    periodic_orbit = np.zeros(n_dim)

    for j in range(max_iter):
        # Find periodic points in current grid
        periodic_points = scan_symmetry_line(
            points,
            parameters,
            mapping,
            period,
            tolerance=tolerance,
            transient_time=transient_time,
        )

        # If no periodic points are found, exit the loop
        if len(periodic_points) == 0:
            if verbose:
                print(f"No periodic points found at iteration {j}")
            if j == 0:
                raise ValueError("No periodic points found in the initial grid")
            break

        # Calculate the new periodic orbit
        periodic_orbit_new = np.zeros(n_dim)
        periodic_orbit_new[0] = periodic_points[:, 0].mean()
        periodic_orbit_new[1] = periodic_points[:, 1].mean()

        # Define the new phase space limits
        x_range = (
            periodic_points[:, 0].min() + tolerance,
            periodic_points[:, 0].max() - tolerance,
        )
        y_range = (
            periodic_points[:, 1].min() + tolerance,
            periodic_points[:, 1].max() - tolerance,
        )

        # Check convergence
        delta_orbit = np.abs(periodic_orbit_new - periodic_orbit)
        delta_bounds = np.abs(
            np.array([x_range[1] - x_range[0], y_range[1] - y_range[0]])
        )

        if verbose:
            print(
                f"Iter {j}: Δorbit={delta_orbit}, Δbounds={delta_bounds}, tol={tolerance:.2e}"
            )

        if np.all(delta_orbit < convergence_threshold) and np.all(
            delta_bounds < convergence_threshold
        ):
            if verbose:
                print(f"Converged at iteration {j}")
            break
        # Update the periodic orbit
        periodic_orbit = periodic_orbit_new.copy()

        # Update the tolerance for the next iteration
        tolerance = max(
            tolerance * tolerance_decay_factor, (delta_bounds[axis] / n_points)
        )

        if axis == 0:
            array = np.linspace(x_range[0], x_range[1], n_points)
        else:
            array = np.linspace(y_range[0], y_range[1], n_points)
        # Update the grid points
        points = generate_symmetry_points(array, func, axis, parameters)

    return periodic_orbit


def find_periodic_orbit(
    grid_points: NDArray[np.float64],
    parameters: NDArray[np.float64],
    mapping: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    period: int,
    tolerance: float = 1e-10,
    max_iter: int = 1000,
    convergence_threshold: float = 1e-15,
    tolerance_decay_factor: float = 1 / 4,
    verbose: bool = False,
    transient_time: Optional[int] = None,
) -> NDArray[np.float64]:
    """Find periodic orbits through iterative grid refinement.

    Parameters
    ----------
    grid_points : NDArray[np.float64]
        3D array of initial conditions with shape (nx, ny, 2)
    parameters : NDArray[np.float64]
        System parameters of shape (p,)
    mapping : Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]
        System mapping function
    period : int
        Period of orbits to find (must be ≥ 1)
    tolerance : float, optional
        Initial periodicity tolerance (default: 1e-10)
    max_iter : int, optional
        Maximum refinement iterations (default: 1000)
    convergence_threshold : float, optional
        Convergence threshold for orbit position (default: 1e-15)
    tolerance_decay_factor : float, optional
        Tolerance reduction factor per iteration (default: 0.25)
    verbose : bool, optional
        Print convergence info if True (default: False)
    transient_time : Optional[int], optional
        Initial iterations to discard (default: None)

    Returns
    -------
    NDArray[np.float64]
        Found periodic orbit of shape (2,)

    Raises
    ------
    ValueError
        If no periodic points found in initial grid
        If invalid grid dimensions
        If invalid period

    Notes
    -----
    - Implements iterative grid refinement:
      1. Scan current grid for periodic points
      2. Calculate mean position and new search bounds
      3. Refine grid around found points
      4. Repeat until convergence
    - For best results:
      - Start with coarse grid covering expected region
      - Use moderate tolerance (1e-8 to 1e-12)
      - Monitor convergence with verbose=True
    """

    # Make a copy of the grid points to avoid modifying the original
    grid_points = grid_points.copy()
    grid_size_x = grid_points.shape[0]
    grid_size_y = grid_points.shape[1]

    # Initialize periodic orbit
    periodic_orbit = np.zeros(2)

    for j in range(max_iter):

        # Scan the phase space grid for periodic points
        scan = scan_phase_space(
            grid_points,
            parameters,
            mapping,
            period,
            tolerance=tolerance,
            transient_time=transient_time,
        )

        # Check if any periodic points were found
        nonzero_rows = np.any(scan != 0, axis=1)

        # Count non-zero rows to determine number of periodic points found
        number_of_periodic_points = np.count_nonzero(nonzero_rows)

        # If no periodic points are found, exit the loop
        if number_of_periodic_points == 0:
            if verbose:
                print(f"No periodic points found at iteration {j}")
            if j == 0:
                raise ValueError("No periodic points found in the initial grid")
            break

        # Resize scan to only include found periodic points
        periodic_points = scan[nonzero_rows]

        # Calculate the new periodic orbit
        periodic_orbit_new = np.zeros(2)
        periodic_orbit_new[0] = periodic_points[:, 0].mean()
        periodic_orbit_new[1] = periodic_points[:, 1].mean()

        # Define the new phase space limits
        x_range = (
            periodic_points[:, 0].min() + tolerance,
            periodic_points[:, 0].max() - tolerance,
        )
        y_range = (
            periodic_points[:, 1].min() + tolerance,
            periodic_points[:, 1].max() - tolerance,
        )

        # Update the grid points
        X = np.linspace(x_range[0], x_range[1], grid_size_x)
        Y = np.linspace(y_range[0], y_range[1], grid_size_y)
        X, Y = np.meshgrid(X, Y)
        grid_points = np.empty((grid_size_x, grid_size_y, 2))
        grid_points[:, :, 0] = X
        grid_points[:, :, 1] = Y

        # Check convergence
        delta_orbit = np.abs(periodic_orbit_new - periodic_orbit)
        delta_bounds = np.abs(
            np.array([x_range[1] - x_range[0], y_range[1] - y_range[0]])
        )

        if verbose:
            print(
                f"Iter {j}: Δorbit={delta_orbit}, Δbounds={delta_bounds}, tol={tolerance:.2e}"
            )

        if np.all(delta_orbit < convergence_threshold) and np.all(
            delta_bounds < convergence_threshold
        ):
            if verbose:
                print(f"Converged after {j} iterations")
            break

        # Update the periodic orbit
        periodic_orbit = periodic_orbit_new.copy()

        # Update the tolerance for the next iteration
        tolerance = max(
            tolerance * tolerance_decay_factor,
            (delta_bounds[0] / grid_size_x + delta_bounds[1] / grid_size_y),
        )

    return periodic_orbit


@njit
def eigenvalues_and_eigenvectors(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    mapping: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    jacobian: Callable[
        [NDArray[np.float64], NDArray[np.float64], Callable], NDArray[np.float64]
    ],
    period: int,
    normalize: bool = True,
    sort_by_magnitude: bool = True,
) -> Tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Compute eigenvalues and eigenvectors of the Jacobian matrix for a periodic orbit.

    Parameters
    ----------
    u : NDArray[np.float64]
        Initial condition of shape (d,) where d is system dimension
    parameters : NDArray[np.float64]
        System parameters of shape (p,)
    mapping : Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]
        System mapping function
    jacobian : Callable[[NDArray[np.float64], NDArray[np.float64], Callable], NDArray[np.float64]]
        Function to compute Jacobian matrix
    period : int
        Period of the orbit (must be ≥ 1)
    normalize : bool, optional
        Whether to normalize eigenvectors (default: True)
    sort_by_magnitude : bool, optional
        Whether to sort by eigenvalue magnitude (default: True)

    Returns
    -------
    Tuple[NDArray[np.complex128], NDArray[np.complex128]]
        - eigenvalues: Array of eigenvalues (shape (d,))
        - eigenvectors: Array of eigenvectors (shape (d, d))
        (each column is an eigenvector)

    Raises
    ------
    ValueError
        If period is not positive
        If input dimensions are invalid

    Notes
    -----
    - Computes the nth iterated Jacobian matrix J = J_p * J_{p-1} * ... * J_1
    - Complex eigenvalues come in conjugate pairs
    - Eigenvectors indicate directions of stretching/contraction
    """
    # Input validation
    if period < 1:
        raise ValueError("period must be ≥ 1")
    if u.ndim != 1:
        raise ValueError("u must be 1D array")
    if jacobian is None:
        raise ValueError("Jacobian function must be provided")

    neq = len(u)
    J = np.eye(neq, dtype=np.complex128)
    current_u = u.copy()

    # Compute Jacobian matrix
    for _ in range(period):
        current_u = mapping(current_u, parameters)
        J = (
            np.asarray(jacobian(current_u, parameters, mapping), dtype=np.complex128)
            @ J
        )

    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eig(J)

    # Post-processing
    if normalize:
        for i in range(neq):
            norm = np.linalg.norm(eigenvectors[:, i])
            if norm > 0:
                eigenvectors[:, i] /= norm

    if sort_by_magnitude:
        idx = np.argsort(np.abs(eigenvalues))[::-1]  # Descending order
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

    return eigenvalues, eigenvectors


def classify_stability(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    mapping: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    jacobian: Callable[
        [NDArray[np.float64], NDArray[np.float64], Callable], NDArray[np.float64]
    ],
    period: int,
    threshold: float = 1.0,
    tol: float = 1e-8,
) -> Dict[str, Union[str, NDArray[np.complex128]]]:
    """
    Classify the local stability of a 2D periodic orbit in a discrete map.

    Parameters
    ----------
    u : (2,) ndarray
        Initial condition in 2D.
    parameters : ndarray
        System parameters.
    mapping : Callable
        Map function f(u, parameters).
    jacobian : Callable
        Jacobian function J(u, parameters, mapping).
    period : int
        Period of the orbit.
    threshold : float
        Radius of the unit circle (default: 1.0).
    tol : float
        Tolerance to determine closeness to the unit circle.

    Returns
    -------
    dict
        Dictionary with:
        - 'classification': str
        - 'eigenvalues': ndarray
        - 'eigenvectors': ndarray

    Raises
    ------
    ValueError
        If u is not 2D or if the Jacobian is not 2x2.
    """
    if u.shape != (2,):
        raise ValueError(
            "This function only supports 2D systems (u.shape must be (2,))."
        )

    # Compute eigenvalues
    eigenvalues, eigenvectors = eigenvalues_and_eigenvectors(
        u, parameters, mapping, jacobian, period
    )
    if eigenvalues.shape[0] != 2:
        raise ValueError("Jacobian must be 2x2 for this classification.")

    λ1, λ2 = eigenvalues
    abs_λ1, abs_λ2 = np.abs(λ1), np.abs(λ2)

    is_real = np.isreal(λ1) and np.isreal(λ2)

    # Classification logic
    if abs_λ1 < threshold - tol and abs_λ2 < threshold - tol:
        classification = "stable node" if is_real else "stable spiral"
    elif abs_λ1 > threshold + tol and abs_λ2 > threshold + tol:
        classification = "unstable node" if is_real else "unstable spiral"
    elif (abs_λ1 < threshold - tol and abs_λ2 > threshold + tol) or (
        abs_λ2 < threshold - tol and abs_λ1 > threshold + tol
    ):
        classification = "saddle"
    elif abs(abs_λ1 - threshold) <= tol and abs(abs_λ2 - threshold) <= tol:
        classification = "center" if is_real else "elliptic (quasi-periodic)"
    else:
        classification = "marginal or degenerate"

    return {
        "classification": classification,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
    }


def calculate_manifolds(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    forward_mapping: Callable[
        [NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]
    ],
    backward_mapping: Callable[
        [NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]
    ],
    jacobian: Callable[
        [NDArray[np.float64], NDArray[np.float64], Callable], NDArray[np.float64]
    ],
    period: int,
    delta: float = 1e-4,
    n_points: Union[NDArray[np.int32], List[int], int] = 100,
    iter_time: Union[List[int], int] = 100,
    stability: str = "unstable",
) -> List[np.ndarray]:
    """Calculate stable or unstable manifolds of a saddle periodic orbit.

    Parameters
    ----------
    u : NDArray[np.float64]
        Initial condition of periodic orbit (shape (2,))
    parameters : NDArray[np.float64]
        System parameters (shape (p,))
    forward_mapping : Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]
        Forward time system mapping
    backward_mapping : Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]
        Backward time system mapping
    jacobian : Callable[[NDArray[np.float64], NDArray[np.float64], Callable], NDArray[np.float64]]
        Jacobian computation function
    period : int
        Period of the orbit (must be ≥ 1)
    delta : float, optional
        Initial displacement from orbit (default: 1e-4)
    n_points : Union[List[int], int], optional
        Number of points per branch (default: 100)
    iter_time : Union[List[int], int], optional
        Iterations per branch (default: 100)
    stability : str, optional
        'stable' or 'unstable' manifold (default: 'unstable')

    Returns
    -------
    List[NDArray[np.float64]]
        List containing two arrays:
        - [0]: Upper branch manifold points
        - [1]: Lower branch manifold points
        Each array has shape (n_points * iter_time, 2)

    Raises
    ------
    ValueError
        If input is not a saddle point
        If invalid stability type
        If invalid point counts or iterations

    Notes
    -----
    - Works only for 2D systems
    - The periodic orbit must be a saddle point
    - Manifold quality depends on:
      - delta (smaller = closer to linear approximation)
      - n_points (more = smoother manifold)
      - iter_time (more = longer manifold)
    """
    # Validate and process n_points
    if isinstance(n_points, int):
        n_points = [n_points, n_points]
    elif len(n_points) != 2:
        raise ValueError("n_points must be int or list of 2 ints")
    n_points = [int(n) for n in n_points]
    if any(n < 1 for n in n_points):
        raise ValueError("n_points must be ≥ 1")

    # Validate and process iter_time
    if isinstance(iter_time, int):
        iter_time = [iter_time, iter_time]
    elif len(iter_time) != 2:
        raise ValueError("iter_time must be int or list of 2 ints")
    iter_time = [int(t) for t in iter_time]
    if any(t < 1 for t in iter_time):
        raise ValueError("iter_time must be ≥ 1")

    # Verify saddle point
    stability_info = classify_stability(
        u, parameters, forward_mapping, jacobian, period
    )
    if stability_info["classification"] != "saddle":
        raise ValueError(
            "Manifolds require saddle point (1 stable + 1 unstable direction)"
        )

    # Get eigenvectors
    eigenvectors: NDArray[np.complex128] = stability_info["eigenvectors"]
    vu = eigenvectors[:, 0]
    vs = eigenvectors[:, 1]

    # Select manifold type
    if stability == "unstable":
        v = vu
        mapping = forward_mapping
    elif stability == "stable":
        v = vs
        mapping = backward_mapping
    else:
        raise ValueError("stability must be 'stable' or 'unstable'")

    # Calculate eigenvector angle (ignore orientation)
    theta = np.arctan2(v[1].real, v[0].real) % np.pi

    def calculate_branch(y_sign):
        if y_sign == 1:
            # Upper branch
            branch = 0
        else:
            # Lower branch
            branch = 1
        """Calculate manifold branch in specified direction."""
        y_range = u[1], (u[1] + y_sign * delta * np.sin(theta))
        # y = np.logspace(np.log10(y_range[0]), np.log10(y_range[1]), n_points[0])
        y = np.linspace(y_range[0], y_range[1], n_points[branch])
        x = (y - u[1]) / np.tan(theta) + u[0]
        points = np.column_stack((x, y))
        return ensemble_trajectories(points, parameters, iter_time[branch], mapping)

    # Calculate both branches
    return [calculate_branch(+1), calculate_branch(-1)]  # Upper branch  # Lower branch


def generate_symmetry_points(
    array: NDArray[np.float64],
    func: Callable[..., NDArray[np.float64]],
    axis: int,
    *args: Any,
    **kwargs: Any,
) -> NDArray[np.float64]:
    """
    Generate points along a symmetry line or curve.

    Parameters:
        x_array (array-like): x-coordinates or y-coordinates depending on axis
        func: constant value (for horizontal/vertical) or function (for curve)
        axis (int): 0 for y = f(x), 1 for x = g(y)
        *args, **kwargs: extra parameters for the function if func is callable

    Returns:
        np.ndarray: 2D array of points [[x, y], [x, y], ...]
    """

    if not callable(func):
        raise TypeError(
            f"func must be a number or a callable function, got {type(func)}."
        )

    if axis == 0:
        # y = f(x)
        x_array = array.copy()
        y_array = func(x_array, *args, **kwargs)
    elif axis == 1:
        # x = g(y)
        y_array = np.asarray(array)
        x_array = func(y_array, *args, **kwargs)
    else:
        raise ValueError(f"Invalid axis {axis}. Use 0 for y = f(x), 1 for x = g(y).")

    return np.column_stack((x_array, y_array))


@njit(cache=True, parallel=True)
def ensemble_time_average(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    mapping: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    total_time: int,
    axis: int = 1,
) -> NDArray[np.float64]:

    u = u.copy()
    num_ic = u.shape[0]
    average = np.zeros(num_ic, dtype=np.float64)

    for i in prange(num_ic):
        for _ in range(total_time):
            u[i] = mapping(u[i], parameters)
            average[i] += u[i, axis]

    x_average = np.sum(average) / (num_ic * total_time)

    average = average - total_time * x_average

    return average
