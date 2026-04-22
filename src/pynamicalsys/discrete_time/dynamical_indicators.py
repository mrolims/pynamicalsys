# dynamical_indicators.py

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

from typing import Callable, Optional, Tuple, Union

import numpy as np
from numba import njit
from numpy.typing import NDArray

from pynamicalsys.common.recurrence_quantification_analysis import (
    RTEConfig,
    build_recurrence_matrix,
    white_vertline_distr,
    calculate_threshold,
)
from pynamicalsys.common.time_series_metrics import hurst_exponent
from pynamicalsys.discrete_time.trajectory_analysis import (
    generate_trajectory,
    iterate_mapping,
)


def dig(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: int,
    mapping: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    transient_time: Optional[int] = None,
) -> float:
    """Compute the number of zeros after the decimal point (dig) of the weighted Birkhoff
    average convergence of a trajectory.

    Parameters
    ----------
    u : NDArray[np.float64]
        Initial condition of shape (d,)
    parameters : NDArray[np.float64]
        System parameters
    total_time : int
        Total number of iterations (must be even and >= 100)
    mapping : Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]
        System mapping function (must be Numba-compatible)
    func : Callable[[NDArray[np.float64]], NDArray[np.float64]]
        Observable function
    transient_time : Optional[int]
        Burn-in period to discard

    Returns
    -------
    float
        dig value (higher values indicate better convergence)

    Notes
    -----
    - Implements the weighted Birkhoff average method
    - Requires total_time to be even (split into two halves)
    - For reliable results, total_time should be >= 1000
    """

    u = u.copy()

    # Handle transient
    if transient_time is not None:
        if transient_time >= total_time:
            raise ValueError("transient_time must be < total_time")
        u = iterate_mapping(u, parameters, transient_time, mapping)
        sample_size = total_time - transient_time
    else:
        sample_size = total_time

    N = sample_size // 2
    if N < 2:
        raise ValueError("Effective sample size too small after transient removal")

    N = sample_size // 2

    t = np.arange(1, N) / N
    S = np.exp(-1 / (t * (1 - t))).sum()
    w = np.exp(-1 / (t * (1 - t))) / S

    # Weighted Birkhoff average for the first half of iterations
    time_series = generate_trajectory(u, parameters, N, mapping)
    WB0 = (w * func(time_series[:-1, :])).sum()

    # Weighted Birkhoff average for the second half of iterations
    u = time_series[-1, :]
    time_series = generate_trajectory(u, parameters, N, mapping)
    WB1 = (w * func(time_series[:-1, :])).sum()

    return -np.log10(abs(WB0 - WB1))


def hurst_exponent_wrapped(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: int,
    mapping: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    wmin: int = 2,
    transient_time: Optional[int] = None,
    return_last: bool = False,
) -> NDArray[np.float64]:
    u = u.copy()
    neq = len(u)
    H = np.zeros(neq)

    time_series = generate_trajectory(
        u, parameters, total_time, mapping, transient_time=transient_time
    )

    H = hurst_exponent(time_series, wmin=wmin)

    if return_last:
        result = np.zeros(2 * neq)
        result[:neq] = H
        result[neq:] = time_series[-1, :]
        return result
    else:
        return H


def finite_time_hurst_exponent(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: int,
    finite_time: int,
    mapping: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    wmin: int = 2,
    return_points: bool = False,
) -> Union[NDArray[np.float64], Tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """
    Compute finite-time Hurst exponents for a dynamical system.

    Parameters
    ----------
    u : NDArray[np.float64]
        Initial condition vector of shape (n,).
    parameters : NDArray[np.float64]
        Parameters passed to the mapping function.
    total_time : int
        Total number of iterations used to generate the trajectory.
    finite_time : int
        Length of each analysis window (iterations).
    mapping : Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]
        A function that defines the system dynamics, i.e., how `u` evolves over time given `parameters`.
    wmin : int, optional
        Minimum window size for the rescaled range calculation. Default is 2.

    Returns
    -------
    NDArray[np.float64]
        Array of estimated Hurst exponents for each window.

    Notes
    -----
    The function computes the Hurst exponent for non-overlapping windows of size `finite_time`.
    """

    u = u.copy()

    num_windows = total_time // finite_time
    H_values = np.zeros((num_windows, len(u)))
    phase_space_points = np.zeros((num_windows, len(u)))

    # Compute Hurst exponent for each window
    for i in range(num_windows):
        time_series = generate_trajectory(u, parameters, finite_time, mapping)
        H_values[i] = hurst_exponent(time_series, wmin=wmin)
        phase_space_points[i] = time_series[-1, :]
        u = time_series[-1, :]

    if return_points:
        return H_values, phase_space_points
    else:
        return H_values


@njit
def lagrangian_descriptors(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: int,
    mapping: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    backwards_mapping: Callable[
        [NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]
    ],
    mod: float = 1.0,
    transient_time: Optional[int] = None,
) -> NDArray[np.float64]:
    """Compute Lagrangian Descriptors (LDs) for a dynamical system.

    Parameters
    ----------
    u : NDArray[np.float64]
        Initial condition of shape (d,), where d is system dimension
    parameters : NDArray[np.float64]
        System parameters of shape (p,)
    total_time : int
        Total number of iterations (must be > 0)
    mapping : Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]
        Forward mapping function: u_{n+1} = mapping(u_n, parameters)
    backwards_mapping : Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]
        Backward mapping function: u_{n-1} = backwards_mapping(u_n, parameters)
    transient_time : Optional[int], optional
        Number of initial iterations to discard (default None)

    Returns
    -------
    NDArray[np.float64]
        Array of shape (2,) containing:
        - LDs[0]: Forward LD (sum of forward trajectory distances)
        - LDs[1]: Backward LD (sum of backward trajectory distances)

    Notes
    -----
    - LDs reveal phase space structures and invariant manifolds
    - Higher values indicate more "stretching" in phase space
    - For best results:
      - Use total_time >> 1 (typically 1000-10000)
      - Ensure mapping and backwards_mapping are exact inverses
    - Numba-optimized for performance

    Examples
    --------
    >>> # Basic usage
    >>> u0 = np.array([0.1, 0.2])
    >>> params = np.array([0.5, 1.0])
    >>> lds = lagrangian_descriptors(u0, params, 1000, fwd_map, bwd_map)
    >>> forward_ld, backward_ld = lds
    """
    # Initialize descriptors
    LDs = np.zeros(2)
    u_forward = u.copy()
    u_backward = u.copy()

    # Handle transient period
    if transient_time is not None:
        if transient_time >= total_time:
            return LDs  # Return zeros if no sample time remains

        # Evolve through transient
        for _ in range(transient_time):
            u_forward = mapping(u_forward, parameters)
            u_backward = backwards_mapping(u_backward, parameters)
        sample_size = total_time - transient_time
    else:
        sample_size = total_time

    # Main computation loop
    for _ in range(sample_size):
        # Forward evolution
        u_new_forward = mapping(u_forward, parameters)
        dx = abs(u_new_forward[0] - u_forward[0])
        if dx > mod / 2:
            dx = mod - dx
        dy = u_new_forward[1] - u_forward[1]
        LDs[0] += np.sqrt(dx**2 + dy**2)
        u_forward = u_new_forward

        # Backward evolution
        u_new_backward = backwards_mapping(u_backward, parameters)
        dx = abs(u_new_backward[0] - u_backward[0])
        if dx > mod / 2:
            dx = mod - dx
        dy = u_new_backward[1] - u_backward[1]
        LDs[1] += np.sqrt(dx**2 + dy**2)
        u_backward = u_new_backward

    return LDs


def RTE(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: int,
    mapping: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    transient_time: Optional[int] = None,
    **kwargs,
) -> Union[float, Tuple]:
    """
    Calculate Recurrence Time Entropy (RTE) for a dynamical system.

    RTE quantifies the complexity of a system by analyzing the distribution
    of white vertical lines, i.e., the gap between two diagonal lines.
    Higher entropy indicates more complex dynamics.

    Parameters
    ----------
    u : NDArray[np.float64]
        Initial state vector (shape: (neq,))
    parameters : NDArray[np.float64]
        System parameters passed to mapping function
    total_time : int
        Number of iterations to simulate
    mapping : Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]
        System evolution function: u_next = mapping(u, parameters)
    transient_time : Optional[int], default=None
        Time to wait before starting RTE calculation.
    **kwargs
        Configuration parameters (see RTEConfig)

    Returns
    -------
    Union[float, Tuple]
        - Base case: RTE value (float)
        - With optional returns: List containing [RTE, *requested_additional_data]

    Raises
    ------
    ValueError
        - If invalid metric specified
        - If trajectory generation fails

    Notes
    -----
    - Implements the method described in [1]
    - For optimal results:
        - Use total_time > 1000 for reliable statistics
        - Typical threshold values: 0.05-0.3
        - Set lmin=1 to include single-point recurrences

    References
    ----------
    [1] M. R. Sales, M. Mugnaine, J. Szezech, José D., R. L. Viana, I. L. Caldas, N. Marwan, and J. Kurths, Stickiness and recurrence plots: An entropy-based approach, Chaos: An Interdisciplinary Journal of Nonlinear Science 33, 033140 (2023)
    """

    u = u.copy()

    # Configuration handling
    config = RTEConfig(**kwargs)

    if transient_time is not None:
        u = iterate_mapping(u, parameters, transient_time, mapping)
        total_time -= transient_time

    # Generate trajectory
    try:
        time_series = generate_trajectory(u, parameters, total_time, mapping)
    except Exception as e:
        raise ValueError(f"Trajectory generation failed: {str(e)}")

    eps = calculate_threshold(time_series, config)

    # Recurrence matrix calculation
    recmat = build_recurrence_matrix(time_series, float(eps), metric=config.metric)

    # White line distribution
    P = white_vertline_distr(recmat, wmin=config.lmin)
    P = P[P > 0]  # Remove zeros
    P /= P.sum()  # Normalize

    # Entropy calculation
    rte = -np.sum(P * np.log(P))

    # Prepare output
    result = [rte]
    if config.return_final_state:
        result.append(time_series[-1])
    if config.return_recmat:
        result.append(recmat)
    if config.return_p:
        result.append(P)

    return result[0] if len(result) == 1 else tuple(result)


def finite_time_RTE(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: int,
    finite_time: int,
    mapping: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    return_points: bool = False,
    **kwargs,
) -> Union[NDArray[np.float64], Tuple[NDArray[np.float64], NDArray[np.float64]]]:
    # Validate window size
    if finite_time > total_time:
        raise ValueError(
            f"finite_time ({finite_time}) exceeds available samples ({total_time})"
        )

    num_windows = total_time // finite_time
    RTE_values = np.zeros(num_windows)
    phase_space_points = np.zeros((num_windows, u.shape[0]))

    for i in range(num_windows):
        result = RTE(
            u, parameters, finite_time, mapping, return_final_state=True, **kwargs
        )
        if isinstance(result, tuple):
            RTE_values[i], u_new = result
            phase_space_points[i] = u
            u = u_new.copy()

    if return_points:
        return RTE_values, phase_space_points
    else:
        return RTE_values
