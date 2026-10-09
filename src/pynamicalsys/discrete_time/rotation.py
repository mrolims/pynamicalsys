# rotation.py

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
from numba import njit
from numpy.typing import NDArray

from pynamicalsys.common.types import map_t, numeric_t, int_t
from pynamicalsys.discrete_time.birkhoff import (
    evaluate_observable,
    orbit_states,
    two_window_wba,
)

# Orbit classes returned by `classify_orbit`
CHAOTIC = 0
ISLAND = 1
CIRCLE = 2


@njit
def rotation_number(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: int_t,
    mapping: map_t,
    mod: numeric_t = 1.0,
) -> float:
    """
    Compute the rotation number of a trajectory from wrapped increments.

    Deprecated path, used when no frequency map is set. The rotation number is estimated as the time average of the wrapped increment
    of the first coordinate,
    ``[(x_{n+1} - x_n) mod mod]``,
    along a trajectory of length `total_time`.

    Parameters
    ----------
    u : NDArray[np.float64]
        Initial condition of shape `(system_dimension,)`.
    parameters : NDArray[np.float64]
        System parameters passed to the mapping function.
    total_time : time_t
        Number of iterations used in the average.
    mapping : map_t
        System mapping function.
    mod : numeric_t, optional
        Period used to wrap the increment of the first coordinate.
        The default is `1.0`.

    Returns
    -------
    float
        Estimated rotation number.

    Notes
    -----
    This low-level routine assumes all inputs were already validated by the wrapper.
    In particular:
    - `u` must be a 1D array,
    - `total_time` must be positive,
    - `mod` must be positive.
    """
    state_old = u.copy()
    rotation = 0.0

    for _ in range(total_time):
        state_new = mapping(state_old, parameters)
        rotation += (state_new[0] - state_old[0]) % mod
        state_old = state_new

    return float(rotation / total_time)


@njit
def rotation_number_average(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: int_t,
    mapping: map_t,
    frequency_map: map_t,
) -> float:
    """
    Compute the rotation number as the time average of the frequency map.

    The frequency map `Omega(u, parameters)` returns the unwrapped advance of
    the angle, in turns, in the iteration that starts at `u`, so that

        `omega = (1 / T) sum_{n=0}^{T-1} Omega(x_n)`.

    Parameters
    ----------
    u : NDArray[np.float64]
        Initial condition of shape `(system_dimension,)`.
    parameters : NDArray[np.float64]
        System parameters passed to the mapping and frequency map.
    total_time : int_t
        Number of iterations `T` used in the average.
    mapping : map_t
        System mapping function.
    frequency_map : map_t
        Frequency map.

    Returns
    -------
    float
        Estimated rotation number, in turns per iteration.
    """
    state = u.copy()
    rotation = 0.0
    for _ in range(total_time):
        rotation += frequency_map(state, parameters)
        state = mapping(state, parameters)

    return rotation / total_time


@njit
def rotation_number_wba(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: int_t,
    mapping: map_t,
    frequency_map: map_t,
) -> tuple[float, float]:
    """
    Compute the rotation number with the weighted Birkhoff average.

    The frequency map is averaged with the weighted Birkhoff average over the
    states `x_0, ..., x_{N-1}` and `x_N, ..., x_{2N-1}` (Sander and Meiss, 2020,
    Eq. 22). The rotation number is the mean of the two averages, and the
    number of digits in which they agree,

        `dig = -log10(|WB_0 - WB_1|)`,

    measures its convergence.

    Parameters
    ----------
    u : NDArray[np.float64]
        Initial condition of shape `(system_dimension,)`.
    parameters : NDArray[np.float64]
        System parameters passed to the mapping and frequency map.
    total_time : int_t
        Length `N` of each window. The trajectory is iterated `2 N` times.
    mapping : map_t
        System mapping function.
    frequency_map : map_t
        Frequency map returning the unwrapped advance of the angle, in turns,
        in the iteration that starts at the given state.

    Returns
    -------
    tuple[float, float]
        Estimated rotation number, in turns per iteration, and the
        convergence indicator `dig`.

    Notes
    -----
    This low-level routine assumes `total_time >= 2`.
    """
    values = np.empty(2 * total_time, dtype=np.float64)
    state = u.copy()
    for n in range(2 * total_time):
        values[n] = frequency_map(state, parameters)
        state = mapping(state, parameters)

    wb0, wb1 = two_window_wba(values, total_time)

    return 0.5 * (wb0 + wb1), -np.log10(abs(wb0 - wb1))


@njit
def smallest_denominator(x: float, delta: float) -> int:
    """
    Find the smallest denominator of a rational number close to `x`.

    Returns `q_min`, the smallest `q` such that some rational `p / q` lies in
    the interval `(x - delta, x + delta)`. The rational is found by descending
    the Stern-Brocot tree, whose first element inside the interval is the one
    with the smallest denominator (Sander and Meiss, 2020, Lemma 1 and
    Algorithm 2).

    Parameters
    ----------
    x : float
        Number to approximate.
    delta : float
        Half-width of the interval. Must be positive.

    Returns
    -------
    int
        The smallest denominator `q_min`.

    Notes
    -----
    Since `q_min` is invariant under integer shifts of `x`, only the
    fractional part of `x` is used.
    """
    x = x - np.floor(x)

    if x < delta or 1.0 - x < delta:
        return 1

    p_left, q_left = 0, 1
    p_right, q_right = 1, 1

    while True:
        p = p_left + p_right
        q = q_left + q_right

        if abs(x - p / q) < delta:
            return q

        if x < p / q:
            p_right, q_right = p, q
        else:
            p_left, q_left = p, q


@njit
def _frequency_values(
    states: NDArray[np.float64],
    parameters: NDArray[np.float64],
    frequency_map: map_t,
) -> NDArray[np.float64]:
    values = np.empty(states.shape[0], dtype=np.float64)
    for n in range(states.shape[0]):
        values[n] = frequency_map(states[n], parameters)

    return values


def classify_orbit(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: int_t,
    mapping: map_t,
    frequency_map: map_t,
    func,
    form: str,
    omega_threshold: numeric_t,
    dig_threshold: numeric_t,
    tol: numeric_t,
    dev_threshold: numeric_t,
) -> tuple[int, float, float, float]:
    """
    Classify an orbit as chaotic, in an island chain, or on a rotational circle.

    Two weighted Birkhoff averages are computed over the states
    `x_0, ..., x_{N-1}` and `x_N, ..., x_{2N-1}`: one of the observable `h`,
    which gives `dig_h`, and one of the frequency map, which gives the
    rotation number `omega` and its convergence `dig_omega`.

    Rationality of `omega` is decided as in Sander and Meiss (2020, Sec. 3.2).
    With `delta = 10^(-tol)`, `q_min` is the smallest denominator of a rational
    within `delta` of `omega`, and `dev = |log10(q_min) - tol / 2|`.

    The orbit is classified as follows:

    - If `dig_h >= dig_threshold`, the orbit is regular. It lies on a
      rotational circle if `dev < dev_threshold` and in an island chain
      otherwise.
    - If `dig_h < dig_threshold`, the orbit lies in an island chain if
      `dig_omega >= omega_threshold` and `log10(q_min) < tol / 2 - dev_threshold`.
      This covers island orbits on which the average of the observable
      converges slowly. Otherwise, the orbit is chaotic.

    Parameters
    ----------
    u : NDArray[np.float64]
        Initial condition of shape `(system_dimension,)`.
    parameters : NDArray[np.float64]
        System parameters passed to the mapping and frequency map.
    total_time : int_t
        Length `N` of each window. The trajectory is iterated `2 N` times.
    mapping : map_t
        System mapping function.
    frequency_map : map_t
        Frequency map, in turns per iteration.
    func : Callable
        Observable, as returned by `resolve_observable`.
    form : str
        Form of the observable, as returned by `resolve_observable`.
    omega_threshold : numeric_t
        Minimum `dig_omega` for the rotation number to be converged.
    dig_threshold : numeric_t
        Minimum `dig_h` for the orbit to be regular.
    tol : numeric_t
        Exponent of the tolerance `delta = 10^(-tol)` used to find `q_min`.
    dev_threshold : numeric_t
        Maximum `dev` for a regular orbit to lie on a rotational circle.

    Returns
    -------
    tuple[int, float, float, float]
        Orbit class (`CHAOTIC`, `ISLAND`, or `CIRCLE`), rotation number (NaN if
        chaotic), `dig_h`, and `dig_omega`.

    Notes
    -----
    This low-level routine assumes all inputs were already validated by the
    wrapper.
    """
    states = orbit_states(u, parameters, 2 * total_time, mapping)

    h0, h1 = two_window_wba(evaluate_observable(states, func, form), total_time)
    w0, w1 = two_window_wba(
        _frequency_values(states, parameters, frequency_map), total_time
    )

    with np.errstate(divide="ignore"):
        dig_h = float(-np.log10(abs(h0 - h1)))
        dig_omega = float(-np.log10(abs(w0 - w1)))
    omega = 0.5 * (w0 + w1)

    if not np.isfinite(omega):
        return CHAOTIC, np.nan, dig_h, dig_omega

    log_q_min = np.log10(smallest_denominator(omega, 10.0 ** (-tol)))

    if dig_h >= dig_threshold:
        if abs(log_q_min - 0.5 * tol) < dev_threshold:
            return CIRCLE, omega, dig_h, dig_omega
        return ISLAND, omega, dig_h, dig_omega

    if dig_omega >= omega_threshold and log_q_min < 0.5 * tol - dev_threshold:
        return ISLAND, omega, dig_h, dig_omega

    return CHAOTIC, np.nan, dig_h, dig_omega
