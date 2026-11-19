# pseudo_orbits_bollt.py

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

from typing import Callable

import numpy as np
from numba import njit
from numpy.typing import NDArray

from pynamicalsys.common.types import int_t, map_t, numeric_t
from pynamicalsys.discrete_time.periodic_orbits import _states_close


@njit
def _is_inside(
    state: NDArray[np.float64],
    exits: NDArray[np.float64],
) -> bool:
    """
    Check whether a state lies inside the box defined by `exits`.

    Parameters
    ----------
    state : NDArray[np.float64]
        State vector of shape `(system_dimension,)`.
    exits : NDArray[np.float64]
        Array of shape `(system_dimension, 2)` with the lower and upper
        bounds of each component. Use +-inf for unbounded components.

    Returns
    -------
    bool
        True if every component is within its bounds.
    """
    for i in range(state.shape[0]):
        if state[i] < exits[i, 0] or state[i] > exits[i, 1]:
            return False
    return True


@njit
def escape_time(
    state: NDArray[np.float64],
    parameters: NDArray[np.float64],
    mapping: map_t,
    exits: NDArray[np.float64],
    max_escape_time: int_t = 1000,
) -> int:
    """
    Number of iterations before a state leaves the box defined by `exits`.

    Returns
    -------
    int
        Escape time, capped at `max_escape_time`.
    """
    for t in range(max_escape_time):
        if not _is_inside(state, exits):
            return t
        state = mapping(state, parameters)
    return max_escape_time


@njit
def _lifetime_gradient(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    mapping: map_t,
    exits: NDArray[np.float64],
    max_escape_time: int_t,
    delta: numeric_t,
    n_dim: int,
) -> NDArray[np.float64]:
    """
    Central finite-difference gradient of the escape time with respect to
    the first `n_dim` components of `u`.
    """
    grad = np.zeros(u.shape[0], dtype=np.float64)

    for i in range(n_dim):
        u_plus = u.copy()
        u_minus = u.copy()
        u_plus[i] += delta
        u_minus[i] -= delta
        t_plus = escape_time(
            u_plus, parameters, mapping, exits, max_escape_time
        )
        t_minus = escape_time(
            u_minus, parameters, mapping, exits, max_escape_time
        )
        grad[i] = (t_plus - t_minus) / (2.0 * delta)

    return grad


@njit
def refine_gradient(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    mapping: map_t,
    exits: NDArray[np.float64],
    goal: int_t,
    max_escape_time: int_t,
    h_max: numeric_t,
    n_dim: int,
    delta: numeric_t,
    n_trials: int,
) -> NDArray[np.float64]:
    """
    Increase the escape time of `u` by gradient ascent (approach A).

    The step size is doubled after each accepted step (up to `h_max`) and
    halved after each rejected step. `n_trials` is unused.

    Returns
    -------
    NDArray[np.float64]
        Perturbed state with escape time >= `goal`, or the best state
        reached before the search stalls.
    """
    state = u.copy()
    lifetime = escape_time(state, parameters, mapping, exits, max_escape_time)
    h = h_max

    while lifetime < goal:
        grad = _lifetime_gradient(
            state, parameters, mapping, exits, max_escape_time, delta, n_dim
        )
        norm = np.sqrt(np.sum(grad * grad))
        if norm == 0.0:
            break

        candidate = state + h * grad / norm
        new_lifetime = escape_time(
            candidate, parameters, mapping, exits, max_escape_time
        )

        if new_lifetime > lifetime:
            state = candidate
            lifetime = new_lifetime
            h = min(2.0 * h, h_max)
        else:
            h *= 0.5
            if h < 1e-10:
                break

    return state


@njit
def refine_sampling(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    mapping: map_t,
    exits: NDArray[np.float64],
    goal: int_t,
    max_escape_time: int_t,
    h_max: numeric_t,
    n_dim: int,
    delta: numeric_t,
    n_trials: int,
) -> NDArray[np.float64]:
    """
    Increase the escape time of `u` by random local sampling (approach B).

    At each radius `h`, `n_trials` candidates are drawn uniformly in a box
    of half-width `h` around the current state. The best candidate is
    accepted if it improves the escape time; otherwise `h` is halved.
    `delta` is unused.

    Returns
    -------
    NDArray[np.float64]
        Perturbed state with escape time >= `goal`, or the best state
        reached before the radius collapses.
    """
    state = u.copy()
    lifetime = escape_time(state, parameters, mapping, exits, max_escape_time)
    h = h_max

    while lifetime < goal and h > 1e-10:
        best = state
        best_lifetime = lifetime

        for _ in range(n_trials):
            candidate = state.copy()
            for i in range(n_dim):
                candidate[i] += h * (2.0 * np.random.random() - 1.0)
            t = escape_time(
                candidate, parameters, mapping, exits, max_escape_time
            )
            if t > best_lifetime:
                best = candidate
                best_lifetime = t

        if best_lifetime > lifetime:
            state = best
            lifetime = best_lifetime
        else:
            h *= 0.5

    return state


@njit
def _detect_cycle(
    trajectory: NDArray[np.float64],
    i: int,
    n_dim: int,
    tolerance: numeric_t,
    max_freq: int,
) -> int:
    """
    Return the cycle length if `trajectory[i]` was visited more than
    `max_freq` times before (first `n_dim` components), otherwise -1.
    """
    hits = 0
    last = -1

    for j in range(i - 1, -1, -1):
        if _states_close(
            trajectory[j, :n_dim], trajectory[i, :n_dim], tolerance
        ):
            if last < 0:
                last = j
            hits += 1

    if hits > max_freq:
        return i - last
    return -1


@njit
def build_pseudo_orbit(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    mapping: map_t,
    exits: NDArray[np.float64],
    refine: Callable[..., NDArray[np.float64]],
    total_time: int_t = 10000,
    min_lifetime: int_t = 50,
    max_escape_time: int_t = 1000,
    h_max: numeric_t = 0.1,
    n_dim: int = 2,
    delta: numeric_t = 1e-6,
    n_trials: int = 50,
    tolerance: numeric_t = 1e-10,
    max_freq: int = 10,
    seed: int = 0,
) -> tuple[NDArray[np.float64], int]:
    """
    Build a pseudo-orbit that remains inside `exits` for at least
    `min_lifetime` iterations from every point (Bollt method).

    A pseudo-orbit (or ``epsilon-chain``) is a sequence of states ``{z_i}``
    satisfying ``||z_{i+1} - F(z_i)|| < epsilon`` at every step. Rather than
    searching strictly for periodic trajectories, this routine applies small
    state perturbations so that each point maintains a escape lifetime of at
    least `min_lifetime` iterations within `exits`, tracing an approximately
    invariant segment in phase space. An auxiliary recurrence check may also
    identify an approximate cycle length.

    Parameters
    ----------
    u : NDArray[np.float64]
        Initial condition of shape `(system_dimension,)`.
    parameters : NDArray[np.float64]
        System parameters.
    mapping : map_t
        System mapping function.
    exits : NDArray[np.float64]
        Bounds of shape `(system_dimension, 2)`.
    refine : Callable[..., NDArray[np.float64]]
        Perturbation routine: `refine_gradient` or `refine_sampling`.
    total_time : int_t, optional
        Number of points in the pseudo-orbit.
    min_lifetime : int_t, optional
        Minimum escape time required at each point.
    max_escape_time : int_t, optional
        Cap on the escape-time computation.
    h_max : numeric_t, optional
        Maximum perturbation size.
    n_dim : int, optional
        Number of leading components that may be perturbed. Auxiliary
        components (e.g. a drifting parameter and time) are left untouched.
    delta : numeric_t, optional
        Finite-difference step (used by `refine_gradient`).
    n_trials : int, optional
        Candidates per radius (used by `refine_sampling`).
    tolerance : numeric_t, optional
        Absolute tolerance used in the recurrence test.
    max_freq : int, optional
        Visits that trigger cycle detection after 80% of the orbit. Values
        <= 0 disable the detection.
    seed : int, optional
        Seed of the random generator.

    Returns
    -------
    tuple[NDArray[np.float64], int]
        Pseudo-orbit of shape `(total_time, system_dimension)` and the
        detected cycle length (-1 if no cycle was detected).

    Notes
    -----
    Checks and validation are expected to be done in the wrapper.

    References
    ----------
    .. [1] E. M. Bollt, "The Path Towards a Longer Life: On Invariant Sets and
       the Escape Time Landscape", International Journal of Bifurcation and
       Chaos, 15(5), 1615--1624 (2005).
    """
    np.random.seed(seed)

    trajectory = np.zeros((total_time, u.shape[0]), dtype=np.float64)
    trajectory[0] = u
    start_check = int(0.8 * total_time)

    for i in range(1, total_time):
        state = trajectory[i - 1]
        lifetime = escape_time(
            state, parameters, mapping, exits, max_escape_time
        )

        if lifetime >= min_lifetime:
            trajectory[i] = mapping(state, parameters)
        else:
            trajectory[i] = refine(
                state,
                parameters,
                mapping,
                exits,
                min_lifetime,
                max_escape_time,
                h_max,
                n_dim,
                delta,
                n_trials,
            )

        if max_freq > 0 and i > start_check:
            period = _detect_cycle(trajectory, i, n_dim, tolerance, max_freq)
            if period > 0:
                for j in range(i + 1, total_time):
                    trajectory[j] = trajectory[j - period]
                return trajectory, period

    return trajectory, -1
