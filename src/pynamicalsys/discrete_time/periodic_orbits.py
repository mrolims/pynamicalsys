from typing import Callable

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

from pynamicalsys.common.types import int_t, map_t, numeric_t
from pynamicalsys.discrete_time.trajectory import iterate_mapping
from pynamicalsys.discrete_time.symmetry import generate_symmetry_points


@njit
def _states_close(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    tolerance: numeric_t,
) -> bool:
    """
    Check whether two state vectors are equal within a componentwise tolerance.

    Parameters
    ----------
    x : NDArray[np.float64]
        First state vector of shape `(system_dimension,)`.
    y : NDArray[np.float64]
        Second state vector of shape `(system_dimension,)`.
    tolerance : numeric_t
        Absolute tolerance used in the comparison.

    Returns
    -------
    bool
        True if `|x_i - y_i| <= tolerance` for every component, False otherwise.
    """
    for i in range(x.shape[0]):
        if np.abs(x[i] - y[i]) > tolerance:
            return False
    return True


@njit
def period_counter(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    mapping: map_t,
    total_time: int_t = 5000,
    transient_time: int_t | None = None,
    tolerance: numeric_t = 1e-10,
    min_period: int = 1,
    max_period: int = 1000,
    stability_checks: int = 3,
) -> int:
    """
    Estimate the smallest detected period of an orbit from state recurrences.

    The routine evolves the system and looks for repeated returns of the state
    to its initial post-transient value within a given tolerance. A candidate
    period is accepted only after it is detected repeatedly
    `stability_checks` times.

    Parameters
    ----------
    u : NDArray[np.float64]
        Initial condition of shape `(system_dimension,)`.
    parameters : NDArray[np.float64]
        System parameters.
    mapping : map_t
        System mapping function.
    total_time : int_t, optional
        Maximum number of iterations used in the search.
    transient_time : int_t | None, optional
        Number of initial iterations discarded before the search.
    tolerance : numeric_t, optional
        Absolute tolerance used to detect recurrence.
    min_period : int, optional
        Minimum admissible period.
    max_period : int, optional
        Maximum admissible period.
    stability_checks : int, optional
        Number of identical consecutive detections required before accepting a
        period.

    Returns
    -------
    int
        Detected period. Returns `-1` if no valid period is found within the
        search window.

    Notes
    -----
    Checks and validation are expected to be done in the wrapper.
    """
    state = u.copy()

    if transient_time is not None:
        state = iterate_mapping(state, parameters, transient_time, mapping)
        sample_size = total_time - transient_time
    else:
        sample_size = total_time

    state_ini = state.copy()
    detected_periods = np.full(stability_checks, -1, dtype=np.int64)
    num_hits = 0
    p = 1

    for _ in range(sample_size):
        state = mapping(state, parameters)

        if _states_close(state, state_ini, tolerance):
            detected_periods[num_hits % stability_checks] = p
            num_hits += 1

            if num_hits >= stability_checks:
                same = True
                reference = detected_periods[0]
                for i in range(1, stability_checks):
                    if detected_periods[i] != reference:
                        same = False
                        break

                if same and min_period <= reference <= max_period:
                    return int(reference)

            p = 0

        p += 1

    return -1


@njit
def is_periodic(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    mapping: map_t,
    period: int,
    tolerance: numeric_t = 1e-10,
    transient_time: int_t | None = None,
) -> bool:
    """
    Check whether a point is periodic with a prescribed period.

    Parameters
    ----------
    u : NDArray[np.float64]
        Initial condition of shape `(system_dimension,)`.
    parameters : NDArray[np.float64]
        System parameters.
    mapping : map_t
        System mapping function.
    period : int
        Period to test.
    tolerance : numeric_t, optional
        Absolute tolerance used in the periodicity check.
    transient_time : int_t | None, optional
        Number of initial iterations discarded before testing periodicity.

    Returns
    -------
    bool
        True if the orbit returns to the same state after `period` iterations,
        within the specified tolerance.

    Notes
    -----
    Checks and validation are expected to be done in the wrapper.
    """
    state = u.copy()

    if transient_time is not None:
        state = iterate_mapping(state, parameters, transient_time, mapping)

    state_periodic = iterate_mapping(state.copy(), parameters, period, mapping)

    return _states_close(state, state_periodic, tolerance)


@njit(cache=True, parallel=True)
def scan_phase_space(
    grid_points: NDArray[np.float64],
    parameters: NDArray[np.float64],
    mapping: map_t,
    period: int,
    tolerance: numeric_t = 1e-10,
    transient_time: int_t | None = None,
) -> NDArray[np.float64]:
    """
    Scan a phase-space grid for periodic points of a prescribed period.

    Parameters
    ----------
    grid_points : NDArray[np.float64]
        Grid of initial conditions with shape `(nx, ny, system_dimension)`.
    parameters : NDArray[np.float64]
        System parameters.
    mapping : map_t
        System mapping function.
    period : int
        Period to search for.
    tolerance : numeric_t, optional
        Absolute tolerance used in the periodicity test.
    transient_time : int_t | None, optional
        Number of initial iterations discarded before testing periodicity.

    Returns
    -------
    NDArray[np.float64]
        Array of shape `(nx * ny, system_dimension)` whose nonzero rows
        correspond to detected periodic points.

    Notes
    -----
    The compact filtering of zero rows is intentionally left to the caller.
    Checks and validation are expected to be done in the wrapper.
    """
    nx = grid_points.shape[0]
    ny = grid_points.shape[1]
    n_dim = grid_points.shape[2]

    result = np.zeros((nx * ny, n_dim), dtype=np.float64)

    for i in prange(nx):
        for j in range(ny):
            k = i * ny + j
            u0 = grid_points[i, j, :].copy()

            if is_periodic(
                u0,
                parameters,
                mapping,
                period,
                tolerance=tolerance,
                transient_time=transient_time,
            ):
                result[k, :] = grid_points[i, j, :]

    return result


@njit
def scan_symmetry_line(
    points: NDArray[np.float64],
    parameters: NDArray[np.float64],
    mapping: map_t,
    period: int,
    tolerance: numeric_t = 1e-10,
    transient_time: int_t | None = None,
) -> NDArray[np.float64]:
    """
    Scan a set of points lying on a symmetry line or symmetry curve for periodic
    points of a prescribed period.

    Parameters
    ----------
    points : NDArray[np.float64]
        Array of candidate initial conditions of shape `(n_points, system_dimension)`.
    parameters : NDArray[np.float64]
        System parameters.
    mapping : map_t
        System mapping function.
    period : int
        Period to search for.
    tolerance : numeric_t, optional
        Absolute tolerance used in the periodicity test.
    transient_time : int_t | None, optional
        Number of initial iterations discarded before testing periodicity.

    Returns
    -------
    NDArray[np.float64]
        Array containing only the detected periodic points. If none are found,
        returns an empty array of shape `(0, system_dimension)`.

    Notes
    -----
    Checks and validation are expected to be done in the wrapper.
    """
    n_points = points.shape[0]
    n_dim = points.shape[1]

    periodic_points = np.empty((n_points, n_dim), dtype=np.float64)
    num_periodic_points = 0

    for i in range(n_points):
        u0 = points[i, :].copy()

        if is_periodic(
            u0,
            parameters,
            mapping,
            period,
            tolerance=tolerance,
            transient_time=transient_time,
        ):
            periodic_points[num_periodic_points, :] = points[i, :]
            num_periodic_points += 1

    if num_periodic_points == 0:
        return np.empty((0, n_dim), dtype=np.float64)

    return periodic_points[:num_periodic_points, :]


def find_periodic_orbit_symmetry_line(
    points: NDArray[np.float64],
    parameters: NDArray[np.float64],
    mapping: map_t,
    period: int,
    func: Callable[..., NDArray[np.float64]],
    axis: int,
    tolerance: numeric_t = 1e-10,
    max_iter: int_t = 1000,
    convergence_threshold: numeric_t = 1e-15,
    tolerance_decay_factor: numeric_t = 0.25,
    verbose: bool = False,
    transient_time: int_t | None = None,
) -> NDArray[np.float64]:
    """
    Refine a periodic point search along a symmetry line or symmetry curve.

    Parameters
    ----------
    points : NDArray[np.float64]
        Initial sampling array.
    parameters : NDArray[np.float64]
        System parameters.
    mapping : map_t
        System mapping function.
    period : int
        Period to search for.
    func : Callable[..., NDArray[np.float64]]
        Function defining the symmetry line or symmetry curve.
    axis : int
        Axis convention passed to `generate_symmetry_points`.
    tolerance : numeric_t, optional
        Initial tolerance used in the periodicity test.
    max_iter : int_t, optional
        Maximum number of refinement iterations.
    convergence_threshold : numeric_t, optional
        Convergence threshold for both orbit displacement and interval size.
    tolerance_decay_factor : numeric_t, optional
        Multiplicative factor used to reduce the tolerance during refinement.
    verbose : bool, optional
        If True, print iteration diagnostics.
    transient_time : int_t | None, optional
        Number of initial iterations discarded before testing periodicity.

    Returns
    -------
    NDArray[np.float64]
        Approximation of the periodic orbit.

    Notes
    -----
    Checks and validation are expected to be done in the wrapper.
    """
    points = points.copy()
    points = generate_symmetry_points(points, func, axis, parameters)

    n_points = points.shape[0]
    n_dim = points.shape[1]

    periodic_orbit = np.zeros(n_dim, dtype=np.float64)

    for j in range(max_iter):
        periodic_points = scan_symmetry_line(
            points,
            parameters,
            mapping,
            period,
            tolerance=tolerance,
            transient_time=transient_time,
        )

        if len(periodic_points) == 0:
            if verbose:
                print(f"No periodic points found at iteration {j}")
            if j == 0:
                raise ValueError("No periodic points found in the initial grid")
            break

        periodic_orbit_new = np.zeros(n_dim, dtype=np.float64)
        periodic_orbit_new[0] = periodic_points[:, 0].mean()
        periodic_orbit_new[1] = periodic_points[:, 1].mean()

        x_range = (
            periodic_points[:, 0].min() + tolerance,
            periodic_points[:, 0].max() - tolerance,
        )
        y_range = (
            periodic_points[:, 1].min() + tolerance,
            periodic_points[:, 1].max() - tolerance,
        )

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
            periodic_orbit = periodic_orbit_new.copy()
            break

        periodic_orbit = periodic_orbit_new.copy()

        tolerance = max(
            tolerance * tolerance_decay_factor,
            delta_bounds[axis] / n_points,
        )

        if axis == 0:
            array = np.linspace(x_range[0], x_range[1], n_points)
        else:
            array = np.linspace(y_range[0], y_range[1], n_points)

        points = generate_symmetry_points(array, func, axis, parameters)

    return periodic_orbit


def find_periodic_orbit(
    grid_points: NDArray[np.float64],
    parameters: NDArray[np.float64],
    mapping: map_t,
    period: int,
    tolerance: numeric_t = 1e-10,
    max_iter: int_t = 1000,
    convergence_threshold: numeric_t = 1e-15,
    tolerance_decay_factor: numeric_t = 0.25,
    verbose: bool = False,
    transient_time: int_t | None = None,
) -> NDArray[np.float64]:
    """
    Find a periodic orbit through iterative grid refinement.

    Parameters
    ----------
    grid_points : NDArray[np.float64]
        Initial phase-space grid of shape `(nx, ny, 2)`.
    parameters : NDArray[np.float64]
        System parameters.
    mapping : map_t
        System mapping function.
    period : int
        Period to search for.
    tolerance : numeric_t, optional
        Initial periodicity tolerance.
    max_iter : int_t, optional
        Maximum number of refinement iterations.
    convergence_threshold : numeric_t, optional
        Convergence threshold for both orbit displacement and interval size.
    tolerance_decay_factor : numeric_t, optional
        Multiplicative factor used to reduce the tolerance during refinement.
    verbose : bool, optional
        If True, print iteration diagnostics.
    transient_time : int_t | None, optional
        Number of initial iterations discarded before testing periodicity.

    Returns
    -------
    NDArray[np.float64]
        Approximation of the periodic orbit of shape `(2,)`.

    Notes
    -----
    Checks and validation are expected to be done in the wrapper.
    """
    grid_points = grid_points.copy()
    grid_size_x = grid_points.shape[0]
    grid_size_y = grid_points.shape[1]

    periodic_orbit = np.zeros(2, dtype=np.float64)

    for j in range(max_iter):
        scan = scan_phase_space(
            grid_points,
            parameters,
            mapping,
            period,
            tolerance=tolerance,
            transient_time=transient_time,
        )

        nonzero_rows = np.any(scan != 0.0, axis=1)
        num_periodic_points = np.count_nonzero(nonzero_rows)

        if num_periodic_points == 0:
            if verbose:
                print(f"No periodic points found at iteration {j}")
            if j == 0:
                raise ValueError("No periodic points found in the initial grid")
            break

        periodic_points = scan[nonzero_rows]

        periodic_orbit_new = np.zeros(2, dtype=np.float64)
        periodic_orbit_new[0] = periodic_points[:, 0].mean()
        periodic_orbit_new[1] = periodic_points[:, 1].mean()

        x_range = (
            periodic_points[:, 0].min() + tolerance,
            periodic_points[:, 0].max() - tolerance,
        )
        y_range = (
            periodic_points[:, 1].min() + tolerance,
            periodic_points[:, 1].max() - tolerance,
        )

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
            periodic_orbit = periodic_orbit_new.copy()
            break

        periodic_orbit = periodic_orbit_new.copy()

        X = np.linspace(x_range[0], x_range[1], grid_size_x)
        Y = np.linspace(y_range[0], y_range[1], grid_size_y)
        X, Y = np.meshgrid(X, Y)

        grid_points = np.empty((grid_size_x, grid_size_y, 2), dtype=np.float64)
        grid_points[:, :, 0] = X
        grid_points[:, :, 1] = Y

        tolerance = max(
            tolerance * tolerance_decay_factor,
            delta_bounds[0] / grid_size_x + delta_bounds[1] / grid_size_y,
        )

    return periodic_orbit
