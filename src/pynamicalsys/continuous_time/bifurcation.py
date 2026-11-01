# bifurcation.py

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

from pynamicalsys.common.types import flow_t
from pynamicalsys.continuous_time.maxima_map import generate_maxima_map
from pynamicalsys.continuous_time.poincare import generate_poincare_section
from pynamicalsys.continuous_time.step_methods import estimate_initial_step
from pynamicalsys.continuous_time.stroboscopic import generate_stroboscopic_map

# Codes of the reduced maps, passed to the compiled sweeps.
POINCARE_SECTION = 0
STROBOSCOPIC_MAP = 1
MAXIMA_MAP = 2


@njit
def _map_points(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    num_points: int,
    map_code: int,
    equations_of_motion: flow_t,
    transient_time: np.float64,
    time_step: np.float64,
    estimate_step: bool,
    atol: np.float64,
    rtol: np.float64,
    integrator,
    section_index: int,
    section_value: np.float64,
    crossing: int,
    sampling_time: np.float64,
    maxima_index: int,
) -> NDArray[np.float64]:
    """
    Compute `num_points` points of the selected reduced map from `u`, as
    `poincare_section`, `stroboscopic_map`, or `maxima_map` would, including
    the estimate of the initial step of an adaptive integrator.
    """
    if estimate_step:
        time_step = estimate_initial_step(
            np.float64(0.0), u, parameters, equations_of_motion, atol=atol, rtol=rtol
        )

    if map_code == POINCARE_SECTION:
        return generate_poincare_section(
            u,
            parameters,
            num_points,
            equations_of_motion,
            transient_time,
            time_step,
            atol,
            rtol,
            integrator,
            section_index,
            section_value,
            crossing,
        )
    if map_code == STROBOSCOPIC_MAP:
        return generate_stroboscopic_map(
            u,
            parameters,
            num_points,
            sampling_time,
            equations_of_motion,
            transient_time,
            time_step,
            atol,
            rtol,
            integrator,
        )
    return generate_maxima_map(
        u,
        parameters,
        num_points,
        maxima_index,
        equations_of_motion,
        transient_time,
        time_step,
        atol,
        rtol,
        integrator,
    )


@njit(parallel=True)
def _sweep_parallel(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    param_index: int,
    param_values: NDArray[np.float64],
    num_points: int,
    map_code: int,
    observable_index: int,
    equations_of_motion: flow_t,
    transient_time: np.float64,
    time_step: np.float64,
    estimate_step: bool,
    atol: np.float64,
    rtol: np.float64,
    integrator,
    section_index: int,
    section_value: np.float64,
    crossing: int,
    sampling_time: np.float64,
    maxima_index: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    # Every parameter value starts from `u`, so the values are independent.
    num_values = param_values.shape[0]
    results = np.empty((num_values, num_points), dtype=np.float64)
    last_points = np.empty((num_values, u.shape[0]), dtype=np.float64)

    for i in prange(num_values):
        current_params = parameters.copy()
        current_params[param_index] = param_values[i]
        points = _map_points(
            u.copy(),
            current_params,
            num_points,
            map_code,
            equations_of_motion,
            transient_time,
            time_step,
            estimate_step,
            atol,
            rtol,
            integrator,
            section_index,
            section_value,
            crossing,
            sampling_time,
            maxima_index,
        )
        results[i] = points[:, 1 + observable_index]
        last_points[i] = points[-1, 1:]

    return results, last_points[-1].copy()


@njit
def _sweep_continuation(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    param_index: int,
    param_values: NDArray[np.float64],
    num_points: int,
    map_code: int,
    observable_index: int,
    equations_of_motion: flow_t,
    transient_time: np.float64,
    time_step: np.float64,
    estimate_step: bool,
    atol: np.float64,
    rtol: np.float64,
    integrator,
    section_index: int,
    section_value: np.float64,
    crossing: int,
    sampling_time: np.float64,
    maxima_index: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    # Each parameter value starts from the state of the last map point of the
    # previous one, so the values are computed in order.
    num_values = param_values.shape[0]
    results = np.empty((num_values, num_points), dtype=np.float64)
    current_params = parameters.copy()
    state = u.copy()

    for i in range(num_values):
        current_params[param_index] = param_values[i]
        points = _map_points(
            state,
            current_params,
            num_points,
            map_code,
            equations_of_motion,
            transient_time,
            time_step,
            estimate_step,
            atol,
            rtol,
            integrator,
            section_index,
            section_value,
            crossing,
            sampling_time,
            maxima_index,
        )
        results[i] = points[:, 1 + observable_index]
        state = points[-1, 1:].copy()

    return results, state


def bifurcation_diagram(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    param_index: int,
    param_values: NDArray[np.float64],
    num_points: int,
    map_code: int,
    observable_index: int,
    equations_of_motion: flow_t,
    transient_time: np.float64,
    time_step: np.float64,
    estimate_step: bool,
    atol: np.float64,
    rtol: np.float64,
    integrator,
    continuation: bool = False,
    section_index: int = 0,
    section_value: np.float64 = np.float64(0.0),
    crossing: int = 0,
    sampling_time: np.float64 = np.float64(1.0),
    maxima_index: int = 0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Sweep one parameter of a continuous-time system and record one coordinate
    of a reduced map at each value.

    Parameters
    ----------
    u : NDArray[np.float64]
        Initial condition of shape `(neq,)`.
    parameters : NDArray[np.float64]
        Full parameter array; the entry at `param_index` is overwritten.
    param_index : int
        Index of the swept parameter.
    param_values : NDArray[np.float64]
        Values of the swept parameter.
    num_points : int
        Number of map points recorded at each parameter value.
    map_code : int
        `POINCARE_SECTION`, `STROBOSCOPIC_MAP`, or `MAXIMA_MAP`.
    observable_index : int
        State coordinate recorded at each map point.
    equations_of_motion : flow_t
        Continuous-time vector field with signature
        `(time, u, parameters) -> dudt`.
    transient_time : np.float64
        Integration time discarded at each parameter value.
    time_step : np.float64
        Integration step size, used as it is unless `estimate_step` is True.
    estimate_step : bool
        If True, the initial step of the adaptive integrator is estimated at
        each parameter value from its starting state.
    atol, rtol : np.float64
        Tolerances of adaptive integrators.
    integrator : callable
        Step method with unified signature returning
        `(u_next, time_next, time_step_next, accept)`.
    continuation : bool, optional
        If True, each parameter value starts from the state of the last map
        point of the previous one, and the values are computed in order.
        Otherwise every value starts from `u`, and the values are computed in
        parallel.
    section_index, section_value, crossing : optional
        Section of `POINCARE_SECTION`.
    sampling_time : np.float64, optional
        Sampling time of `STROBOSCOPIC_MAP`.
    maxima_index : int, optional
        Coordinate of `MAXIMA_MAP`.

    Returns
    -------
    tuple[NDArray[np.float64], NDArray[np.float64]]
        `(results, last_state)`, where `results` has shape
        `(len(param_values), num_points)` and `last_state` is the state at the
        last map point of the last parameter value.
    """
    sweep = _sweep_continuation if continuation else _sweep_parallel
    return sweep(
        np.ascontiguousarray(u, dtype=np.float64),
        np.ascontiguousarray(parameters, dtype=np.float64),
        param_index,
        np.ascontiguousarray(param_values, dtype=np.float64),
        num_points,
        map_code,
        observable_index,
        equations_of_motion,
        transient_time,
        time_step,
        estimate_step,
        atol,
        rtol,
        integrator,
        section_index,
        np.float64(section_value),
        crossing,
        np.float64(sampling_time),
        maxima_index,
    )
