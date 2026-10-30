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

from typing import Callable

import numpy as np
from numba import njit
from numpy.typing import NDArray

from pynamicalsys.common.types import int_t
from pynamicalsys.discrete_time.trajectory import generate_trajectory


@njit
def _sweep(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    param_index: int,
    param_values: NDArray[np.float64],
    total_time: int_t,
    mapping: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    transient_time: int_t,
    continuation: bool,
    observable_index: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    # Compiled loop over the parameter values of `bifurcation_diagram`.
    state = u.copy()
    sample_size = total_time - transient_time
    num_points = param_values.shape[0]

    results = np.empty((num_points, sample_size), dtype=np.float64)
    current_params = parameters.copy()
    last_state = state.copy()

    for i in range(num_points):
        current_params[param_index] = param_values[i]

        trajectory = generate_trajectory(
            state,
            current_params,
            total_time,
            mapping,
            transient_time=transient_time,
        )

        for j in range(sample_size):
            results[i, j] = trajectory[j, observable_index]

        last_state = trajectory[-1].copy()
        if continuation:
            state = last_state.copy()

    return results, last_state


def bifurcation_diagram(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    param_index: int,
    param_range: NDArray[np.float64] | tuple[float, float, int],
    total_time: int_t,
    mapping: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    transient_time: int_t = 0,
    continuation: bool = False,
    return_last_state: bool = False,
    observable_index: int = 0,
) -> (
    tuple[NDArray[np.float64], NDArray[np.float64]]
    | tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
):
    """
    Compute a bifurcation diagram by sweeping a system parameter and recording
    an observable along the resulting trajectories.

    Parameters
    ----------
    u : NDArray[np.float64]
        Initial condition of shape `(system_dimension,)`.
    parameters : NDArray[np.float64]
        Parameter array of shape `(num_parameters,)`.
    param_index : int
        Index of the parameter to be varied.
    param_range : NDArray[np.float64] | tuple[float, float, int]
        Parameter values used in the sweep. It can be either:
        - a 1D array of parameter values, or
        - a tuple `(start, stop, num_points)` passed to `numpy.linspace`.
    total_time : int
        Number of iterations computed for each parameter value.
    mapping : Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]
        Mapping function of the system.
    transient_time : int, optional
        Number of initial iterations discarded for each parameter value.
        Default is `0`.
    continuation : bool, optional
        If True, use the final state from the previous parameter value as the
        initial condition for the next one.
    return_last_state : bool, optional
        If True, also return the final state from the last parameter value.
    observable_index : int, optional
        Index of the state coordinate recorded along each trajectory. Default
        is `0`.

    Returns
    -------
    tuple
        If `return_last_state` is False:
            `(param_values, results)`
        If `return_last_state` is True:
            `(param_values, results, last_state)`

        Here:
        - `param_values` has shape `(num_points,)`
        - `results` has shape `(num_points, sample_size)`
        - `last_state` has shape `(system_dimension,)`

    Notes
    -----
    The output `results[i, j]` stores the observable evaluated at the `j`-th
    post-transient iterate for the `i`-th parameter value. The sweep runs in
    compiled code.
    """
    if isinstance(param_range, tuple):
        param_values = np.linspace(param_range[0], param_range[1], param_range[2])
    else:
        param_values = np.ascontiguousarray(param_range, dtype=np.float64)

    results, last_state = _sweep(
        u,
        np.asarray(parameters, dtype=np.float64),
        param_index,
        param_values,
        total_time,
        mapping,
        0 if transient_time is None else transient_time,
        continuation,
        observable_index,
    )

    if return_last_state:
        return param_values, results, last_state

    return param_values, results
