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
from numpy.typing import NDArray

from pynamicalsys.common.types import int_t, numeric_t, state_observable_t
from pynamicalsys.discrete_time.trajectory import generate_trajectory


def _default_state_observable(x: NDArray[np.float64]) -> numeric_t:
    return x[0]


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
    observable_fn: state_observable_t | None = None,
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
    observable_fn : state_observable_t | None, optional
        Observable applied to each state of the trajectory. If None, the first
        coordinate is used.

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
    post-transient iterate for the `i`-th parameter value.
    """
    state = u.copy()

    if isinstance(param_range, tuple):
        param_values = np.linspace(param_range[0], param_range[1], param_range[2])
    else:
        param_values = np.ascontiguousarray(param_range)

    sample_size = total_time - transient_time
    num_points = len(param_values)

    results = np.empty((num_points, sample_size), dtype=np.float64)
    current_params = parameters.copy()

    trajectory = np.empty((sample_size, state.shape[0]), dtype=np.float64)

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
            if observable_fn is None:
                results[i, j] = _default_state_observable(trajectory[j])
            else:
                results[i, j] = observable_fn(trajectory[j])

        if continuation:
            state = trajectory[-1].copy()

    if return_last_state:
        return param_values, results, trajectory[-1].copy()

    return param_values, results
