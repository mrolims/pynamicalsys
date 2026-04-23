# rte.py

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


from typing import Any

import numpy as np
from numpy.typing import NDArray

from pynamicalsys.common.recurrence_quantification_analysis import (
    RTEConfig,
    build_recurrence_matrix,
    calculate_threshold,
    white_vertline_distr,
)
from pynamicalsys.common.types import flow_t
from pynamicalsys.continuous_time.maxima_map import generate_maxima_map
from pynamicalsys.continuous_time.poincare import generate_poincare_section
from pynamicalsys.continuous_time.stroboscopic import generate_stroboscopic_map


def recurrence_time_entropy(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    num_points: int,
    transient_time: np.float64 | None,
    equations_of_motion: flow_t,
    time_step: np.float64,
    atol: np.float64,
    rtol: np.float64,
    integrator,
    map_type: str,
    section_index: int | None,
    section_value: np.float64 | None,
    crossing: int | None,
    sampling_time: np.float64 | None,
    maxima_index: int | None,
    **kwargs: Any,
) -> (
    float
    | tuple[float, NDArray[np.float64]]
    | tuple[float, NDArray[np.uint8]]
    | tuple[float, NDArray[np.float64], NDArray[np.uint8]]
    | tuple[float, NDArray[np.float64], NDArray[np.float64]]
    | tuple[float, NDArray[np.uint8], NDArray[np.float64]]
    | tuple[float, NDArray[np.float64], NDArray[np.uint8], NDArray[np.float64]]
):
    """
    Compute the recurrence time entropy (RTE) from a reduced continuous-time map.

    Parameters
    ----------
    u : NDArray[np.float64]
        Initial state vector.
    parameters : NDArray[np.float64]
        System parameters passed to the equations of motion.
    num_points : int
        Number of map points used in the recurrence analysis.
    transient_time : np.float64 | None
        Initial integration time discarded before generating the reduced map.
    equations_of_motion : flow_t
        Continuous-time vector field with signature
        `(time, u, parameters) -> dudt`.
    time_step : np.float64
        Initial integration step size.
    atol : np.float64
        Absolute tolerance used by adaptive integrators.
    rtol : np.float64
        Relative tolerance used by adaptive integrators.
    integrator : callable
        Step method with unified signature returning
        `(u_next, time_next, time_step_next, accept)`.
    map_type : str
        Reduced map used to generate the recurrence data:
        - `"PS"` for Poincaré section
        - `"SM"` for stroboscopic map
        - `"MM"` for maxima map
    section_index : int | None
        Coordinate index defining the Poincaré section when `map_type="PS"`.
    section_value : np.float64 | None
        Section value for the Poincaré section when `map_type="PS"`.
    crossing : int | None
        Crossing orientation selector for the Poincaré section when `map_type="PS"`.
    sampling_time : np.float64 | None
        Sampling interval for the stroboscopic map when `map_type="SM"`.
    maxima_index : int | None
        Index of the state variable whose maxima are used when `map_type="MM"`.
    **kwargs : Any
        Additional keyword arguments passed to `RTEConfig`, including:
        - `metric`
        - `std_metric`
        - `threshold`
        - `threshold_mode`
        - `threshold_std`
        - `lmin`
        - `return_final_state`
        - `return_recmat`
        - `return_p`

    Returns
    -------
    float or tuple
        The RTE value, optionally followed by:
        - the final reduced-map point
        - the recurrence matrix
        - the white-vertical-line distribution

    Notes
    -----
    The reduced data are generated first from the selected map. The recurrence
    threshold is then computed from `RTEConfig`, the recurrence matrix is built,
    the white-vertical-line distribution is extracted, and the entropy

    - `RTE = -sum(P * log(P))`

    is evaluated from the normalized distribution.
    """
    config = RTEConfig(**kwargs)

    if map_type == "PS":
        if section_index is None or section_value is None or crossing is None:
            raise ValueError(
                "section_index, section_value, and crossing must be provided when map_type='PS'"
            )

        points = generate_poincare_section(
            u=u,
            parameters=parameters,
            num_intersections=num_points,
            equations_of_motion=equations_of_motion,
            transient_time=transient_time,
            time_step=time_step,
            atol=atol,
            rtol=rtol,
            integrator=integrator,
            section_index=section_index,
            section_value=section_value,
            crossing=crossing,
        )
        data = points[:, 1:]
        data = np.delete(data, section_index, axis=1)

    elif map_type == "SM":
        if sampling_time is None:
            raise ValueError("sampling_time must be provided when map_type='SM'")

        points = generate_stroboscopic_map(
            u=u,
            parameters=parameters,
            num_intersections=num_points,
            sampling_time=sampling_time,
            equations_of_motion=equations_of_motion,
            transient_time=transient_time,
            time_step=time_step,
            atol=atol,
            rtol=rtol,
            integrator=integrator,
        )
        data = points[:, 1:]

    elif map_type == "MM":
        if maxima_index is None:
            raise ValueError("maxima_index must be provided when map_type='MM'")

        points = generate_maxima_map(
            u=u,
            parameters=parameters,
            num_peaks=num_points,
            maxima_index=maxima_index,
            equations_of_motion=equations_of_motion,
            transient_time=transient_time,
            time_step=time_step,
            atol=atol,
            rtol=rtol,
            integrator=integrator,
        )
        data = points[:, 1:]

    else:
        raise ValueError("map_type must be 'PS', 'SM', or 'MM'")

    eps = calculate_threshold(data, config)
    recmat = build_recurrence_matrix(data, eps, metric=config.metric)

    P = white_vertline_distr(recmat, wmin=config.lmin)
    P = P[P > 0]
    P /= P.sum()

    rte = -np.sum(P * np.log(P))

    result = [rte]
    if config.return_final_state:
        result.append(points[-1, 1:])
    if config.return_recmat:
        result.append(recmat)
    if config.return_p:
        result.append(P)

    return result[0] if len(result) == 1 else tuple(result)
