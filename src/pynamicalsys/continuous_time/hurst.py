# hurst.py

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

from numpy.typing import NDArray
import numpy as np

from pynamicalsys.common.hurst import hurst_exponent
from pynamicalsys.common.types import flow_t
from pynamicalsys.continuous_time.maxima_map import generate_maxima_map
from pynamicalsys.continuous_time.poincare import generate_poincare_section
from pynamicalsys.continuous_time.stroboscopic import generate_stroboscopic_map


def hurst_exponent_wrapped(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    num_points: int,
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
    wmin: int = 2,
    transient_time: np.float64 | None = None,
) -> NDArray[np.float64]:
    """
    Compute the Hurst exponent from a reduced continuous-time map.

    Parameters
    ----------
    u : NDArray[np.float64]
        Initial state vector.
    parameters : NDArray[np.float64]
        System parameters passed to the equations of motion.
    num_points : int
        Number of reduced-map points used in the Hurst analysis.
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
        Reduced map used to generate the data:
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
    wmin : int, optional
        Minimum window size used in the Hurst exponent calculation.
    transient_time : np.float64 | None, optional
        Initial integration time discarded before generating the reduced map.

    Returns
    -------
    NDArray[np.float64]
        Hurst exponent values computed from the reduced map data.

    Notes
    -----
    The reduced data are generated first from the selected map. The time column
    is discarded before the Hurst exponent is computed. In the Poincaré case,
    the section coordinate is also removed from the reduced data.
    """
    u = u.copy()

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

    return hurst_exponent(data, wmin=wmin)
