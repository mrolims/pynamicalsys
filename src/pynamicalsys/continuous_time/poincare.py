# poincare.py

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
from numba import njit, prange

from pynamicalsys.common.types import flow_t
from pynamicalsys.continuous_time.step import evolve_system, step


@njit
def generate_poincare_section(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    num_intersections: int,
    equations_of_motion: flow_t,
    transient_time: np.float64 | None,
    time_step: np.float64,
    atol: np.float64,
    rtol: np.float64,
    integrator,
    section_index: int,
    section_value: np.float64,
    crossing: int,
) -> NDArray[np.float64]:
    """
    Generate a Poincaré section for a single initial condition.

    Parameters
    ----------
    u : NDArray[np.float64]
        Initial state vector.
    parameters : NDArray[np.float64]
        System parameters passed to the equations of motion.
    num_intersections : int
        Number of section intersections to record.
    equations_of_motion : flow_t
        Continuous-time vector field with signature
        `(time, u, parameters) -> dudt`.
    transient_time : np.float64 | None
        Initial integration time discarded before recording intersections.
    time_step : np.float64
        Initial integration step size.
    atol : np.float64
        Absolute tolerance used by adaptive integrators.
    rtol : np.float64
        Relative tolerance used by adaptive integrators.
    integrator : callable
        Step method with unified signature returning
        `(u_next, time_next, time_step_next, accept)`.
    section_index : int
        Index of the coordinate defining the section.
    section_value : np.float64
        Section value for the selected coordinate.
    crossing : int
        Crossing orientation selector:
        - `0` records all crossings
        - `+1` records only positive crossings
        - `-1` records only negative crossings

    Returns
    -------
    NDArray[np.float64]
        Array of shape `(num_intersections, neq + 1)` whose first column
        contains the crossing time and whose remaining columns contain the
        interpolated phase-space point on the section.

    Notes
    -----
    Crossings are detected when the selected coordinate changes sign relative to
    `section_value` between two successive integration steps. The crossing time
    and state are estimated by linear interpolation between the two bracketing
    states.
    """
    neq = len(u)
    section_points = np.zeros((num_intersections, neq + 1), dtype=np.float64)
    count = 0

    u = u.copy()
    if transient_time is not None:
        u = evolve_system(
            u=u,
            parameters=parameters,
            total_time=transient_time,
            equations_of_motion=equations_of_motion,
            time_step=time_step,
            atol=atol,
            rtol=rtol,
            integrator=integrator,
        )
        time = transient_time
    else:
        time = np.float64(0.0)

    time_step_prev = time_step
    time_prev = time
    u_prev = u.copy()

    while count < num_intersections:
        u_new, time_new, time_step_new = step(
            time=time_prev,
            u=u_prev,
            parameters=parameters,
            equations_of_motion=equations_of_motion,
            time_step=time_step_prev,
            atol=atol,
            rtol=rtol,
            integrator=integrator,
        )

        if (u_prev[section_index] - section_value) * (
            u_new[section_index] - section_value
        ) < 0.0:
            lam = (section_value - u_prev[section_index]) / (
                u_new[section_index] - u_prev[section_index]
            )

            t_cross = time_new - time_step_prev + lam * time_step_prev
            u_cross = (1.0 - lam) * u_prev + lam * u_new
            velocity = equations_of_motion(t_cross, u_cross, parameters)[section_index]

            if crossing == 0 or np.sign(velocity) == crossing:
                section_points[count, 0] = t_cross
                section_points[count, 1:] = u_cross
                count += 1

        time_prev = time_new
        time_step_prev = time_step_new
        u_prev = u_new

    return section_points


@njit(parallel=True)
def ensemble_poincare_section(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    num_intersections: int,
    equations_of_motion: flow_t,
    transient_time: np.float64 | None,
    time_step: np.float64,
    atol: np.float64,
    rtol: np.float64,
    integrator,
    section_index: int,
    section_value: np.float64,
    crossing: int,
) -> NDArray[np.float64]:
    """
    Generate Poincaré sections for an ensemble of initial conditions.

    Parameters
    ----------
    u : NDArray[np.float64]
        Array of initial conditions with shape `(num_ic, neq)`.
    parameters : NDArray[np.float64]
        System parameters passed to the equations of motion.
    num_intersections : int
        Number of section intersections to record for each trajectory.
    equations_of_motion : flow_t
        Continuous-time vector field with signature
        `(time, u, parameters) -> dudt`.
    transient_time : np.float64 | None
        Initial integration time discarded before recording intersections.
    time_step : np.float64
        Initial integration step size.
    atol : np.float64
        Absolute tolerance used by adaptive integrators.
    rtol : np.float64
        Relative tolerance used by adaptive integrators.
    integrator : callable
        Step method with unified signature returning
        `(u_next, time_next, time_step_next, accept)`.
    section_index : int
        Index of the coordinate defining the section.
    section_value : np.float64
        Section value for the selected coordinate.
    crossing : int
        Crossing orientation selector:
        - `0` records all crossings
        - `+1` records only positive crossings
        - `-1` records only negative crossings

    Returns
    -------
    NDArray[np.float64]
        Array of shape `(num_ic, num_intersections, neq + 1)` containing the
        Poincaré section of each initial condition.
    """
    num_ic, neq = u.shape
    section_points = np.zeros((num_ic, num_intersections, neq + 1), dtype=np.float64)

    for i in prange(num_ic):
        section_points[i] = generate_poincare_section(
            u=u[i],
            parameters=parameters,
            num_intersections=num_intersections,
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

    return section_points
