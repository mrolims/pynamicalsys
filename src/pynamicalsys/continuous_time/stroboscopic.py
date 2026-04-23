# stroboscopic.py

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

from numpy.typing import NDArray
import numpy as np
from numba import njit, prange

from pynamicalsys.common.types import flow_t
from pynamicalsys.continuous_time.step import evolve_system, step


@njit
def generate_stroboscopic_map(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    num_intersections: int,
    sampling_time: np.float64,
    equations_of_motion: flow_t,
    transient_time: np.float64 | None,
    time_step: np.float64,
    atol: np.float64,
    rtol: np.float64,
    integrator,
) -> NDArray[np.float64]:
    """
    Generate a stroboscopic map for a single initial condition.

    Parameters
    ----------
    u : NDArray[np.float64]
        Initial state vector.
    parameters : NDArray[np.float64]
        System parameters passed to the equations of motion.
    num_intersections : int
        Number of stroboscopic samples to record.
    sampling_time : np.float64
        Time interval between consecutive stroboscopic samples.
    equations_of_motion : flow_t
        Continuous-time vector field with signature
        `(time, u, parameters) -> dudt`.
    transient_time : np.float64 | None
        Initial integration time discarded before recording the map.
    time_step : np.float64
        Initial integration step size.
    atol : np.float64
        Absolute tolerance used by adaptive integrators.
    rtol : np.float64
        Relative tolerance used by adaptive integrators.
    integrator : callable
        Step method with unified signature returning
        `(u_next, time_next, time_step_next, accept)`.

    Returns
    -------
    NDArray[np.float64]
        Array of shape `(num_intersections, neq + 1)` whose first column
        contains the sampling times and whose remaining columns contain the
        interpolated state vectors.

    Notes
    -----
    The state is sampled at times separated by `sampling_time`. If the
    integrator overshoots a target sampling time, the state is linearly
    interpolated between the two bracketing integration points.
    """
    u = np.asarray(u, dtype=np.float64)
    neq = len(u)
    strobe_points = np.zeros((num_intersections, neq + 1), dtype=np.float64)

    if transient_time is not None:
        u_curr = evolve_system(
            u=u,
            parameters=parameters,
            total_time=transient_time,
            equations_of_motion=equations_of_motion,
            time_step=time_step,
            atol=atol,
            rtol=rtol,
            integrator=integrator,
        )
        time_curr = transient_time
    else:
        u_curr = u.copy()
        time_curr = np.float64(0.0)

    time_target = time_curr + sampling_time
    count = 0

    while count < num_intersections:
        u_prev = u_curr.copy()
        time_prev = time_curr

        while time_curr < time_target:
            u_curr, time_curr, time_step = step(
                time=time_curr,
                u=u_curr,
                parameters=parameters,
                equations_of_motion=equations_of_motion,
                time_step=time_step,
                atol=atol,
                rtol=rtol,
                integrator=integrator,
            )

        lam = (time_target - time_prev) / (time_curr - time_prev)
        strobe_points[count, 0] = time_target
        strobe_points[count, 1:] = (1.0 - lam) * u_prev + lam * u_curr

        count += 1
        time_target += sampling_time

    return strobe_points


@njit(parallel=True)
def ensemble_stroboscopic_map(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    num_intersections: int,
    sampling_time: np.float64,
    equations_of_motion: flow_t,
    transient_time: np.float64 | None,
    time_step: np.float64,
    atol: np.float64,
    rtol: np.float64,
    integrator,
) -> NDArray[np.float64]:
    """
    Generate stroboscopic maps for an ensemble of initial conditions.

    Parameters
    ----------
    u : NDArray[np.float64]
        Array of initial conditions with shape `(num_ic, neq)`.
    parameters : NDArray[np.float64]
        System parameters passed to the equations of motion.
    num_intersections : int
        Number of stroboscopic samples to record for each trajectory.
    sampling_time : np.float64
        Time interval between consecutive stroboscopic samples.
    equations_of_motion : flow_t
        Continuous-time vector field with signature
        `(time, u, parameters) -> dudt`.
    transient_time : np.float64 | None
        Initial integration time discarded before recording the map.
    time_step : np.float64
        Initial integration step size.
    atol : np.float64
        Absolute tolerance used by adaptive integrators.
    rtol : np.float64
        Relative tolerance used by adaptive integrators.
    integrator : callable
        Step method with unified signature returning
        `(u_next, time_next, time_step_next, accept)`.

    Returns
    -------
    NDArray[np.float64]
        Array of shape `(num_ic, num_intersections, neq + 1)` containing the
        stroboscopic map of each initial condition.
    """
    num_ic, neq = u.shape
    strobe_points = np.zeros((num_ic, num_intersections, neq + 1), dtype=np.float64)

    for i in prange(num_ic):
        strobe_points[i] = generate_stroboscopic_map(
            u=u[i],
            parameters=parameters,
            num_intersections=num_intersections,
            sampling_time=sampling_time,
            equations_of_motion=equations_of_motion,
            transient_time=transient_time,
            time_step=time_step,
            atol=atol,
            rtol=rtol,
            integrator=integrator,
        )

    return strobe_points
