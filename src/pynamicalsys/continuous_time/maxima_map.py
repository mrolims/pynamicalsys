# maxima_map.py

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
def generate_maxima_map(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    num_peaks: int,
    maxima_index: int,
    equations_of_motion: flow_t,
    transient_time: np.float64 | None,
    time_step: np.float64,
    atol: np.float64,
    rtol: np.float64,
    integrator,
) -> NDArray[np.float64]:
    """
    Generate a maxima map for a single initial condition.

    Parameters
    ----------
    u : NDArray[np.float64]
        Initial state vector.
    parameters : NDArray[np.float64]
        System parameters passed to the equations of motion.
    num_peaks : int
        Number of local maxima to record.
    maxima_index : int
        Index of the state variable whose maxima are recorded.
    equations_of_motion : flow_t
        Continuous-time vector field with signature
        `(time, u, parameters) -> dudt`.
    transient_time : np.float64 | None
        Initial integration time discarded before recording maxima.
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
        Array of shape `(num_peaks, neq + 1)` whose first column contains the
        time of each detected maximum and whose remaining columns contain the
        interpolated state at that maximum.

    Notes
    -----
    A local maximum is detected when the selected coordinate satisfies

    - `y_curr > y_prev`
    - `y_curr > y_next`

    The peak time is then refined by quadratic interpolation using three
    consecutive points, and the state is linearly interpolated at the refined
    peak time.
    """
    neq = len(u)
    maxima_points = np.zeros((num_peaks, neq + 1), dtype=np.float64)

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

    u_curr, time_curr, time_step_curr = step(
        time=time_prev,
        u=u_prev,
        parameters=parameters,
        equations_of_motion=equations_of_motion,
        time_step=time_step_prev,
        atol=atol,
        rtol=rtol,
        integrator=integrator,
    )

    count = 0
    while count < num_peaks:
        u_next, time_next, time_step_next = step(
            time=time_curr,
            u=u_curr,
            parameters=parameters,
            equations_of_motion=equations_of_motion,
            time_step=time_step_curr,
            atol=atol,
            rtol=rtol,
            integrator=integrator,
        )

        y_prev = u_prev[maxima_index]
        y_curr = u_curr[maxima_index]
        y_next = u_next[maxima_index]

        if (y_curr > y_prev) and (y_curr > y_next):
            t1, t2, t3 = time_prev, time_curr, time_next
            y1, y2, y3 = y_prev, y_curr, y_next

            denom = (t1 - t2) * (t1 - t3) * (t2 - t3)
            A = (t3 * (y2 - y1) + t2 * (y1 - y3) + t1 * (y3 - y2)) / denom
            B = (t3**2 * (y1 - y2) + t2**2 * (y3 - y1) + t1**2 * (y2 - y3)) / denom

            t_peak = -B / (2.0 * A)

            lam = (t_peak - time_curr) / (time_next - time_curr)
            u_peak = (1.0 - lam) * u_curr + lam * u_next

            maxima_points[count, 0] = t_peak
            maxima_points[count, 1:] = u_peak
            count += 1

        u_prev = u_curr
        time_prev = time_curr
        time_step_prev = time_step_curr

        u_curr = u_next
        time_curr = time_next
        time_step_curr = time_step_next

    return maxima_points


@njit(parallel=True)
def ensemble_maxima_map(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    num_peaks: int,
    maxima_index: int,
    equations_of_motion: flow_t,
    transient_time: np.float64 | None,
    time_step: np.float64,
    atol: np.float64,
    rtol: np.float64,
    integrator,
) -> NDArray[np.float64]:
    """
    Generate maxima maps for an ensemble of initial conditions.

    Parameters
    ----------
    u : NDArray[np.float64]
        Array of initial conditions with shape `(num_ic, neq)`.
    parameters : NDArray[np.float64]
        System parameters passed to the equations of motion.
    num_peaks : int
        Number of local maxima to record for each trajectory.
    maxima_index : int
        Index of the state variable whose maxima are recorded.
    equations_of_motion : flow_t
        Continuous-time vector field with signature
        `(time, u, parameters) -> dudt`.
    transient_time : np.float64 | None
        Initial integration time discarded before recording maxima.
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
        Array of shape `(num_ic, num_peaks, neq + 1)` containing the maxima map
        of each initial condition.
    """
    num_ic, neq = u.shape
    maxima_points = np.zeros((num_ic, num_peaks, neq + 1), dtype=np.float64)

    for i in prange(num_ic):
        maxima_points[i] = generate_maxima_map(
            u=u[i],
            parameters=parameters,
            num_peaks=num_peaks,
            maxima_index=maxima_index,
            equations_of_motion=equations_of_motion,
            transient_time=transient_time,
            time_step=time_step,
            atol=atol,
            rtol=rtol,
            integrator=integrator,
        )

    return maxima_points
