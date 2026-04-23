# trajectory.py

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
from numba import njit
from joblib import Parallel, delayed

from pynamicalsys.common.types import flow_t
from pynamicalsys.continuous_time.step_methods import rk4_step_wrapped
from pynamicalsys.continuous_time.step import evolve_system, step


@njit
def generate_trajectory(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: np.float64,
    equations_of_motion: flow_t,
    transient_time: np.float64 | None = None,
    time_step: np.float64 = np.float64(0.01),
    atol: np.float64 = np.float64(1e-6),
    rtol: np.float64 = np.float64(1e-3),
    integrator=rk4_step_wrapped,
) -> NDArray[np.float64]:
    """
    Generate a trajectory of a continuous-time dynamical system.

    Parameters
    ----------
    u : NDArray[np.float64]
        Initial state vector.
    parameters : NDArray[np.float64]
        System parameters passed to the equations of motion.
    total_time : np.float64
        Final integration time.
    equations_of_motion : flow_t
        Continuous-time vector field with signature
        `(time, u, parameters) -> dudt`.
    transient_time : np.float64 | None, optional
        Initial integration time discarded before recording the trajectory.
    time_step : np.float64, optional
        Initial integration step size.
    atol : np.float64, optional
        Absolute tolerance used by adaptive integrators.
    rtol : np.float64, optional
        Relative tolerance used by adaptive integrators.
    integrator : callable, optional
        Step method with unified signature returning
        `(u_next, time_next, time_step_next, accept)`.

    Returns
    -------
    NDArray[np.float64]
        Trajectory array of shape `(n_samples, neq + 1)`, where the first column
        contains time and the remaining columns contain the state variables.

    Notes
    -----
    The trajectory starts after the transient, if one is provided. Each stored
    row has the form

    - `[time, u_0, u_1, ..., u_{neq-1}]`
    """
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

    neq = len(u)
    trajectory = []

    while time < total_time:
        if time + time_step > total_time:
            time_step = total_time - time

        u, time, time_step = step(
            time=time,
            u=u,
            parameters=parameters,
            equations_of_motion=equations_of_motion,
            time_step=time_step,
            atol=atol,
            rtol=rtol,
            integrator=integrator,
        )

        result = [time]
        for i in range(neq):
            result.append(u[i])
        trajectory.append(result)

    return np.asarray(trajectory, dtype=np.float64)


def ensemble_trajectories(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: np.float64,
    equations_of_motion: flow_t,
    transient_time: np.float64 | None = None,
    time_step: np.float64 = np.float64(0.01),
    atol: np.float64 = np.float64(1e-6),
    rtol: np.float64 = np.float64(1e-3),
    integrator=rk4_step_wrapped,
) -> NDArray[np.float64]:
    """
    Generate trajectories for an ensemble of initial conditions.

    Parameters
    ----------
    u : NDArray[np.float64]
        Array of initial conditions with shape `(num_ic, neq)`.
    parameters : NDArray[np.float64]
        System parameters passed to the equations of motion.
    total_time : np.float64
        Final integration time.
    equations_of_motion : flow_t
        Continuous-time vector field with signature
        `(time, u, parameters) -> dudt`.
    transient_time : np.float64 | None, optional
        Initial integration time discarded before recording each trajectory.
    time_step : np.float64, optional
        Initial integration step size.
    atol : np.float64, optional
        Absolute tolerance used by adaptive integrators.
    rtol : np.float64, optional
        Relative tolerance used by adaptive integrators.
    integrator : callable, optional
        Step method with unified signature returning
        `(u_next, time_next, time_step_next, accept)`.

    Returns
    -------
    NDArray[np.float64]
        Array containing one trajectory per initial condition. The output has
        shape `(num_ic, n_samples, neq + 1)` if all trajectories have the same
        number of stored points.
    """

    def run_one(
        u_i: NDArray[np.float64],
        parameters: NDArray[np.float64],
        total_time: np.float64,
        equations_of_motion: flow_t,
    ) -> NDArray[np.float64]:
        return generate_trajectory(
            u=u_i,
            parameters=parameters,
            total_time=total_time,
            equations_of_motion=equations_of_motion,
            transient_time=transient_time,
            time_step=time_step,
            atol=atol,
            rtol=rtol,
            integrator=integrator,
        )

    results = Parallel(n_jobs=-1)(
        delayed(run_one)(
            u[i],
            parameters,
            total_time,
            equations_of_motion,
        )
        for i in range(len(u))
    )

    return np.asarray(results, dtype=np.float64)
