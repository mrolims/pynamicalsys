# step.py

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

from pynamicalsys.common.types import flow_t, flow_jacobian_t
from pynamicalsys.continuous_time.step_methods import rk4_step_wrapped


@njit
def step(
    time: np.float64,
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    equations_of_motion: flow_t,
    jacobian: flow_jacobian_t | None = None,
    time_step: np.float64 = np.float64(0.01),
    atol: np.float64 = np.float64(1e-6),
    rtol: np.float64 = np.float64(1e-3),
    integrator=rk4_step_wrapped,
    number_of_deviation_vectors: int | None = None,
) -> tuple[NDArray[np.float64], np.float64, np.float64]:
    """
    Advance the system by one accepted integration step.

    This function repeatedly calls the selected step method until an accepted
    step is obtained. For fixed-step methods, acceptance is immediate. For
    adaptive methods, rejected steps are retried using the updated step size.

    Parameters
    ----------
    time : np.float64
        Current integration time.
    u : NDArray[np.float64]
        Current state vector, or extended state vector when `jacobian` is
        provided.
    parameters : NDArray[np.float64]
        System parameters passed to the equations of motion.
    equations_of_motion : flow_t
        Continuous-time vector field with signature
        `(time, u, parameters) -> dudt`.
    jacobian : flow_jacobian_t | None, optional
        Jacobian of the vector field. If provided, the integrator evolves the
        extended variational system.
    time_step : np.float64, optional
        Current integration step size.
    atol : np.float64, optional
        Absolute tolerance used by adaptive integrators.
    rtol : np.float64, optional
        Relative tolerance used by adaptive integrators.
    integrator : callable, optional
        Step method with unified signature returning
        `(u_next, time_next, time_step_next, accept)`.
    number_of_deviation_vectors : int | None, optional
        Number of deviation vectors in the extended state when variational
        integration is used.

    Returns
    -------
    tuple[NDArray[np.float64], np.float64, np.float64]
        Tuple containing:
        - the accepted updated state
        - the accepted updated time
        - the suggested next step size
    """
    u_new = u.copy()
    time_new = time
    time_step_new = time_step
    accept = False

    while not accept:
        u_new, time_new, time_step_new, accept = integrator(
            time,
            u,
            parameters,
            equations_of_motion,
            jacobian=jacobian,
            time_step=time_step,
            atol=atol,
            rtol=rtol,
            number_of_deviation_vectors=number_of_deviation_vectors,
        )

        if accept:
            time = time_new
            u = u_new.copy()

        time_step = time_step_new

    return u_new, time_new, time_step_new


@njit
def evolve_system(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: np.float64,
    equations_of_motion: flow_t,
    time_step: np.float64 = np.float64(0.01),
    atol: np.float64 = np.float64(1e-6),
    rtol: np.float64 = np.float64(1e-3),
    integrator=rk4_step_wrapped,
) -> NDArray[np.float64]:
    """
    Evolve the system from time `0` up to `total_time`.

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
        State vector at the final integration time.

    Notes
    -----
    If the current step would overshoot `total_time`, the step size is reduced
    so that the final accepted step lands exactly at the requested final time.
    """
    u = u.copy()
    time = np.float64(0.0)

    while time < total_time:
        u, time, time_step = step(
            time,
            u,
            parameters,
            equations_of_motion,
            time_step=time_step,
            atol=atol,
            rtol=rtol,
            integrator=integrator,
        )

        if time + time_step > total_time:
            time_step = total_time - time

    return u
