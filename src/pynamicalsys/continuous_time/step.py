# step.py

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
from numba import njit

from pynamicalsys.common.types import flow_t, flow_jacobian_t
from pynamicalsys.continuous_time.step_methods import rk4_step_wrapped

# Smallest step size, relative to the current time, that an adaptive
# integrator may reduce to after rejecting a step.
_MIN_RELATIVE_STEP = 1e-14


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

    Raises
    ------
    RuntimeError
        If an adaptive integrator rejects a step and the reduced step size is
        smaller than `1e-14 * max(|time|, 1)` or is not a number. This happens
        when the solution diverges or becomes non-finite, and prevents the step
        from being retried indefinitely.
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
        elif not time_step_new >= _MIN_RELATIVE_STEP * max(abs(time), 1.0):
            # Written so that a NaN step size also fails the comparison.
            raise RuntimeError(
                "The adaptive integrator could not take a step: the step size "
                "fell below the minimum allowed or became NaN. The solution may "
                "diverge or contain non-finite values, or the tolerances may be "
                "too strict."
            )

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
        if time + time_step > total_time:
            time_step = total_time - time

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

    return u


@njit
def step_to(
    time: np.float64,
    u: NDArray[np.float64],
    target_time: np.float64,
    parameters: NDArray[np.float64],
    equations_of_motion: flow_t,
    jacobian: flow_jacobian_t | None = None,
    time_step: np.float64 = np.float64(0.01),
    atol: np.float64 = np.float64(1e-6),
    rtol: np.float64 = np.float64(1e-3),
    integrator=rk4_step_wrapped,
    number_of_deviation_vectors: int | None = None,
) -> tuple[NDArray[np.float64], np.float64, np.float64, bool]:
    """
    Advance the system by one accepted step without passing `target_time`.

    If the current step would reach or pass `target_time`, it is shortened so
    that the step ends exactly at `target_time`. After landing there, the step
    size is restored to the value it had before it was shortened, unless the
    integrator suggests a larger one, so that a target does not reduce the
    step size of the following steps.

    Parameters
    ----------
    time : np.float64
        Current integration time.
    u : NDArray[np.float64]
        Current state vector, or extended state vector when `jacobian` is
        provided.
    target_time : np.float64
        Time that the step must not pass. Must be greater than `time`.
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
    tuple[NDArray[np.float64], np.float64, np.float64, bool]
        Tuple containing:
        - the accepted updated state
        - the accepted updated time, equal to `target_time` when it was reached
        - the suggested next step size
        - whether `target_time` was reached

    Notes
    -----
    With an adaptive integrator, the shortened step may itself be rejected
    and replaced by a smaller one. In that case `target_time` is not reached
    and the returned flag is False.
    """
    shortened = False
    requested_time_step = time_step
    if time + time_step >= target_time:
        time_step = target_time - time
        shortened = True

    u_new, time_new, time_step_new = step(
        time,
        u,
        parameters,
        equations_of_motion,
        jacobian=jacobian,
        time_step=time_step,
        atol=atol,
        rtol=rtol,
        integrator=integrator,
        number_of_deviation_vectors=number_of_deviation_vectors,
    )

    reached = False
    if shortened and target_time - time_new <= 1e-12 * max(1.0, abs(target_time)):
        time_new = target_time
        reached = True
        if requested_time_step > time_step_new:
            time_step_new = requested_time_step

    return u_new, time_new, time_step_new, reached
