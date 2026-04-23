# step_methods.py

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
from pynamicalsys.continuous_time.fixed_step import rk4_step, variational_rk4_step
from pynamicalsys.continuous_time.adaptive_step import rk45_step, variational_rk45_step


@njit
def rk4_step_wrapped(
    t: np.float64,
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    equations_of_motion: flow_t,
    jacobian: flow_jacobian_t | None = None,
    time_step: np.float64 = np.float64(0.01),
    number_of_deviation_vectors: int | None = None,
    atol: np.float64 = np.float64(1e-6),
    rtol: np.float64 = np.float64(1e-3),
) -> tuple[NDArray[np.float64], np.float64, np.float64, bool]:
    """
    Advance one integration step using the fixed-step RK4 method and return the
    result in the same format as the adaptive step methods.

    Parameters
    ----------
    t : np.float64
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
        Jacobian of the vector field. If provided, the variational RK4 step is
        used and `u` is interpreted as an extended state.
    time_step : np.float64, optional
        Integration step size.
    number_of_deviation_vectors : int | None, optional
        Number of deviation vectors in the extended state when the variational
        step is used.
    atol : np.float64, optional
        Unused. Included only to match the signature of adaptive step methods.
    rtol : np.float64, optional
        Unused. Included only to match the signature of adaptive step methods.

    Returns
    -------
    tuple[NDArray[np.float64], np.float64, np.float64, bool]
        Tuple containing:
        - the updated state
        - the advanced time
        - the next step size, equal to `time_step`
        - `True`, since fixed-step RK4 never rejects a step
    """
    if jacobian is None:
        u_next = rk4_step(t, u, parameters, equations_of_motion, time_step)
    else:
        u_next = variational_rk4_step(
            t,
            u,
            parameters,
            equations_of_motion,
            jacobian,
            time_step=time_step,
            number_of_deviation_vectors=number_of_deviation_vectors,
        )

    t_next = np.float64(t + time_step)
    h_next = np.float64(time_step)
    accept = True

    return u_next, t_next, h_next, accept


@njit
def rk45_step_wrapped(
    t: np.float64,
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    equations_of_motion: flow_t,
    jacobian: flow_jacobian_t | None = None,
    time_step: np.float64 = np.float64(0.01),
    number_of_deviation_vectors: int | None = None,
    atol: np.float64 = np.float64(1e-6),
    rtol: np.float64 = np.float64(1e-3),
) -> tuple[NDArray[np.float64], np.float64, np.float64, bool]:
    """
    Advance one integration step using the adaptive RK45 method and return the
    result in a unified step-method format.

    Parameters
    ----------
    t : np.float64
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
        Jacobian of the vector field. If provided, the variational RK45 step is
        used and `u` is interpreted as an extended state.
    time_step : np.float64, optional
        Current integration step size.
    number_of_deviation_vectors : int | None, optional
        Number of deviation vectors in the extended state when the variational
        step is used.
    atol : np.float64, optional
        Absolute tolerance used in the adaptive error control.
    rtol : np.float64, optional
        Relative tolerance used in the adaptive error control.

    Returns
    -------
    tuple[NDArray[np.float64], np.float64, np.float64, bool]
        Tuple containing:
        - the proposed updated state
        - the advanced time
        - the suggested next step size
        - whether the step was accepted
    """
    if jacobian is None:
        return rk45_step(
            t,
            u,
            parameters,
            equations_of_motion,
            time_step,
            atol=atol,
            rtol=rtol,
        )

    return variational_rk45_step(
        t,
        u,
        parameters,
        equations_of_motion,
        jacobian,
        time_step,
        number_of_deviation_vectors=number_of_deviation_vectors,
        atol=atol,
        rtol=rtol,
    )


@njit
def estimate_initial_step(
    t0: np.float64,
    u0: NDArray[np.float64],
    parameters: NDArray[np.float64],
    equations_of_motion: flow_t,
    order: int = 5,
    atol: np.float64 = np.float64(1e-6),
    rtol: np.float64 = np.float64(1e-3),
) -> np.float64:
    """
    Estimate an initial step size for adaptive time integration.

    Parameters
    ----------
    t0 : np.float64
        Initial integration time.
    u0 : NDArray[np.float64]
        Initial state vector.
    parameters : NDArray[np.float64]
        System parameters passed to the equations of motion.
    equations_of_motion : flow_t
        Continuous-time vector field with signature
        `(time, u, parameters) -> dudt`.
    order : int, optional
        Order of the adaptive method used for the estimate. The Dormand-Prince
        method corresponds to `order=5`.
    atol : np.float64, optional
        Absolute tolerance used in the step-size estimate.
    rtol : np.float64, optional
        Relative tolerance used in the step-size estimate.

    Returns
    -------
    np.float64
        Estimated initial integration step size.

    Notes
    -----
    This function follows the usual heuristic based on the norm of the initial
    state, the norm of the initial derivative, and a one-step estimate of the
    second derivative. It is intended for adaptive Runge-Kutta methods.
    """
    f0 = equations_of_motion(t0, u0, parameters)

    scale = atol + rtol * np.abs(u0)
    d0 = np.linalg.norm(u0 / scale)
    d1 = np.linalg.norm(f0 / scale)

    if d0 < 1e-5 or d1 < 1e-5:
        h0 = 1e-6
    else:
        h0 = 0.01 * d0 / d1

    u1 = u0 + h0 * f0
    f1 = equations_of_motion(t0 + h0, u1, parameters)
    d2 = np.linalg.norm((f1 - f0) / scale) / h0

    if d1 <= 1e-15 and d2 <= 1e-15:
        h1 = max(1e-6, h0 * 1e-3)
    else:
        h1 = (0.01 / max(d1, d2)) ** (1.0 / (order + 1))

    return np.float64(min(100 * h0, h1))
