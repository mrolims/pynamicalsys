# fixed_step.py

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
from pynamicalsys.continuous_time.variational import variational_equations


@njit
def rk4_step(
    t: np.float64,
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    equations_of_motion: flow_t,
    time_step: np.float64 = np.float64(0.01),
) -> NDArray[np.float64]:
    """
    Advance one fixed-step fourth-order Runge-Kutta integration step.

    Parameters
    ----------
    t : np.float64
        Current integration time.
    u : NDArray[np.float64]
        Current state vector.
    parameters : NDArray[np.float64]
        System parameters passed to the equations of motion.
    equations_of_motion : flow_t
        Continuous-time vector field with signature
        `(time, u, parameters) -> dudt`.
    time_step : np.float64, optional
        Fixed integration step size.

    Returns
    -------
    NDArray[np.float64]
        State vector at time `t + time_step`.

    Notes
    -----
    This function applies the classical explicit RK4 scheme,

    - `k1 = f(t, u)`
    - `k2 = f(t + h/2, u + h k1 / 2)`
    - `k3 = f(t + h/2, u + h k2 / 2)`
    - `k4 = f(t + h, u + h k3)`

    and returns

    - `u_{n+1} = u_n + (h/6)(k1 + 2k2 + 2k3 + k4)`

    where `h = time_step`.
    """
    k1 = equations_of_motion(t, u, parameters)
    k2 = equations_of_motion(t + 0.5 * time_step, u + 0.5 * time_step * k1, parameters)
    k3 = equations_of_motion(t + 0.5 * time_step, u + 0.5 * time_step * k2, parameters)
    k4 = equations_of_motion(t + time_step, u + time_step * k3, parameters)

    u_next = u + (time_step / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return u_next


@njit
def variational_rk4_step(
    t: np.float64,
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    equations_of_motion: flow_t,
    jacobian: flow_jacobian_t,
    time_step: np.float64 = np.float64(0.01),
    number_of_deviation_vectors: int | None = None,
) -> NDArray[np.float64]:
    """
    Advance one fixed-step fourth-order Runge-Kutta integration step
    for the extended variational system.

    Parameters
    ----------
    t : np.float64
        Current integration time.
    u : NDArray[np.float64]
        Current extended state vector containing:
        - the physical state in the first `neq` entries
        - the flattened deviation vectors in the remaining entries
    parameters : NDArray[np.float64]
        System parameters passed to the equations of motion and Jacobian.
    equations_of_motion : flow_t
        Continuous-time vector field with signature
        `(time, u, parameters) -> dudt`.
    jacobian : flow_jacobian_t
        Jacobian of the vector field with signature
        `(time, u, parameters) -> J`.
    time_step : np.float64, optional
        Fixed integration step size.
    number_of_deviation_vectors : int | None, optional
        Number of deviation vectors stored in the extended state.
        If None, this number is inferred inside `variational_equations`.

    Returns
    -------
    NDArray[np.float64]
        Extended state vector at time `t + time_step`.

    Notes
    -----
    This function applies the classical explicit RK4 scheme to the combined
    system formed by the equations of motion and the variational equations.
    The derivative evaluations are computed through `variational_equations`,
    so the returned state advances both:
    - the phase-space trajectory
    - the tangent-space dynamics of the deviation vectors
    """
    k1 = variational_equations(
        t,
        u,
        parameters,
        equations_of_motion,
        jacobian,
        number_of_deviation_vectors=number_of_deviation_vectors,
    )

    k2 = variational_equations(
        t + 0.5 * time_step,
        u + 0.5 * time_step * k1,
        parameters,
        equations_of_motion,
        jacobian,
        number_of_deviation_vectors=number_of_deviation_vectors,
    )
    k3 = variational_equations(
        t + 0.5 * time_step,
        u + 0.5 * time_step * k2,
        parameters,
        equations_of_motion,
        jacobian,
        number_of_deviation_vectors=number_of_deviation_vectors,
    )
    k4 = variational_equations(
        t + time_step,
        u + time_step * k3,
        parameters,
        equations_of_motion,
        jacobian,
        number_of_deviation_vectors=number_of_deviation_vectors,
    )

    u_next = u + (time_step / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return u_next
