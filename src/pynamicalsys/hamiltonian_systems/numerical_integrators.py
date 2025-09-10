# numerical_integrators.py

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

from typing import Callable, Tuple
from numpy.typing import NDArray
from numba import njit
import numpy as np

# Yoshida 4th-order symplectic integrator coefficients
ALPHA: float = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
BETA: float = -(2.0 ** (1.0 / 3.0)) / (2.0 - 2.0 ** (1.0 / 3.0))


@njit
def velocity_verlet_2nd_step(
    q: NDArray[np.float64],
    p: NDArray[np.float64],
    time_step: float,
    grad_T: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    grad_V: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    parameters: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Perform one step of the velocity Verlet integrator (second-order, symplectic).

    Parameters
    ----------
    q : NDArray[np.float64]
        Current generalized coordinates.
    p : NDArray[np.float64]
        Current generalized momenta.
    time_step : float
        Integration time step.
    grad_T : Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]
        Function returning the gradient of the kinetic energy with respect to `p`.
    grad_V : Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]
        Function returning the gradient of the potential energy with respect to `q`.
    parameters : NDArray[np.float64]
        Additional parameters passed to `grad_T` and `grad_V`.

    Returns
    -------
    q_new : NDArray[np.float64]
        Updated generalized coordinates after one step.
    p_new : NDArray[np.float64]
        Updated generalized momenta after one step.
    """
    q_new = q.copy()
    p_new = p.copy()

    # Half kick
    gradV = grad_V(q, parameters)
    p_new -= 0.5 * time_step * gradV

    # Drift
    gradT = grad_T(p_new, parameters)
    q_new += time_step * gradT

    # Half kick
    gradV = grad_V(q_new, parameters)
    p_new -= 0.5 * time_step * gradV

    return q_new, p_new


@njit
def velocity_verlet_2nd_step_tangent(
    q: NDArray[np.float64],
    p: NDArray[np.float64],
    dq: NDArray[np.float64],
    dp: NDArray[np.float64],
    time_step: float,
    hess_T: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    hess_V: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    parameters: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Perform one step of the tangent map associated with the velocity Verlet integrator.

    This evolves deviation (tangent) vectors `(dq, dp)` along the flow of the system,
    which is necessary for computing Lyapunov exponents and stability analysis.

    Parameters
    ----------
    q : NDArray[np.float64]
        Current generalized coordinates.
    p : NDArray[np.float64]
        Current generalized momenta.
    dq : NDArray[np.float64]
        Deviation in coordinates.
    dp : NDArray[np.float64]
        Deviation in momenta.
    time_step : float
        Integration time step.
    hess_T : Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]
        Function returning the Hessian of the kinetic energy with respect to `p`.
    hess_V : Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]
        Function returning the Hessian of the potential energy with respect to `q`.
    parameters : NDArray[np.float64]
        Additional parameters passed to `hess_T` and `hess_V`.

    Returns
    -------
    dq_new : NDArray[np.float64]
        Updated deviation in coordinates after one step.
    dp_new : NDArray[np.float64]
        Updated deviation in momenta after one step.
    """
    dq_new = dq.copy()
    dp_new = dp.copy()

    # Compute Hessians
    HT = hess_T(p, parameters)
    HV = hess_V(q, parameters)

    # Half kick
    HV_dot_dq = HV @ dq
    dp_new -= 0.5 * time_step * HV_dot_dq

    # Drift
    HT_dot_dp = HT @ dp_new
    dq_new += time_step * HT_dot_dp

    # Half kick
    HV_dot_dq = HV @ dq_new
    dp_new -= 0.5 * time_step * HV_dot_dq

    return dq_new, dp_new


@njit
def yoshida_4th_step(
    q: NDArray[np.float64],
    p: NDArray[np.float64],
    time_step: float,
    grad_T: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    grad_V: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    parameters: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Perform one step of the 4th-order Yoshida symplectic integrator.

    This is constructed by composing three velocity Verlet steps with
    appropriately chosen coefficients (`ALPHA`, `BETA`).

    Parameters
    ----------
    q : NDArray[np.float64]
        Current generalized coordinates.
    p : NDArray[np.float64]
        Current generalized momenta.
    time_step : float
        Integration time step.
    grad_T : Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]
        Function returning the gradient of the kinetic energy with respect to `p`.
    grad_V : Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]
        Function returning the gradient of the potential energy with respect to `q`.
    parameters : NDArray[np.float64]
        Additional parameters passed to `grad_T` and `grad_V`.

    Returns
    -------
    q_new : NDArray[np.float64]
        Updated generalized coordinates after one Yoshida step.
    p_new : NDArray[np.float64]
        Updated generalized momenta after one Yoshida step.
    """
    q_new, p_new = velocity_verlet_2nd_step(
        q, p, ALPHA * time_step, grad_T, grad_V, parameters
    )
    q_new, p_new = velocity_verlet_2nd_step(
        q_new, p_new, BETA * time_step, grad_T, grad_V, parameters
    )
    q_new, p_new = velocity_verlet_2nd_step(
        q_new, p_new, ALPHA * time_step, grad_T, grad_V, parameters
    )

    return q_new, p_new


@njit
def yoshida_4th_step_tangent(
    q: NDArray[np.float64],
    p: NDArray[np.float64],
    dq: NDArray[np.float64],
    dp: NDArray[np.float64],
    time_step: float,
    hess_T: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    hess_V: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    parameters: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Perform one step of the tangent map associated with the Yoshida 4th-order integrator.

    This evolves deviation vectors `(dq, dp)` along the flow of the Yoshida integrator,
    which is useful for stability analysis and Lyapunov exponent computation.

    Parameters
    ----------
    q : NDArray[np.float64]
        Current generalized coordinates.
    p : NDArray[np.float64]
        Current generalized momenta.
    dq : NDArray[np.float64]
        Deviation in coordinates.
    dp : NDArray[np.float64]
        Deviation in momenta.
    time_step : float
        Integration time step.
    hess_T : Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]
        Function returning the Hessian of the kinetic energy with respect to `p`.
    hess_V : Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]
        Function returning the Hessian of the potential energy with respect to `q`.
    parameters : NDArray[np.float64]
        Additional parameters passed to `hess_T` and `hess_V`.

    Returns
    -------
    dq_new : NDArray[np.float64]
        Updated deviation in coordinates after one Yoshida step.
    dp_new : NDArray[np.float64]
        Updated deviation in momenta after one Yoshida step.
    """
    dq_new, dp_new = velocity_verlet_2nd_step_tangent(
        q, p, dq, dp, ALPHA * time_step, hess_T, hess_V, parameters
    )
    dq_new, dp_new = velocity_verlet_2nd_step_tangent(
        q, p, dq_new, dp_new, BETA * time_step, hess_T, hess_V, parameters
    )
    dq_new, dp_new = velocity_verlet_2nd_step_tangent(
        q, p, dq_new, dp_new, ALPHA * time_step, hess_T, hess_V, parameters
    )

    return dq_new, dp_new
