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

import numpy as np
from numba import njit
from numpy.typing import NDArray

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
def velocity_verlet_2nd_step_traj_tan(
    q: NDArray[np.float64],
    p: NDArray[np.float64],
    dv: NDArray[np.float64],
    time_step: float,
    grad_T: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    grad_V: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    hess_T: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    hess_V: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    parameters: NDArray[np.float64],
):
    """
    Perform one step of the trajectory and tangent map associated with the velocity Verlet integrator.

    This evolves the trajectory `(q, p)` and deviation (tangent) vectors `dv` along the flow of the system,
    which is necessary for computing Lyapunov exponents and stability analysis.

    Parameters
    ----------
    q : NDArray[np.float64]
        Current generalized coordinates.
    p : NDArray[np.float64]
        Current generalized momenta.
    dv : NDArray[np.float64]
        Deviation vectors with shape (dim, n_dev).
    time_step : float
        Integration time step.
    grad_T : Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]
        Function returning the gradient of the kinetic energy with respect to `p`.
    grad_V : Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]
        Function returning the grandienf of the potential energy with respect to `q`.
    hess_T : Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]
        Function returning the Hessian of the kinetic energy with respect to `p`.
    hess_V : Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]
        Function returning the Hessian of the potential energy with respect to `q`.
    parameters : NDArray[np.float64]
        Additional parameters passed to `grad_T`, `grad_V`, hess_T`, and `hess_V`.

    Returns
    -------
    q_new : NDArray[np.float64]
        Updated coordinates
    p_new : NDArray[np.float64]
        Updated momenta
    dv_new : NDArray[np.float64]
        Updated deviation vectors.
    """

    q_new = q.copy()
    p_new = p.copy()
    dv_new = dv.copy()
    dof = len(q)

    # --- Half kick --- #
    # on the main trajectory
    gradV = grad_V(q_new, parameters)
    p_new -= 0.5 * time_step * gradV

    # on tangent momenta
    HV = hess_V(q_new, parameters)
    HV_dot_dq = HV @ dv_new[:dof, :]  # HV cdot dq
    # Update dp
    dv_new[dof:, :] -= 0.5 * time_step * HV_dot_dq

    # --- Drift --- #
    # on the main trajectory
    gradT = grad_T(p_new, parameters)
    q_new += time_step * gradT

    # on the tangent coordinates
    HT = hess_T(p_new, parameters)
    HT_dot_dp = HT @ dv_new[dof:, :]  # HT cdot dp
    # Update dq
    dv_new[:dof, :] += time_step * HT_dot_dp

    # --- Half kick --- #
    # on the main trajectory
    gradV = grad_V(q_new, parameters)
    p_new -= 0.5 * time_step * gradV

    # on tangent momenta
    HV = hess_V(q_new, parameters)
    HV_dot_dq = HV @ dv_new[:dof, :]  # HV cdot dq
    # Update dp
    dv_new[dof:, :] -= 0.5 * time_step * HV_dot_dq

    return q_new, p_new, dv_new


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
def yoshida_4th_step_traj_tan(
    q: NDArray[np.float64],
    p: NDArray[np.float64],
    dv: NDArray[np.float64],
    time_step: float,
    grad_T: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    grad_V: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    hess_T: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    hess_V: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    parameters: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Perform one step of the trajectory and tangent map associated with the Yoshida 4th-order integrator.

    This evolves the trajectory `(q, p)` and deviation (tangent) vectors `dv` along the flow of the system,
    which is necessary for computing Lyapunov exponents and stability analysis.

    Parameters
    ----------
    q : NDArray[np.float64]
        Current generalized coordinates.
    p : NDArray[np.float64]
        Current generalized momenta.
    dv : NDArray[np.float64]
        Deviation vectors with shape (dim, n_dev).
    time_step : float
        Integration time step.
    grad_T : Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]
        Function returning the gradient of the kinetic energy with respect to `p`.
    grad_V : Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]
        Function returning the grandienf of the potential energy with respect to `q`.
    hess_T : Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]
        Function returning the Hessian of the kinetic energy with respect to `p`.
    hess_V : Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]
        Function returning the Hessian of the potential energy with respect to `q`.
    parameters : NDArray[np.float64]
        Additional parameters passed to `grad_T`, `grad_V`, hess_T`, and `hess_V`.

    Returns
    -------
    q_new : NDArray[np.float64]
        Updated coordinates
    p_new : NDArray[np.float64]
        Updated momenta
    dv_new : NDArray[np.float64]
        Updated deviation vectors.
    """

    q_new, p_new, dv_new = velocity_verlet_2nd_step_traj_tan(
        q, p, dv, ALPHA * time_step, grad_T, grad_V, hess_T, hess_V, parameters
    )

    q_new, p_new, dv_new = velocity_verlet_2nd_step_traj_tan(
        q_new,
        p_new,
        dv_new,
        BETA * time_step,
        grad_T,
        grad_V,
        hess_T,
        hess_V,
        parameters,
    )

    q_new, p_new, dv_new = velocity_verlet_2nd_step_traj_tan(
        q_new,
        p_new,
        dv_new,
        ALPHA * time_step,
        grad_T,
        grad_V,
        hess_T,
        hess_V,
        parameters,
    )

    return q_new, p_new, dv_new
