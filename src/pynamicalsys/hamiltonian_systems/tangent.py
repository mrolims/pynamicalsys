# tangent.py

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

from pynamicalsys.common.types import grad_t, hess_t, symplectic_tangent_step_t
from pynamicalsys.hamiltonian_systems.coefficients import ALPHA, BETA


@njit
def velocity_verlet_2nd_step_traj_tan(
    q: NDArray[np.float64],
    p: NDArray[np.float64],
    dv: NDArray[np.float64],
    time_step: np.float64,
    grad_T: grad_t,
    grad_V: grad_t,
    hess_T: hess_t,
    hess_V: hess_t,
    parameters: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Perform one step of the trajectory and tangent map associated with the
    second-order velocity Verlet integrator.

    This evolves both the phase-space trajectory `(q, p)` and the deviation
    vectors `dv`.

    Parameters
    ----------
    q : NDArray[np.float64]
        Current generalized coordinates.
    p : NDArray[np.float64]
        Current generalized momenta.
    dv : NDArray[np.float64]
        Deviation vectors of shape `(2 * dof, n_dev)`, where the first `dof`
        rows correspond to coordinate deviations and the last `dof` rows
        correspond to momentum deviations.
    time_step : np.float64
        Integration time step.
    grad_T : grad_t
        Gradient of the kinetic energy with respect to the momenta.
    grad_V : grad_t
        Gradient of the potential energy with respect to the coordinates.
    hess_T : hess_t
        Hessian of the kinetic energy with respect to the momenta.
    hess_V : hess_t
        Hessian of the potential energy with respect to the coordinates.
    parameters : NDArray[np.float64]
        Additional parameters passed to the gradient and Hessian functions.

    Returns
    -------
    tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
        Updated coordinates, momenta, and deviation vectors after one step.
    """
    q_new = q.copy()
    p_new = p.copy()
    dv_new = dv.copy()
    dof = len(q)

    gradV = grad_V(q_new, parameters)
    p_new -= np.float64(0.5) * time_step * gradV

    HV = hess_V(q_new, parameters)
    dv_new[dof:, :] -= (
        np.float64(0.5) * time_step * (HV @ np.ascontiguousarray(dv_new[:dof, :]))
    )

    gradT = grad_T(p_new, parameters)
    q_new += time_step * gradT

    HT = hess_T(p_new, parameters)
    dv_new[:dof, :] += time_step * (HT @ np.ascontiguousarray(dv_new[dof:, :]))

    gradV = grad_V(q_new, parameters)
    p_new -= np.float64(0.5) * time_step * gradV

    HV = hess_V(q_new, parameters)
    dv_new[dof:, :] -= (
        np.float64(0.5) * time_step * (HV @ np.ascontiguousarray(dv_new[:dof, :]))
    )

    return q_new, p_new, dv_new


@njit
def yoshida_4th_step_traj_tan(
    q: NDArray[np.float64],
    p: NDArray[np.float64],
    dv: NDArray[np.float64],
    time_step: np.float64,
    grad_T: grad_t,
    grad_V: grad_t,
    hess_T: hess_t,
    hess_V: hess_t,
    parameters: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Perform one step of the fourth-order Yoshida trajectory and tangent-map
    integrator.

    This integrator is obtained by composing three second-order velocity Verlet
    trajectory+tangent steps with coefficients `ALPHA` and `BETA`.

    Parameters
    ----------
    q : NDArray[np.float64]
        Current generalized coordinates.
    p : NDArray[np.float64]
        Current generalized momenta.
    dv : NDArray[np.float64]
        Deviation vectors of shape `(2 * dof, n_dev)`.
    time_step : np.float64
        Integration time step.
    grad_T : grad_t
        Gradient of the kinetic energy with respect to the momenta.
    grad_V : grad_t
        Gradient of the potential energy with respect to the coordinates.
    hess_T : hess_t
        Hessian of the kinetic energy with respect to the momenta.
    hess_V : hess_t
        Hessian of the potential energy with respect to the coordinates.
    parameters : NDArray[np.float64]
        Additional parameters passed to the gradient and Hessian functions.

    Returns
    -------
    tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
        Updated coordinates, momenta, and deviation vectors after one Yoshida step.
    """
    q_new, p_new, dv_new = velocity_verlet_2nd_step_traj_tan(
        q,
        p,
        dv,
        np.float64(ALPHA) * time_step,
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
        np.float64(BETA) * time_step,
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
        np.float64(ALPHA) * time_step,
        grad_T,
        grad_V,
        hess_T,
        hess_V,
        parameters,
    )

    return q_new, p_new, dv_new


@njit
def advance_block(
    q: NDArray[np.float64],
    p: NDArray[np.float64],
    Q: NDArray[np.float64],
    qr_steps: int,
    time_step: np.float64,
    grad_T: grad_t,
    grad_V: grad_t,
    hess_T: hess_t,
    hess_V: hess_t,
    parameters: NDArray[np.float64],
    integrator_traj_tan: symplectic_tangent_step_t,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Advance the trajectory and tangent basis for a fixed number of steps.

    Parameters
    ----------
    q : NDArray[np.float64]
        Current generalized coordinates.
    p : NDArray[np.float64]
        Current generalized momenta.
    Q : NDArray[np.float64]
        Current deviation-vector matrix.
    qr_steps : int
        Number of integration steps to perform.
    time_step : np.float64
        Integration time step.
    grad_T : grad_t
        Gradient of the kinetic energy with respect to the momenta.
    grad_V : grad_t
        Gradient of the potential energy with respect to the coordinates.
    hess_T : hess_t
        Hessian of the kinetic energy with respect to the momenta.
    hess_V : hess_t
        Hessian of the potential energy with respect to the coordinates.
    parameters : NDArray[np.float64]
        Additional parameters passed to the gradient and Hessian functions.
    integrator_traj_tan : callable
        Trajectory+tangent integrator step.

    Returns
    -------
    tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
        Updated coordinates, momenta, and deviation-vector matrix after
        `qr_steps` steps.
    """
    for _ in range(qr_steps):
        q, p, Q = integrator_traj_tan(
            q,
            p,
            Q,
            time_step,
            grad_T,
            grad_V,
            hess_T,
            hess_V,
            parameters,
        )

    return q, p, Q
