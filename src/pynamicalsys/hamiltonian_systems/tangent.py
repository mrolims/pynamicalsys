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

from pynamicalsys.common.types import (
    symplectic_tangent_step_t,
    system_func_t,
)
from pynamicalsys.hamiltonian_systems.coefficients import ALPHA, BETA
from pynamicalsys.hamiltonian_systems.fixed_step import _symplectic_J_inv_matmul


@njit
def velocity_verlet_2nd_step_traj_tan(
    q: NDArray[np.float64],
    p: NDArray[np.float64],
    dv: NDArray[np.float64],
    time_step: np.float64,
    grad_T: system_func_t,
    grad_V: system_func_t,
    hess_T: system_func_t,
    hess_V: system_func_t,
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
    grad_T : system_func_t
        Gradient of the kinetic energy with respect to the momenta.
    grad_V : system_func_t
        Gradient of the potential energy with respect to the coordinates.
    hess_T : system_func_t
        Hessian of the kinetic energy with respect to the momenta.
    hess_V : system_func_t
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
    grad_T: system_func_t,
    grad_V: system_func_t,
    hess_T: system_func_t,
    hess_V: system_func_t,
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
    grad_T : system_func_t
        Gradient of the kinetic energy with respect to the momenta.
    grad_V : system_func_t
        Gradient of the potential energy with respect to the coordinates.
    hess_T : system_func_t
        Hessian of the kinetic energy with respect to the momenta.
    hess_V : system_func_t
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
def implicit_midpoint_step_traj_tan(
    q: NDArray[np.float64],
    p: NDArray[np.float64],
    dv: NDArray[np.float64],
    time_step: np.float64,
    eom: system_func_t,
    hess_H: system_func_t,
    parameters: NDArray[np.float64],
    tol: np.float64 = np.float64(1e-12),
    max_iter: int = 50,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Perform one step of the implicit midpoint integrator, advancing both
    the trajectory (q, p) and a set of tangent (deviation) vectors dv.

    The trajectory update is identical to implicit_midpoint_step: Newton's
    method solves for the midpoint z_bar = (q_bar, p_bar) satisfying
        z_bar = z_n + (h/2) * J^{-1} grad_H(z_bar)
    then sets z_{n+1} = 2 * z_bar - z_n.

    The tangent update advances dv by the monodromy matrix M, where M
    satisfies (derived by differentiating the discrete map above w.r.t.
    z_n, with bar{z} = (z_n + z_{n+1}) / 2):
        DF(z_bar) @ M = I + (h/2) * J^{-1} grad^2 H(z_bar)
    with DF(z_bar) = I - (h/2) * J^{-1} grad^2 H(z_bar) the same Jacobian
    used in the last Newton iteration for the trajectory update. Since
    dv has shape (2n, n_dev) with n_dev typically << 2n, dv_new = M @ dv
    is computed by solving
        DF(z_bar) @ dv_new = (I + (h/2) * J^{-1} grad^2 H(z_bar)) @ dv
    directly, without ever forming the (2n, 2n) matrix M.

    Works for arbitrary (including non-separable) Hamiltonians H(q, p).

    If the Newton iteration fails to converge within `max_iter` steps,
    a warning is printed and the best available (unconverged) result
    is returned for both the trajectory and the tangent vectors.

    Parameters
    ----------
    q : NDArray[np.float64]
        Current generalized coordinates.
    p : NDArray[np.float64]
        Current generalized momenta.
    dv : NDArray[np.float64]
        Deviation vectors of shape `(2 * dof, n_dev)`, where the first
        `dof` rows correspond to coordinate deviations and the last
        `dof` rows correspond to momentum deviations.
    time_step : np.float64
        Integration time step.
    eom : system_func_t
        Equations of motion: eom(q, p, parameters) -> (dH/dp, -dH/dq).
    hess_H : system_func_t
        Full Hessian of H w.r.t. z=(q,p), shape (2n, 2n).
    parameters : NDArray[np.float64]
        Additional parameters passed to `eom` and `hess_H`.
    tol : np.float64
        Newton convergence tolerance on the residual norm.
    max_iter : int
        Maximum Newton iterations per step.

    Returns
    -------
    tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
        Updated coordinates, momenta, and deviation vectors after one step.
    """
    n = q.shape[0]
    h = time_step

    z_n = np.concatenate((q, p))

    q_bar = q.copy()
    p_bar = p.copy()

    identity = np.eye(2 * n)

    converged = False
    residual_norm = np.float64(0.0)
    Jinv_Hmid = np.zeros((2 * n, 2 * n))
    DF = identity.copy()

    for _ in range(max_iter):
        qdot, pdot = eom(q_bar, p_bar, parameters)
        f_mid = np.concatenate((qdot, pdot))

        z_bar = np.concatenate((q_bar, p_bar))
        F = z_bar - z_n - np.float64(0.5) * h * f_mid

        residual_norm = np.sqrt(np.sum(F * F))

        H_mid = hess_H(q_bar, p_bar, parameters)
        Jinv_Hmid = _symplectic_J_inv_matmul(n, H_mid)
        DF = identity - np.float64(0.5) * h * Jinv_Hmid

        if residual_norm < tol:
            converged = True
            break

        delta = np.linalg.solve(DF, -F)
        z_bar = z_bar + delta
        q_bar = z_bar[:n]
        p_bar = z_bar[n:]

    if not converged:
        print(
            "implicit_midpoint_step_traj_tan: Newton iteration did not "
            "converge. residual norm =",
            residual_norm,
            " tol =",
            tol,
            " max_iter =",
            max_iter,
            ". Returning unconverged result; consider reducing time_step "
            "or increasing max_iter.",
        )

    rhs = (identity + np.float64(0.5) * h * Jinv_Hmid) @ dv
    dv_new = np.linalg.solve(DF, rhs).astype(np.float64)

    q_new = np.float64(2.0) * q_bar - q
    p_new = np.float64(2.0) * p_bar - p

    return q_new, p_new, dv_new


@njit
def advance_block_sep(
    q: NDArray[np.float64],
    p: NDArray[np.float64],
    Q: NDArray[np.float64],
    qr_steps: int,
    time_step: np.float64,
    grad_T: system_func_t,
    grad_V: system_func_t,
    hess_T: system_func_t,
    hess_V: system_func_t,
    parameters: NDArray[np.float64],
    integrator_traj_tan: symplectic_tangent_step_t,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Advance the trajectory and tangent basis for a fixed number of steps of a
    separable Hamiltonian system, H(q, p) = T(p) + V(q), integrated with an
    explicit symplectic stepper (velocity Verlet or fourth-order Yoshida).

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
    grad_T : system_func_t
        Gradient of the kinetic energy with respect to the momenta.
    grad_V : system_func_t
        Gradient of the potential energy with respect to the coordinates.
    hess_T : system_func_t
        Hessian of the kinetic energy with respect to the momenta.
    hess_V : system_func_t
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


@njit
def advance_block_imp(
    q: NDArray[np.float64],
    p: NDArray[np.float64],
    Q: NDArray[np.float64],
    qr_steps: int,
    time_step: np.float64,
    eom: system_func_t,
    hess_H: system_func_t,
    parameters: NDArray[np.float64],
    tol: np.float64,
    max_iter: int,
    integrator_traj_tan: symplectic_tangent_step_t,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Advance the trajectory and tangent basis for a fixed number of steps of a
    general (possibly non-separable) Hamiltonian system H(q, p), integrated
    with the implicit midpoint method.

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
    eom : system_func_t
        Equations of motion of the system.
    hess_H : system_func_t
        Hessian of the Hamiltonian w.r.t. z = (q, p).
    parameters : NDArray[np.float64]
        Additional parameters passed to the gradient and Hessian functions.
    tol : np.float64
        Newton convergence tolerance on the residual norm.
    max_iter : int
        Maximum Newton iterations per step.
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
            q, p, Q, time_step, eom, hess_H, parameters, tol, max_iter
        )

    return q, p, Q
