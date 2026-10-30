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

from pynamicalsys.hamiltonian_systems.coefficients import ALPHA, BETA
from pynamicalsys.common.types import system_func_t


@njit
def velocity_verlet_2nd_step(
    q: NDArray[np.float64],
    p: NDArray[np.float64],
    time_step: np.float64,
    grad_T: system_func_t,
    grad_V: system_func_t,
    parameters: NDArray[np.float64],
    tol: np.float64
    | None = None,  # Added just to match the midpoint methods's signature
    max_iter: int | None = None,  # Added just to match the midpoint method's signature
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Perform one step of the second-order velocity Verlet symplectic integrator.

    Parameters
    ----------
    q : NDArray[np.float64]
        Current generalized coordinates.
    p : NDArray[np.float64]
        Current generalized momenta.
    time_step : np.float64
        Integration time step.
    grad_T : system_func_t
        Gradient of the kinetic energy with respect to the momenta.
    grad_V : system_func_t
        Gradient of the potential energy with respect to the coordinates.
    parameters : NDArray[np.float64]
        Additional parameters passed to `grad_T` and `grad_V`.

    Returns
    -------
    tuple[NDArray[np.float64], NDArray[np.float64]]
        Updated coordinates and momenta after one integration step.
    """
    q_new = q.copy()
    p_new = p.copy()
    half_step = np.float64(0.5) * time_step

    # Element loops perform the operations of the array expressions, in the
    # same order, without temporary arrays.
    gradV = grad_V(q, parameters)
    for i in range(p_new.shape[0]):
        p_new[i] -= half_step * gradV[i]

    gradT = grad_T(p_new, parameters)
    for i in range(q_new.shape[0]):
        q_new[i] += time_step * gradT[i]

    gradV = grad_V(q_new, parameters)
    for i in range(p_new.shape[0]):
        p_new[i] -= half_step * gradV[i]

    return q_new, p_new


@njit
def yoshida_4th_step(
    q: NDArray[np.float64],
    p: NDArray[np.float64],
    time_step: np.float64,
    grad_T: system_func_t,
    grad_V: system_func_t,
    parameters: NDArray[np.float64],
    tol: np.float64
    | None = None,  # Added just to match the midpoint methods's signature
    max_iter: int | None = None,  # Added just to match the midpoint method's signature
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Perform one step of the fourth-order Yoshida symplectic integrator.

    This integrator is obtained by composing three second-order velocity Verlet
    steps with coefficients `ALPHA` and `BETA`.

    Parameters
    ----------
    q : NDArray[np.float64]
        Current generalized coordinates.
    p : NDArray[np.float64]
        Current generalized momenta.
    time_step : np.float64
        Integration time step.
    grad_T : system_func_t
        Gradient of the kinetic energy with respect to the momenta.
    grad_V : system_func_t
        Gradient of the potential energy with respect to the coordinates.
    parameters : NDArray[np.float64]
        Additional parameters passed to `grad_T` and `grad_V`.

    Returns
    -------
    tuple[NDArray[np.float64], NDArray[np.float64]]
        Updated coordinates and momenta after one Yoshida step.
    """
    q_new, p_new = velocity_verlet_2nd_step(
        q, p, np.float64(ALPHA) * time_step, grad_T, grad_V, parameters
    )
    q_new, p_new = velocity_verlet_2nd_step(
        q_new, p_new, np.float64(BETA) * time_step, grad_T, grad_V, parameters
    )
    q_new, p_new = velocity_verlet_2nd_step(
        q_new, p_new, np.float64(ALPHA) * time_step, grad_T, grad_V, parameters
    )

    return q_new, p_new


@njit
def _symplectic_J_inv_matmul(
    n: int,
    M: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Compute J^{-1} @ M for the canonical symplectic matrix
    J = [[0, I], [-I, 0]], applied to a (2n, 2n) matrix M.

    J^{-1} = -J, so J^{-1} @ M swaps and negates the block rows:
        (J^{-1} @ M)[:n, :]  =  M[n:, :]
        (J^{-1} @ M)[n:, :]  = -M[:n, :]
    """
    out = np.empty_like(M)
    out[:n, :] = M[n:, :]
    out[n:, :] = -M[:n, :]
    return out


@njit
def _midpoint_residual(
    z_bar: NDArray[np.float64],
    z_n: NDArray[np.float64],
    half_step: np.float64,
    qdot: NDArray[np.float64],
    pdot: NDArray[np.float64],
    F: NDArray[np.float64],
) -> None:
    """Store `F = z_bar - z_n - (h / 2) f(z_bar)` in `F`."""
    n = qdot.shape[0]
    for i in range(n):
        F[i] = z_bar[i] - z_n[i] - half_step * qdot[i]
        F[n + i] = z_bar[n + i] - z_n[n + i] - half_step * pdot[i]


@njit
def _midpoint_newton_matrix(
    n: int, half_step: np.float64, H: NDArray[np.float64], DF: NDArray[np.float64]
) -> None:
    """
    Store `DF = I - (h / 2) J^{-1} H` in `DF`.

    `J^{-1} H` stacks the momentum rows of `H` over the negated position rows,
    as in `_symplectic_J_inv_matmul`, and is read from `H` directly.
    """
    for row in range(2 * n):
        for col in range(2 * n):
            if row < n:
                jinv_h = H[row + n, col]
            else:
                jinv_h = -H[row - n, col]
            identity = 1.0 if row == col else 0.0
            DF[row, col] = identity - half_step * jinv_h


@njit
def implicit_midpoint_step(
    q: NDArray[np.float64],
    p: NDArray[np.float64],
    time_step: np.float64,
    eom: system_func_t,
    hess_H: system_func_t,
    parameters: NDArray[np.float64],
    tol: np.float64 = np.float64(1e-12),
    max_iter: int = 50,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Perform one step of the implicit midpoint symplectic integrator.

    Solves, via Newton's method, for the midpoint state
    z_bar = (q_n + q_{n+1}) / 2, (p_n + p_{n+1}) / 2
    satisfying
        z_bar = z_n + (h/2) * J^{-1} grad_H(z_bar)
    then sets z_{n+1} = 2 * z_bar - z_n.

    Works for arbitrary (including non-separable) Hamiltonians H(q, p),
    unlike velocity_verlet_2nd_step / yoshida_4th_step which require
    H = T(p) + V(q).

    If the Newton iteration fails to converge within `max_iter` steps,
    the residual norm, `tol`, and `max_iter` are printed and a
    `RuntimeError` is raised. Numba-compiled code cannot include these
    values in the exception message, so they are printed instead.

    Parameters
    ----------
    q : NDArray[np.float64]
        Current generalized coordinates.
    p : NDArray[np.float64]
        Current generalized momenta.
    time_step : np.float64
        Integration time step.
    eom : eom_t
        Equations of motion: eom(q, p, parameters) -> (dH/dp, -dH/dq).
    hess_H : hess_H_t
        Full Hessian of H w.r.t. z=(q,p), shape (2n, 2n).
    parameters : NDArray[np.float64]
        Additional parameters passed to `eom` and `hess_H`.
    tol : np.float64
        Newton convergence tolerance on the residual norm.
    max_iter : int
        Maximum Newton iterations per step.

    Returns
    -------
    tuple[NDArray[np.float64], NDArray[np.float64]]
        Updated coordinates and momenta after one integration step.
    """
    n = q.shape[0]
    h = time_step

    half_step = np.float64(0.5) * h

    # z_n, the midpoint z_bar (initially z_n itself), the residual F, and the
    # Newton matrix DF are kept in buffers that are reused by every Newton
    # iteration.
    z_n = np.empty(2 * n, dtype=np.float64)
    z_n[:n] = q
    z_n[n:] = p
    z_bar = z_n.copy()
    F = np.empty(2 * n, dtype=np.float64)
    DF = np.empty((2 * n, 2 * n), dtype=np.float64)

    converged = False
    residual_norm = np.float64(0.0)

    for _ in range(max_iter):
        q_bar = z_bar[:n]
        p_bar = z_bar[n:]
        qdot, pdot = eom(q_bar, p_bar, parameters)  # J^{-1} grad_H(z_bar)
        _midpoint_residual(z_bar, z_n, half_step, qdot, pdot, F)

        residual_norm = np.sqrt(np.sum(F * F))
        if residual_norm < tol:
            converged = True
            break

        H_mid = hess_H(q_bar, p_bar, parameters)  # (2n,2n)
        _midpoint_newton_matrix(n, half_step, H_mid, DF)

        delta = np.linalg.solve(DF, -F)
        for i in range(2 * n):
            z_bar[i] = z_bar[i] + delta[i]

    q_bar = z_bar[:n]
    p_bar = z_bar[n:]

    if not converged:
        print(
            "implicit_midpoint_step: Newton iteration did not converge. "
            "residual norm =",
            residual_norm,
            " tol =",
            tol,
            " max_iter =",
            max_iter,
            ". Consider reducing time_step, increasing tol, or increasing max_iter.",
        )
        raise RuntimeError("implicit_midpoint_step failed to converge")

    q_new = np.float64(2.0) * q_bar - q
    p_new = np.float64(2.0) * p_bar - p

    return q_new, p_new
