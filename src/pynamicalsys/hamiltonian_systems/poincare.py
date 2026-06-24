# poincare.py

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

import numpy as np
from numba import njit
from numpy.typing import NDArray
from pynamicalsys.common.types import system_func_t, symplectic_step_t
from concurrent.futures import ProcessPoolExecutor


"""
TODO

- Factor out the common Poincaré section logic. The only difference
  between the integrators is the derivative callbacks:
  (grad_T, grad_V) vs. (eom, hess_H).
"""


@njit
def generate_poincare_section_sep(
    q: NDArray[np.float64],
    p: NDArray[np.float64],
    num_intersections: np.int64,
    parameters: NDArray[np.float64],
    grad_T: system_func_t,
    grad_V: system_func_t,
    time_step: np.float64,
    integrator: symplectic_step_t,
    section_index: int = 0,
    section_value: np.float64 = np.float64(0.0),
    crossing: int = 1,
    tol: np.float64 = np.float64(1e-12),
    max_iter: int = 50,
) -> NDArray[np.float64]:
    """
    Generate a Poincaré section for a Hamiltonian system using separable integrators (velocity_verlet_2nd_step or yoshida_4th_step).

    Parameters
    ----------
    q : NDArray[np.float64]
        Initial generalized coordinates of shape `(dof,)`.
    p : NDArray[np.float64]
        Initial generalized momenta of shape `(dof,)`.
    num_intersections : np.int32
        Number of section crossings to record.
    parameters : NDArray[np.float64]
        Additional system parameters passed to `grad_T` and `grad_V`.
    grad_T : system_func_t
        Gradient of the kinetic energy with respect to the momenta.
    grad_V : system_func_t
        Gradient of the potential energy with respect to the coordinates.
    time_step : np.float64
        Integration time step.
    integrator : symplectic_step_t
        Symplectic integration step.
    section_index : int, optional
        Index of the coordinate used to define the section.
    section_value : np.float64, optional
        Value of `q[section_index]` defining the section.
    crossing : int, optional
        Crossing rule:
        - `+1` for upward crossings
        - `-1` for downward crossings
        - `0` for all crossings
    tol : np.float64
        Newton convergence tolerance on the residual norm.
    max_iter : int
        Maximum Newton iterations per step.

    Returns
    -------
    NDArray[np.float64]
        Array of shape `(num_intersections, 2 * dof + 1)` containing:
        - column 0: crossing times
        - columns `1:dof+1`: coordinates at the crossing
        - columns `dof+1:2*dof+1`: momenta at the crossing
    """
    dof = len(q)
    section_points = np.zeros((num_intersections, 2 * dof + 1), dtype=np.float64)

    count = 0
    n_steps = 0

    q_prev = q.copy()
    p_prev = p.copy()

    while count < num_intersections:
        q_new, p_new = integrator(
            q_prev, p_prev, time_step, grad_T, grad_V, parameters, tol, max_iter
        )

        if (q_prev[section_index] - section_value) * (
            q_new[section_index] - section_value
        ) < np.float64(0.0):
            lam = (section_value - q_prev[section_index]) / (
                q_new[section_index] - q_prev[section_index]
            )

            q_cross = (np.float64(1.0) - lam) * q_prev + lam * q_new
            p_cross = (np.float64(1.0) - lam) * p_prev + lam * p_new
            t_cross = np.float64(n_steps) * time_step + lam * time_step

            velocity = grad_T(p_cross, parameters)[section_index]

            if crossing == 0 or np.sign(velocity) == crossing:
                section_points[count, 0] = t_cross
                section_points[count, 1 : dof + 1] = q_cross
                section_points[count, dof + 1 :] = p_cross
                count += 1

        q_prev = q_new
        p_prev = p_new
        n_steps += 1

    return section_points


@njit
def generate_poincare_section_midpoint(
    q: NDArray[np.float64],
    p: NDArray[np.float64],
    num_intersections: np.int64,
    parameters: NDArray[np.float64],
    eom: system_func_t,
    hess_H: system_func_t,
    time_step: np.float64,
    integrator: symplectic_step_t,
    section_index: int = 0,
    section_value: np.float64 = np.float64(0.0),
    crossing: int = 1,
    tol: np.float64 = np.float64(1e-12),
    max_iter: int = 50,
) -> NDArray[np.float64]:
    """
    Generate a Poincaré section for a Hamiltonian system.

    Parameters
    ----------
    q : NDArray[np.float64]
        Initial generalized coordinates of shape `(dof,)`.
    p : NDArray[np.float64]
        Initial generalized momenta of shape `(dof,)`.
    num_intersections : np.int32
        Number of section crossings to record.
    parameters : NDArray[np.float64]
        Additional system parameters passed to `grad_T` and `grad_V`.
    eom : system_func_t
        Equations of motion of the system.
    hess_H : system_func_t
        Hessian of the Hamiltonian w.r.t. z = (q, p).
    time_step : np.float64
        Integration time step.
    integrator : symplectic_step_t
        Symplectic integration step.
    section_index : int, optional
        Index of the coordinate used to define the section.
    section_value : np.float64, optional
        Value of `q[section_index]` defining the section.
    crossing : int, optional
        Crossing rule:
        - `+1` for upward crossings
        - `-1` for downward crossings
        - `0` for all crossings
    tol : np.float64
        Newton convergence tolerance on the residual norm.
    max_iter : int
        Maximum Newton iterations per step.

    Returns
    -------
    NDArray[np.float64]
        Array of shape `(num_intersections, 2 * dof + 1)` containing:
        - column 0: crossing times
        - columns `1:dof+1`: coordinates at the crossing
        - columns `dof+1:2*dof+1`: momenta at the crossing
    """
    dof = len(q)
    section_points = np.zeros((num_intersections, 2 * dof + 1), dtype=np.float64)

    count = 0
    n_steps = 0

    q_prev = q.copy()
    p_prev = p.copy()

    while count < num_intersections:
        q_new, p_new = integrator(
            q_prev, p_prev, time_step, eom, hess_H, parameters, tol, max_iter
        )

        if (q_prev[section_index] - section_value) * (
            q_new[section_index] - section_value
        ) < np.float64(0.0):
            lam = (section_value - q_prev[section_index]) / (
                q_new[section_index] - q_prev[section_index]
            )

            q_cross = (np.float64(1.0) - lam) * q_prev + lam * q_new
            p_cross = (np.float64(1.0) - lam) * p_prev + lam * p_new
            t_cross = np.float64(n_steps) * time_step + lam * time_step

            qdot, _ = eom(q_cross, p_cross, parameters)[section_index]

            if crossing == 0 or np.sign(qdot) == crossing:
                section_points[count, 0] = t_cross
                section_points[count, 1 : dof + 1] = q_cross
                section_points[count, dof + 1 :] = p_cross
                count += 1

        q_prev = q_new
        p_prev = p_new
        n_steps += 1

    return section_points


def ensemble_poincare_section_sep(
    q: NDArray[np.float64],
    p: NDArray[np.float64],
    num_intersections: np.int64,
    parameters: NDArray[np.float64],
    grad_T: system_func_t,
    grad_V: system_func_t,
    time_step: np.float64,
    integrator: symplectic_step_t,
    section_index: int = 0,
    section_value: np.float64 = np.float64(0.0),
    crossing: int = 1,
    tol: np.float64 = np.float64(1e-12),
    max_iter: int = 50,
    n_workers=10,
) -> NDArray[np.float64]:
    """
    Generate Poincaré sections for an ensemble of initial conditions.

    Parameters
    ----------
    q : NDArray[np.float64]
        Initial generalized coordinates of shape `(num_ic, dof)`.
    p : NDArray[np.float64]
        Initial generalized momenta of shape `(num_ic, dof)`.
    num_intersections : int
        Number of section crossings to record for each trajectory.
    parameters : NDArray[np.float64]
        Additional system parameters passed to `grad_T` and `grad_V`.
    grad_T : system_func_t
        Gradient of the kinetic energy with respect to the momenta.
    grad_V : system_func_t
        Gradient of the potential energy with respect to the coordinates.
    time_step : np.float64
        Integration time step.
    integrator : symplectic_step_t
        Symplectic integration step.
    section_index : int, optional
        Index of the coordinate used to define the section.
    section_value : np.float64, optional
        Value of `q[section_index]` defining the section.
    crossing : int, optional
        Crossing rule:
        - `+1` for upward crossings
        - `-1` for downward crossings
        - `0` for all crossings
    tol : np.float64
        Newton convergence tolerance on the residual norm.
    max_iter : int
        Maximum Newton iterations per step.

    Returns
    -------
    NDArray[np.float64]
        Array of shape `(num_ic, num_intersections, 2 * dof + 1)` containing the
        Poincaré section points for each initial condition, with the first column
        storing the crossing times.
    """
    num_ic = q.shape[0]

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(
                generate_poincare_section_sep,
                q[i],
                p[i],
                num_intersections,
                parameters,
                grad_T,
                grad_V,
                time_step,
                integrator,
                section_index,
                section_value,
                crossing,
                tol,
                max_iter,
            )
            for i in range(num_ic)
        ]
        results = [future.result() for future in futures]

    return np.stack(results)


def ensemble_poincare_section_midpoint(
    q: NDArray[np.float64],
    p: NDArray[np.float64],
    num_intersections: np.int64,
    parameters: NDArray[np.float64],
    eom: system_func_t,
    hess_H: system_func_t,
    time_step: np.float64,
    integrator: symplectic_step_t,
    section_index: int = 0,
    section_value: np.float64 = np.float64(0.0),
    crossing: int = 1,
    tol: np.float64 = np.float64(1e-12),
    max_iter: int = 50,
    n_workers=-1,
) -> NDArray[np.float64]:
    """
    Generate Poincaré sections for an ensemble of initial conditions using the midpoint implicit method.

    Parameters
    ----------
    q : NDArray[np.float64]
        Initial generalized coordinates of shape `(num_ic, dof)`.
    p : NDArray[np.float64]
        Initial generalized momenta of shape `(num_ic, dof)`.
    num_intersections : int
        Number of section crossings to record for each trajectory.
    parameters : NDArray[np.float64]
        Additional system parameters passed to `grad_T` and `grad_V`.
    eom : system_func_t
        Equations of motion of the system.
    hess_H : system_func_t
        Hessian of the Hamiltonian w.r.t. z = (q, p)
    time_step : np.float64
        Integration time step.
    integrator : symplectic_step_t
        Symplectic integration step.
    section_index : int, optional
        Index of the coordinate used to define the section.
    section_value : np.float64, optional
        Value of `q[section_index]` defining the section.
    crossing : int, optional
        Crossing rule:
        - `+1` for upward crossings
        - `-1` for downward crossings
        - `0` for all crossings
    tol : np.float64
        Newton convergence tolerance on the residual norm.
    max_iter : int
        Maximum Newton iterations per step.

    Returns
    -------
    NDArray[np.float64]
        Array of shape `(num_ic, num_intersections, 2 * dof + 1)` containing the
        Poincaré section points for each initial condition, with the first column
        storing the crossing times.
    """
    num_ic = q.shape[0]

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(
                generate_poincare_section_midpoint,
                q[i],
                p[i],
                num_intersections,
                parameters,
                eom,
                hess_H,
                time_step,
                integrator,
                section_index,
                section_value,
                crossing,
                tol,
                max_iter,
            )
            for i in range(num_ic)
        ]
        results = [future.result() for future in futures]

    return np.stack(results)


@njit
def generate_poincare_section_from_traj(
    q: NDArray[np.float64],
    p: NDArray[np.float64],
    parameters: NDArray[np.float64],
    grad_T: system_func_t,
    time_step: np.float64,
    section_index: int = 0,
    section_value: np.float64 = np.float64(0.0),
    crossing: int = 1,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """
    Extract Poincaré-section crossings from precomputed trajectory samples.

    Parameters
    ----------
    q : NDArray[np.float64]
        Sampled generalized coordinates of shape `(num_points, dof)`.
    p : NDArray[np.float64]
        Sampled generalized momenta of shape `(num_points, dof)`.
    parameters : NDArray[np.float64]
        Additional system parameters passed to `grad_T`.
    grad_T : system_func_t
        Gradient of the kinetic energy with respect to the momenta.
    time_step : np.float64
        Time interval between successive stored trajectory samples.
    section_index : int, optional
        Index of the coordinate used to define the section.
    section_value : np.float64, optional
        Value of `q[:, section_index]` defining the section.
    crossing : int, optional
        Crossing rule:
        - `+1` for upward crossings
        - `-1` for downward crossings
        - `0` for all crossings

    Returns
    -------
    tuple[NDArray[np.float64], NDArray[np.int64]]
        - `section_points`: array of shape `(n_hits, 1 + 2 * dof)` whose rows are
          `[t_cross, q_cross..., p_cross...]`
        - `section_k`: integer array of shape `(n_hits,)` containing the index `k`
          such that each crossing lies between samples `k` and `k + 1`
    """
    dof = q.shape[1]
    num_points = q.shape[0]

    n_hits = 0
    for i in range(1, num_points):
        q_prev_i = q[i - 1, section_index]
        q_new_i = q[i, section_index]

        if (q_prev_i - section_value) * (q_new_i - section_value) < np.float64(0.0):
            denom = q_new_i - q_prev_i
            if denom == np.float64(0.0):
                continue

            lam = (section_value - q_prev_i) / denom
            p_cross = (np.float64(1.0) - lam) * p[i - 1, :] + lam * p[i, :]
            vel = grad_T(p_cross, parameters)[section_index]

            if crossing == 0 or np.sign(vel) == crossing:
                n_hits += 1

    section_points = np.empty((n_hits, 1 + 2 * dof), dtype=np.float64)
    section_k = np.empty(n_hits, dtype=np.int64)

    hit = 0
    for i in range(1, num_points):
        q_prev_i = q[i - 1, section_index]
        q_new_i = q[i, section_index]

        if (q_prev_i - section_value) * (q_new_i - section_value) < np.float64(0.0):
            denom = q_new_i - q_prev_i
            if denom == np.float64(0.0):
                continue

            lam = (section_value - q_prev_i) / denom

            q_cross = (np.float64(1.0) - lam) * q[i - 1, :] + lam * q[i, :]
            p_cross = (np.float64(1.0) - lam) * p[i - 1, :] + lam * p[i, :]
            t_cross = np.float64(i - 1) * time_step + lam * time_step

            vel = grad_T(p_cross, parameters)[section_index]
            ok = (crossing == 0) or (np.sign(vel) == crossing)

            if ok:
                section_points[hit, 0] = t_cross

                for j in range(dof):
                    section_points[hit, 1 + j] = q_cross[j]

                for j in range(dof):
                    section_points[hit, 1 + dof + j] = p_cross[j]

                section_k[hit] = i - 1
                hit += 1

    return section_points, section_k
