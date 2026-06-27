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
from pynamicalsys.common.poincare import detect_crossing, wrap_period
from concurrent.futures import ProcessPoolExecutor


@njit
def generate_poincare_section(
    q: NDArray[np.float64],
    p: NDArray[np.float64],
    num_intersections: np.int64,
    parameters: NDArray[np.float64],
    system_func_1: system_func_t,
    system_func_2: system_func_t,
    time_step: np.float64,
    integrator: symplectic_step_t,
    section_index: int = 0,
    section_value: np.float64 = np.float64(0.0),
    crossing: int = 1,
    periodic_section_coordinate: bool = False,
    period: np.float64 = np.float64(2.0 * np.pi),
    tol: np.float64 = np.float64(1e-12),
    max_iter: int = 50,
) -> NDArray[np.float64]:
    """
    Generate a Poincaré surface of section for a Hamiltonian system.

    The section is defined by monitoring a single coordinate q[section_index]
    and detecting crossings of a reference value. This coordinate can be either
    a real-valued variable or a periodic (angular) variable.

    If `periodic_section_coordinate=True`, the coordinate is treated as living
    on a circle S¹ with given `period`, and crossings are detected using wrapped
    differences modulo `period`. Otherwise, standard Euclidean sign-change
    crossings are used.

    Parameters
    ----------
    q : NDArray[np.float64]
        Initial generalized coordinates of shape `(dof,)`.
    p : NDArray[np.float64]
        Initial generalized momenta of shape `(dof,)`.
    num_intersections : np.int32
        Number of section crossings to record.
    parameters : NDArray[np.float64]
        Additional system parameters passed to `system_func_1` and `system_func_2`.
    system_func_1 : system_func_t
        Gradient of the kinetic energy with respect to the momenta when using the vv2 or svy4 integrators
        or the equations of motion when using the imp integrator.
    system_func_2 : system_func_t
        Gradient of the potential energy with respect to the coordinates when using the vv2 or svy4 integrators
        or the hessian of the Hamiltonian w.r.t. z = (q, p) when using the imp integrator.
    time_step : np.float64
        Integration time step.
    integrator : symplectic_step_t
        Symplectic integration step.
    section_index : int, optional
        Index of the coordinate used to define the section.
    section_value : np.float64, optional
        Value of q[section_index] defining the section.
    crossing : int, optional
        Crossing rule:
        - `+1` for upward crossings
        - `-1` for downward crossings
        - `0` for all crossings
    periodic_section_coordinate : bool, optional
        If True, treats q[section_index] as a periodic coordinate on S¹ and
        performs crossing detection using modulo arithmetic.
        If False, uses standard Euclidean crossing detection.
    period : np.float64, optional
        Period of the angular coordinate when
        `periodic_section_coordinate=True`.
        Typically 2π for action-angle systems.
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

    num_crossings_found = 0
    step = 0

    q_prev = q.copy()
    p_prev = p.copy()

    g_old = q_prev[section_index] - section_value
    if periodic_section_coordinate:
        g_old = wrap_period(g_old, period)

    while num_crossings_found < num_intersections:
        q_new, p_new = integrator(
            q_prev,
            p_prev,
            time_step,
            system_func_1,
            system_func_2,
            parameters,
            tol,
            max_iter,
        )

        if periodic_section_coordinate:
            raw_old = q_prev[section_index] - section_value
            raw_new = q_new[section_index] - section_value
            delta = wrap_period(raw_new - raw_old, period)
            g_new = g_old + delta
        else:
            g_new = q_new[section_index] - section_value

        if detect_crossing(g_old, g_new, crossing):
            lam = g_old / (g_old - g_new)
            q_cross = (np.float64(1.0) - lam) * q_prev + lam * q_new
            p_cross = (np.float64(1.0) - lam) * p_prev + lam * p_new
            t_cross = np.float64(step) * time_step + lam * time_step
            section_points[num_crossings_found, 0] = t_cross
            section_points[num_crossings_found, 1 : dof + 1] = q_cross
            section_points[num_crossings_found, dof + 1 :] = p_cross
            num_crossings_found += 1

        if periodic_section_coordinate:
            g_old = wrap_period(g_new, period)
        else:
            g_old = g_new

        q_prev = q_new
        p_prev = p_new
        step += 1

    return section_points


def ensemble_poincare_section(
    q: NDArray[np.float64],
    p: NDArray[np.float64],
    num_intersections: np.int64,
    parameters: NDArray[np.float64],
    system_func_1: system_func_t,
    system_func_2: system_func_t,
    time_step: np.float64,
    integrator: symplectic_step_t,
    section_index: int = 0,
    section_value: np.float64 = np.float64(0.0),
    crossing: int = 1,
    periodic_section_coordinate: bool = False,
    period: np.float64 = np.float64(2.0 * np.pi),
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
        Additional system parameters passed to `system_func_1` and `system_func_2`.
    system_func_1 : system_func_t
        Gradient of the kinetic energy with respect to the momenta.
    system_func_2 : system_func_t
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
    periodic_section_coordinate : bool, optional
        If True, treats q[section_index] as a periodic coordinate on S¹ and
        performs crossing detection using modulo arithmetic.
        If False, uses standard Euclidean crossing detection.
    period : np.float64, optional
        Period of the angular coordinate when
        `periodic_section_coordinate=True`.
        Typically 2π for action-angle systems.
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
                generate_poincare_section,
                q[i],
                p[i],
                num_intersections,
                parameters,
                system_func_1,
                system_func_2,
                time_step,
                integrator,
                section_index,
                section_value,
                crossing,
                periodic_section_coordinate,
                period,
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
    time_step: np.float64,
    section_index: int = 0,
    section_value: np.float64 = np.float64(0.0),
    crossing: int = 1,
    periodic_section_coordinate: bool = False,
    period: np.float64 = np.float64(2.0 * np.pi),
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """
    Extract Poincaré-section crossings from precomputed trajectory samples.

    Works for both separable and non-separable Hamiltonian systems:
    `system_func_1` must be the full equations-of-motion function
    `(qdot, pdot) = system_func_1(q, p, parameters)`. For a separable
    system, `qdot` may depend only on `p` internally; this function does
    not assume otherwise.

    Parameters
    ----------
    q : NDArray[np.float64]
        Sampled generalized coordinates of shape `(num_points, dof)`.
    p : NDArray[np.float64]
        Sampled generalized momenta of shape `(num_points, dof)`.
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
    periodic_section_coordinate : bool, optional
        If True, treats q[:, section_index] as a periodic coordinate on S¹
        with the given `period`, accumulating unbounded across samples
        (never re-wrapped). Crossing detection shifts the wrapped offset
        using delta arithmetic, mirroring generate_poincare_section.
        If False, uses standard Euclidean crossing detection.
    period : np.float64, optional
        Period of the angular coordinate when
        `periodic_section_coordinate=True`. Typically 2π.

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

    g_old = q[0, section_index] - section_value
    if periodic_section_coordinate:
        g_old = wrap_period(g_old, period)

    n_hits = 0
    for i in range(1, num_points):
        if periodic_section_coordinate:
            raw_old = q[i - 1, section_index] - section_value
            raw_new = q[i, section_index] - section_value
            delta = wrap_period(raw_new - raw_old, period)
            g_new = g_old + delta
        else:
            g_new = q[i, section_index] - section_value

        if detect_crossing(g_old, g_new, crossing):
            n_hits += 1

        g_old = wrap_period(g_new, period) if periodic_section_coordinate else g_new

    section_points = np.empty((n_hits, 1 + 2 * dof), dtype=np.float64)
    section_k = np.empty(n_hits, dtype=np.int64)
    hit = 0

    g_old = q[0, section_index] - section_value
    if periodic_section_coordinate:
        g_old = wrap_period(g_old, period)

    for i in range(1, num_points):
        if periodic_section_coordinate:
            raw_old = q[i - 1, section_index] - section_value
            raw_new = q[i, section_index] - section_value
            delta = wrap_period(raw_new - raw_old, period)
            g_new = g_old + delta
        else:
            g_new = q[i, section_index] - section_value

        if detect_crossing(g_old, g_new, crossing):
            lam = g_old / (g_old - g_new)
            q_cross = (np.float64(1.0) - lam) * q[i - 1, :] + lam * q[i, :]
            p_cross = (np.float64(1.0) - lam) * p[i - 1, :] + lam * p[i, :]
            t_cross = np.float64(i - 1) * time_step + lam * time_step
            section_points[hit, 0] = t_cross
            for j in range(dof):
                section_points[hit, 1 + j] = q_cross[j]
            for j in range(dof):
                section_points[hit, 1 + dof + j] = p_cross[j]
            section_k[hit] = i - 1
            hit += 1

        g_old = wrap_period(g_new, period) if periodic_section_coordinate else g_new

    return section_points, section_k
