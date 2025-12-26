# trajectory_analysis.py

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

from typing import Callable
import numpy as np
from numba import njit, prange
from numpy.typing import NDArray


@njit
def generate_trajectory(
    q: NDArray[np.float64],
    p: NDArray[np.float64],
    total_time: np.float64,
    parameters: NDArray[np.float64],
    grad_T: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    grad_V: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    time_step: np.float64,
    integrator: Callable[
        [
            NDArray[np.float64],
            NDArray[np.float64],
            float,
            Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
            Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
            NDArray[np.float64],
        ],
        tuple[NDArray[np.float64], NDArray[np.float64]],
    ],
) -> NDArray[np.float64]:
    """
    Generate a single trajectory of a Hamiltonian system using a symplectic integrator.

    Parameters
    ----------
    q : NDArray[np.float64], shape (dof,)
        Initial generalized coordinates.
    p : NDArray[np.float64], shape (dof,)
        Initial generalized momenta.
    total_time : float
        Total integration time.
    parameters : NDArray[np.float64]
        Additional system parameters passed to `grad_T` and `grad_V`.
    grad_T : Callable
        Function returning the gradient of the kinetic energy with respect to `p`.
    grad_V : Callable
        Function returning the gradient of the potential energy with respect to `q`.
    time_step : float
        Integration step size.
    integrator : Callable
        Symplectic integrator function (e.g. `velocity_verlet_2nd_step`,
        `yoshida_4th_step`).

    Returns
    -------
    result : NDArray[np.float64], shape (num_steps+1, 2*dof+1)
        Trajectory array:
        - Column 0: time values.
        - Columns 1..dof: generalized coordinates `q`.
        - Columns dof+1..2*dof: generalized momenta `p`.
    """
    num_steps = round(total_time / time_step)
    dof = len(q)
    result = np.zeros((num_steps + 1, 2 * dof + 1))

    result[0, 1 : dof + 1] = q
    result[0, dof + 1 :] = p
    for i in range(1, num_steps + 1):
        q, p = integrator(q, p, time_step, grad_T, grad_V, parameters)
        result[i, 0] = i * time_step
        result[i, 1 : dof + 1] = q
        result[i, dof + 1 :] = p

    return result


@njit(parallel=True)
def ensemble_trajectories(
    q: NDArray[np.float64],
    p: NDArray[np.float64],
    total_time: np.float64,
    parameters: NDArray[np.float64],
    grad_T: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    grad_V: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    time_step: np.float64,
    integrator: Callable,
) -> NDArray[np.float64]:
    """
    Generate an ensemble of trajectories from multiple initial conditions.

    Parameters
    ----------
    q : NDArray[np.float64], shape (num_ic, dof)
        Initial coordinates for each trajectory.
    p : NDArray[np.float64], shape (num_ic, dof)
        Initial momenta for each trajectory.
    total_time : float
        Total integration time.
    parameters : NDArray[np.float64]
        Additional system parameters passed to `grad_T` and `grad_V`.
    grad_T : Callable
        Function returning the gradient of the kinetic energy with respect to `p`.
    grad_V : Callable
        Function returning the gradient of the potential energy with respect to `q`.
    time_step : float
        Integration step size.
    integrator : Callable
        Symplectic integrator function.

    Returns
    -------
    trajectories : NDArray[np.float64], shape (num_ic, num_steps+1, 2*dof+1)
        Array of trajectories for all initial conditions.
    """
    num_steps = round(total_time / time_step)
    num_ic, dof = q.shape

    trajectories = np.zeros((num_ic, num_steps + 1, 2 * dof + 1), dtype=np.float64)

    for i in prange(num_ic):
        trajectory = generate_trajectory(
            q[i], p[i], total_time, parameters, grad_T, grad_V, time_step, integrator
        )
        trajectories[i] = trajectory

    return trajectories


@njit
def generate_poincare_section(
    q: NDArray[np.float64],
    p: NDArray[np.float64],
    num_intersections: np.int32,
    parameters: NDArray[np.float64],
    grad_T: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    grad_V: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    time_step: np.float64,
    integrator: Callable,
    section_index: int = 0,
    section_value: float = 0.0,
    crossing: int = 1,
) -> NDArray[np.float64]:
    """
    Generate a Poincaré section for a Hamiltonian system.

    Parameters
    ----------
    q : NDArray[np.float64], shape (dof,)
        Initial generalized coordinates.
    p : NDArray[np.float64], shape (dof,)
        Initial generalized momenta.
    num_intersections : int
        Total number of section crossings to record.
    parameters : NDArray[np.float64]
        Additional system parameters passed to `grad_T` and `grad_V`.
    grad_T : Callable
        Function returning the gradient of the kinetic energy with respect to `p`.
    grad_V : Callable
        Function returning the gradient of the potential energy with respect to `q`.
    time_step : float
        Integration step size.
    integrator : Callable
        Symplectic integrator function.
    section_index : int, optional
        Index of coordinate `q[i]` used to define the section (default 0).
    section_value : float, optional
        Value of `q[section_index]` defining the section (default 0.0).
    crossing : int, optional
        Crossing rule:
        - +1 : only upward crossings (dq/dt > 0),
        - -1 : only downward crossings (dq/dt < 0),
        -  0 : count all crossings.

    Returns
    -------
    section_points : NDArray[np.float64], shape (num_intersections, 2*dof)
        Phase-space points `(q, p)` recorded at section crossings.
    """
    dof = len(q)
    section_points = np.zeros((num_intersections, 2 * dof + 1))
    count = 0
    n_steps = 0
    q_prev, p_prev = q.copy(), p.copy()
    while count < num_intersections:
        q_new, p_new = integrator(q_prev, p_prev, time_step, grad_T, grad_V, parameters)

        # Check if crossing occurred
        if (q_prev[section_index] - section_value) * (
            q_new[section_index] - section_value
        ) < 0.0:
            lam = (section_value - q_prev[section_index]) / (
                q_new[section_index] - q_prev[section_index]
            )
            q_cross = (1 - lam) * q_prev + lam * q_new
            p_cross = (1 - lam) * p_prev + lam * p_new
            t_cross = n_steps * time_step + lam * time_step

            velocity = grad_T(p_cross, parameters)[section_index]

            if crossing == 0 or np.sign(velocity) == crossing:
                section_points[count, 0] = t_cross
                section_points[count, 1 : dof + 1] = q_cross
                section_points[count, dof + 1 :] = p_cross
                count += 1

        q_prev, p_prev = q_new, p_new
        n_steps += 1

    return section_points


import numpy as np
from numba import njit


@njit
def generate_poincare_section_from_traj(
    q,
    p,
    parameters,
    grad_T,
    time_step,
    section_index=0,
    section_value=0.0,
    crossing=1,
):
    """
    Returns
    -------
    section_points : ndarray, shape (n_hits, 1 + 2*dof)
        Rows are [t_cross, q_cross[0..dof-1], p_cross[0..dof-1]].
    section_k : ndarray, shape (n_hits,)
        Integer indices k such that each crossing lies between samples k and k+1
        (i.e., between q[k],p[k] and q[k+1],p[k+1]).
    """
    dof = q.shape[1]
    num_points = q.shape[0]

    # -------------------------
    # Pass 1: count crossings
    # -------------------------
    n_hits = 0
    for i in range(1, num_points):
        q_prev_i = q[i - 1, section_index]
        q_new_i = q[i, section_index]

        if (q_prev_i - section_value) * (q_new_i - section_value) < 0.0:
            # interpolate
            denom = q_new_i - q_prev_i
            if denom == 0.0:
                continue
            lam = (section_value - q_prev_i) / denom

            # p_cross needed for velocity test
            # p_cross = (1-lam)*p[i-1] + lam*p[i]
            # only section_index component of grad_T is needed, but grad_T takes full p
            p_cross = (1.0 - lam) * p[i - 1, :] + lam * p[i, :]
            vel = grad_T(p_cross, parameters)[section_index]

            if crossing == 0:
                n_hits += 1
            else:
                # match sign convention you used: np.sign(vel) == crossing
                if np.sign(vel) == crossing:
                    n_hits += 1

    # Allocate outputs
    section_points = np.empty((n_hits, 1 + 2 * dof), dtype=np.float64)
    section_k = np.empty(n_hits, dtype=np.int64)

    # -------------------------
    # Pass 2: fill outputs
    # -------------------------
    hit = 0
    for i in range(1, num_points):
        q_prev_i = q[i - 1, section_index]
        q_new_i = q[i, section_index]

        if (q_prev_i - section_value) * (q_new_i - section_value) < 0.0:
            denom = q_new_i - q_prev_i
            if denom == 0.0:
                continue
            lam = (section_value - q_prev_i) / denom

            q_cross = (1.0 - lam) * q[i - 1, :] + lam * q[i, :]
            p_cross = (1.0 - lam) * p[i - 1, :] + lam * p[i, :]

            # crossing time: between (i-1)*dt and i*dt
            t_cross = (i - 1) * time_step + lam * time_step

            vel = grad_T(p_cross, parameters)[section_index]
            ok = (crossing == 0) or (np.sign(vel) == crossing)

            if ok:
                section_points[hit, 0] = t_cross
                # q part
                for j in range(dof):
                    section_points[hit, 1 + j] = q_cross[j]
                # p part
                for j in range(dof):
                    section_points[hit, 1 + dof + j] = p_cross[j]

                section_k[hit] = i - 1  # crossing between (i-1) and i
                hit += 1

    return section_points, section_k


@njit(parallel=True)
def ensemble_poincare_section(
    q: NDArray[np.float64],
    p: NDArray[np.float64],
    num_intersections: int,
    parameters: NDArray[np.float64],
    grad_T: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    grad_V: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    time_step: np.float64,
    integrator: Callable,
    section_index: int = 0,
    section_value: float = 0.0,
    crossing: int = 1,
) -> NDArray[np.float64]:
    """
    Generate Poincaré sections for an ensemble of initial conditions.

    Parameters
    ----------
    q : NDArray[np.float64], shape (num_ic, dof)
        Initial coordinates for each trajectory.
    p : NDArray[np.float64], shape (num_ic, dof)
        Initial momenta for each trajectory.
    num_intersections : int
        Number of section crossings to record per trajectory.
    parameters : NDArray[np.float64]
        Additional system parameters.
    grad_T : Callable
        Gradient of kinetic energy with respect to `p`.
    grad_V : Callable
        Gradient of potential energy with respect to `q`.
    time_step : float
        Integration step size.
    integrator : Callable
        Symplectic integrator function.
    section_index : int, optional
        Index of coordinate `q[i]` used for the section (default 0).
    section_value : float, optional
        Value of `q[section_index]` defining the section (default 0.0).
    crossing : int, optional
        Crossing rule:
        - +1 : upward crossings,
        - -1 : downward crossings,
        -  0 : all crossings.

    Returns
    -------
    section_points : NDArray[np.float64], shape (num_ic, num_intersections, 2*dof + 1)
        Poincaré section points for each initial condition with the first column being the time at each crossing.
    """
    num_ic, dof = q.shape
    section_points = np.zeros((num_ic, num_intersections, 2 * dof + 1))
    for i in prange(num_ic):
        section_points[i] = generate_poincare_section(
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
        )

    return section_points
