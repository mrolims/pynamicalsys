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

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray


from pynamicalsys.common.types import grad_t, symplectic_step_t


@njit
def generate_poincare_section(
    q: NDArray[np.float64],
    p: NDArray[np.float64],
    num_intersections: np.int64,
    parameters: NDArray[np.float64],
    grad_T: grad_t,
    grad_V: grad_t,
    time_step: np.float64,
    integrator: symplectic_step_t,
    section_index: int = 0,
    section_value: np.float64 = np.float64(0.0),
    crossing: int = 1,
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
    grad_T : grad_t
        Gradient of the kinetic energy with respect to the momenta.
    grad_V : grad_t
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
        q_new, p_new = integrator(q_prev, p_prev, time_step, grad_T, grad_V, parameters)

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


@njit(parallel=True)
def ensemble_poincare_section(
    q: NDArray[np.float64],
    p: NDArray[np.float64],
    num_intersections: np.int64,
    parameters: NDArray[np.float64],
    grad_T: grad_t,
    grad_V: grad_t,
    time_step: np.float64,
    integrator: symplectic_step_t,
    section_index: int = 0,
    section_value: np.float64 = np.float64(0.0),
    crossing: int = 1,
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
    grad_T : grad_t
        Gradient of the kinetic energy with respect to the momenta.
    grad_V : grad_t
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

    Returns
    -------
    NDArray[np.float64]
        Array of shape `(num_ic, num_intersections, 2 * dof + 1)` containing the
        Poincaré section points for each initial condition, with the first column
        storing the crossing times.
    """
    num_ic, dof = q.shape
    section_points = np.zeros(
        (num_ic, num_intersections, 2 * dof + 1),
        dtype=np.float64,
    )

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
