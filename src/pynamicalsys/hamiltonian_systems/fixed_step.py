# fixed_step.py

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

from pynamicalsys.hamiltonian_systems.coefficients import ALPHA, BETA
from pynamicalsys.hamiltonian_systems.types import grad_t


@njit
def velocity_verlet_2nd_step(
    q: NDArray[np.float64],
    p: NDArray[np.float64],
    time_step: np.float64,
    grad_T: grad_t,
    grad_V: grad_t,
    parameters: NDArray[np.float64],
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
    grad_T : grad_t
        Gradient of the kinetic energy with respect to the momenta.
    grad_V : grad_t
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

    gradV = grad_V(q, parameters)
    p_new -= np.float64(0.5) * time_step * gradV

    gradT = grad_T(p_new, parameters)
    q_new += time_step * gradT

    gradV = grad_V(q_new, parameters)
    p_new -= np.float64(0.5) * time_step * gradV

    return q_new, p_new


@njit
def yoshida_4th_step(
    q: NDArray[np.float64],
    p: NDArray[np.float64],
    time_step: np.float64,
    grad_T: grad_t,
    grad_V: grad_t,
    parameters: NDArray[np.float64],
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
    grad_T : grad_t
        Gradient of the kinetic energy with respect to the momenta.
    grad_V : grad_t
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
