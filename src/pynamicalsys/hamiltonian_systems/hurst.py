# hurst.py

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
from pynamicalsys.common.hurst import hurst_exponent
from pynamicalsys.common.types import symplectic_step_t, system_func_t
from pynamicalsys.hamiltonian_systems.poincare import generate_poincare_section_sep


def hurst_exponent_wrapped(
    q: NDArray[np.float64],
    p: NDArray[np.float64],
    num_points: np.int64,
    parameters: NDArray[np.float64],
    system_func_1: system_func_t,
    system_func_2: system_func_t,
    time_step: np.float64,
    integrator: symplectic_step_t,
    section_index: int,
    section_value: np.float64,
    crossing: int,
    wmin: int = 2,
    tol: np.float64 = np.float64(1e-12),
    max_iter: int = 50,
    pss_func=generate_poincare_section_sep,
) -> NDArray[np.float64]:
    """
    Estimate the Hurst exponent from a Hamiltonian Poincaré section.

    Parameters
    ----------
    q : NDArray[np.float64]
        Initial generalized coordinates of shape `(dof,)`.
    p : NDArray[np.float64]
        Initial generalized momenta of shape `(dof,)`.
    num_points : np.int32
        Number of Poincaré-section crossings used in the analysis.
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
    section_index : int
        Index of the coordinate used to define the Poincaré section.
    section_value : np.float64
        Value of the section coordinate.
    crossing : int
        Crossing rule:
        - `+1` for upward crossings
        - `-1` for downward crossings
        - `0` for all crossings
    wmin : int, optional
        Minimum window size used in the rescaled-range calculation.
    tol : np.float64
        Newton convergence tolerance on the residual norm. Only used by the implicit midpoint integrator (imp).
    max_iter : int
        Maximum Newton iterations per step. Only used by the implicit midpoint integrator (imp).

    Returns
    -------
    NDArray[np.float64]
        Estimated Hurst exponent values for the reduced Poincaré-section coordinates.
    """
    q = q.copy()
    p = p.copy()

    points = pss_func(
        q=q,
        p=p,
        num_intersections=num_points,
        parameters=parameters,
        system_func_1=system_func_1,
        system_func_2=system_func_2,
        time_step=time_step,
        integrator=integrator,
        section_index=section_index,
        section_value=section_value,
        crossing=crossing,
        tol=tol,
        max_iter=max_iter,
    )

    data = points[:, 1:]
    data = np.delete(data, section_index, axis=1)

    return hurst_exponent(data, wmin=wmin)
