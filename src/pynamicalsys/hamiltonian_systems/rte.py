# rte.py

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


from typing import Any

import numpy as np
from numpy.typing import NDArray

from pynamicalsys.common.recurrence_quantification_analysis import (
    RTEConfig,
    build_recurrence_matrix,
    calculate_threshold,
    white_vertline_distr,
)
from pynamicalsys.common.types import grad_t, symplectic_step_t
from pynamicalsys.hamiltonian_systems.poincare import generate_poincare_section


def recurrence_time_entropy(
    q: NDArray[np.float64],
    p: NDArray[np.float64],
    num_points: np.int64,
    parameters: NDArray[np.float64],
    grad_T: grad_t,
    grad_V: grad_t,
    time_step: np.float64,
    integrator: symplectic_step_t,
    section_index: int,
    section_value: np.float64,
    crossing: int,
    **kwargs: Any,
) -> (
    float
    | tuple[float, NDArray[np.float64]]
    | tuple[float, NDArray[np.uint8]]
    | tuple[float, NDArray[np.float64], NDArray[np.uint8]]
    | tuple[float, NDArray[np.float64], NDArray[np.float64]]
    | tuple[float, NDArray[np.uint8], NDArray[np.float64]]
    | tuple[float, NDArray[np.float64], NDArray[np.uint8], NDArray[np.float64]]
):
    """
    Compute the recurrence time entropy (RTE) from a Hamiltonian Poincaré section.

    Parameters
    ----------
    q : NDArray[np.float64]
        Initial generalized coordinates of shape `(dof,)`.
    p : NDArray[np.float64]
        Initial generalized momenta of shape `(dof,)`.
    num_points : np.int64
        Number of Poincaré-section crossings used in the recurrence analysis.
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
    section_index : int
        Index of the coordinate used to define the Poincaré section.
    section_value : np.float64
        Value of the section coordinate.
    crossing : int
        Crossing rule:
        - `+1` for upward crossings
        - `-1` for downward crossings
        - `0` for all crossings
    **kwargs : Any
        Additional keyword arguments passed to `RTEConfig`, including:
        - `metric`
        - `std_metric`
        - `threshold`
        - `threshold_mode`
        - `threshold_std`
        - `lmin`
        - `return_final_state`
        - `return_recmat`
        - `return_p`

    Returns
    -------
    float or tuple
        The RTE value, optionally followed by:
        - the final Poincaré-section point without time
        - the recurrence matrix
        - the white-vertical-line distribution
    """
    config = RTEConfig(**kwargs)

    points = generate_poincare_section(
        q=q,
        p=p,
        num_intersections=num_points,
        parameters=parameters,
        grad_T=grad_T,
        grad_V=grad_V,
        time_step=time_step,
        integrator=integrator,
        section_index=section_index,
        section_value=section_value,
        crossing=crossing,
    )

    data = points[:, 1:]
    data = np.delete(data, section_index, axis=1)

    eps = calculate_threshold(data, config)
    recmat = build_recurrence_matrix(data, eps, metric=config.metric)

    P = white_vertline_distr(recmat, wmin=config.lmin)
    P = P[P > 0]

    if P.size == 0:
        rte = 0.0
        P = P.astype(np.float64)
    else:
        P = P / P.sum()
        rte = float(-np.sum(P * np.log(P)))

    final_state = points[-1, 1:]

    if config.return_final_state and config.return_recmat and config.return_p:
        return rte, final_state, recmat, P
    elif config.return_final_state and config.return_recmat:
        return rte, final_state, recmat
    elif config.return_final_state and config.return_p:
        return rte, final_state, P
    elif config.return_recmat and config.return_p:
        return rte, recmat, P
    elif config.return_final_state:
        return rte, final_state
    elif config.return_recmat:
        return rte, recmat
    elif config.return_p:
        return rte, P
    else:
        return rte
