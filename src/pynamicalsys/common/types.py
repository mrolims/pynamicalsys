# types.py

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

from typing import TypeAlias
import numpy as np
from collections.abc import Sequence, Callable
from numpy.typing import NDArray

int_t: TypeAlias = int | np.integer
numeric_t: TypeAlias = int | float | np.integer | np.floating
numeric_like_t: TypeAlias = numeric_t | Sequence[numeric_t] | NDArray[np.number]

map_t: TypeAlias = Callable[
    [NDArray[np.float64], NDArray[np.float64]],
    NDArray[np.float64],
]

jacobian_t: TypeAlias = Callable[
    [NDArray[np.float64], NDArray[np.float64], map_t],
    NDArray[np.float64],
]

observable_t = Callable[[NDArray[np.float64]], NDArray[np.float64]]

state_observable_t = Callable[[NDArray[np.float64]], numeric_t]

flow_t: TypeAlias = Callable[
    [np.float64, NDArray[np.float64], NDArray[np.float64]],
    NDArray[np.float64],
]

flow_jacobian_t: TypeAlias = Callable[
    [np.float64, NDArray[np.float64], NDArray[np.float64]],
    NDArray[np.float64],
]


grad_t: TypeAlias = Callable[
    [NDArray[np.float64], NDArray[np.float64]],
    NDArray[np.float64],
]

hess_t: TypeAlias = Callable[
    [NDArray[np.float64], NDArray[np.float64]],
    NDArray[np.float64],
]

system_func_t: TypeAlias = Callable[..., NDArray[np.float64]]

# Equations of motion: f(q, p, parameters) -> (dq/dt, dp/dt)
# i.e. (dH/dp, -dH/dq), supplied directly by the user (not assembled
# from grad_T/grad_V, since H need not split that way)
eom_t: TypeAlias = Callable[
    [NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    tuple[NDArray[np.float64], NDArray[np.float64]],
]

# Full Hessian of H w.r.t. the combined state z = (q, p), shape (2n, 2n).
# Block layout: [[H_qq, H_qp], [H_pq, H_pp]]
hess_H_t: TypeAlias = Callable[
    [NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    NDArray[np.float64],
]

symplectic_step_t: TypeAlias = Callable[
    [
        NDArray[np.float64],
        NDArray[np.float64],
        np.float64,
        system_func_t,
        system_func_t,
        NDArray[np.float64],
        np.float64,
        int,
    ],
    tuple[NDArray[np.float64], NDArray[np.float64]],
]

symplectic_tangent_step_t: TypeAlias = Callable[
    [
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        np.float64,
        grad_t,
        grad_t,
        hess_t,
        hess_t,
        NDArray[np.float64],
    ],
    tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
]
