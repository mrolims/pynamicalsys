# models.py

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

from typing import Callable, Optional

import numpy as np
from numba import njit
from numpy.typing import NDArray


@njit
def lorenz_system(
    time: float, u: NDArray[np.float64], parameters: NDArray[np.float64]
) -> NDArray[np.float64]:
    sigma, rho, beta = parameters
    dudt = np.zeros_like(u)
    dudt[0] = sigma * (u[1] - u[0])
    dudt[1] = u[0] * (rho - u[2]) - u[1]
    dudt[2] = u[0] * u[1] - beta * u[2]

    return dudt


@njit
def lorenz_jacobian(
    time: float, u: NDArray[np.float64], parameters: NDArray[np.float64]
) -> NDArray[np.float64]:

    sigma, rho, beta = parameters

    J = np.empty((3, 3), dtype=np.float64)
    J[0, 0] = -sigma
    J[0, 1] = sigma
    J[0, 2] = 0.0

    J[1, 0] = rho - u[2]
    J[1, 1] = -1.0
    J[1, 2] = -u[0]

    J[2, 0] = u[1]
    J[2, 1] = u[0]
    J[2, 2] = -beta

    return J


@njit
def henon_heiles(
    time: float, u: NDArray[np.float64], parameters: NDArray[np.float64]
) -> NDArray[np.float64]:

    dudt = np.zeros_like(u)

    dudt[0] = u[2]  # dx / dt = px
    dudt[1] = u[3]  # dy / dt = py
    dudt[2] = -u[0] - 2 * u[0] * u[1]  # d(px) / dt = - x - 2xy
    dudt[3] = -u[1] - u[0] ** 2 + u[1] ** 2  # d(py) / dt = - y - x^2 + y^2

    return dudt


@njit
def henon_heiles_jacobian(
    time: float, u: NDArray[np.float64], parameters: NDArray[np.float64]
) -> NDArray[np.float64]:

    neq = len(u)
    J = np.zeros((neq, neq), dtype=np.float64)

    J[0, 0] = 0
    J[0, 1] = 0
    J[0, 2] = 1
    J[0, 3] = 0

    J[1, 0] = 0
    J[1, 1] = 0
    J[1, 2] = 0
    J[1, 3] = 1

    J[2, 0] = -1 - 2 * u[1]
    J[2, 1] = -2 * u[0]
    J[2, 2] = 0
    J[2, 3] = 0

    J[3, 0] = -2 * u[0]
    J[3, 1] = -1 + 2 * u[1]
    J[3, 2] = 0
    J[3, 3] = 0

    return J


@njit
def rossler_system(
    time: float, u: NDArray[np.float64], parameters: NDArray[np.float64]
) -> NDArray[np.float64]:

    a, b, c = parameters

    dudt = np.zeros_like(u)

    dudt[0] = -u[1] - u[2]
    dudt[1] = u[0] + a * u[1]
    dudt[2] = b + u[2] * (u[0] - c)

    return dudt


@njit
def rossler_system_jacobian(
    time: float, u: NDArray[np.float64], parameters: NDArray[np.float64]
) -> NDArray[np.float64]:

    a, b, c = parameters

    neq = len(u)
    J = np.zeros((neq, neq), dtype=np.float64)

    J[0, 0] = 0
    J[0, 1] = -1
    J[0, 2] = -1

    J[1, 0] = 1
    J[1, 1] = a
    J[1, 2] = 0

    J[2, 0] = u[2]
    J[2, 1] = 0
    J[2, 2] = u[0] - c

    return J


@njit
def rossler_system_4D(
    time: float, u: NDArray[np.float64], parameters: NDArray[np.float64]
) -> NDArray[np.float64]:

    a, b, c, d = parameters
    x, y, z, w = u
    dudt = np.zeros_like(u)

    dudt[0] = -(y + z)
    dudt[1] = x + a * y + w
    dudt[2] = b + x * z
    dudt[3] = -c * z + d * w

    return dudt


@njit
def rossler_system_4D_jacobian(
    time: float, u: NDArray[np.float64], parameters: NDArray[np.float64]
) -> NDArray[np.float64]:

    a, b, c, d = parameters
    x, y, z, w = u

    neq = len(u)
    J = np.zeros((neq, neq), dtype=np.float64)

    J[0, 0] = 0
    J[0, 1] = -1
    J[0, 2] = -1
    J[0, 3] = 0

    J[1, 0] = 1
    J[1, 1] = a
    J[1, 2] = 0
    J[1, 3] = 1

    J[2, 0] = z
    J[2, 1] = 0
    J[2, 2] = x
    J[2, 3] = 0

    J[3, 0] = 0
    J[3, 1] = 0
    J[3, 2] = -c
    J[3, 3] = d

    return J


@njit
def duffing(time, u, parameters):
    delta, alpha, beta, gamma, omega = parameters
    dudt = np.zeros_like(u)
    dudt[0] = u[1]
    dudt[1] = (
        -delta * u[1] + alpha * u[0] - beta * u[0] ** 3 + gamma * np.cos(omega * time)
    )

    return dudt


@njit
def duffing_jacobian(time, u, parameters):
    delta, alpha, beta, gamma, omega = parameters
    neq = len(u)
    J = np.zeros((neq, neq), dtype=np.float64)

    J[0, 0] = 0
    J[0, 1] = 1
    J[1, 0] = alpha - 3 * beta * u[0] ** 2
    J[1, 1] = -delta
    return J


@njit
def variational_equations(
    time: float,
    state: NDArray[np.float64],
    parameters: NDArray[np.float64],
    equations_of_motion: Callable[
        [np.float64, NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]
    ],
    jacobian: Callable[
        [np.float64, NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]
    ],
    number_of_deviation_vectors: Optional[int] = None,
) -> NDArray[np.float64]:

    state = state.copy()
    nt = len(state)  # Total number of equations

    if number_of_deviation_vectors is not None:
        ndv = number_of_deviation_vectors
        neq = round(nt / (1 + ndv))  # Number of system's equation
    else:
        neq = round((-1 + np.sqrt(1 + 4 * nt)) / 2)
        ndv = neq

    # Split the state into state variables, u, and deviation matrix, v
    u = state[:neq].copy()  # State vector
    v = state[neq:].reshape(neq, ndv).copy()  # Deviation matrix
    # Compute the Jacobian matrix
    J = jacobian(time, u, parameters)

    # Compute system's dynamics
    dudt = equations_of_motion(time, u, parameters)

    # Variational equation: dvdt = J * v
    dvdt = J @ v

    # Combine into a single output vector of length nt = neq + neq * ndv
    dstatedt = np.zeros(nt, dtype=np.float64)
    dstatedt[:neq] = dudt
    dstatedt[neq:] = dvdt.reshape(neq * ndv)

    return dstatedt
