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

import numpy as np
from numba import njit
from numpy.typing import NDArray
from typing import Union, Sequence, Any

# ! -------------------- !
# ! --- Standard map --- !
# ! -------------------- !


@njit
def standard_map(
    u: NDArray[np.float64], parameters: Union[NDArray[np.float64], Sequence[float]]
) -> NDArray[np.float64]:
    k = parameters[0]
    x, y = u
    y_new = y + k * np.sin(2 * np.pi * x) / (2 * np.pi)
    x_new = x + y_new  # + 0.5

    x_new = x_new % 1
    y_new = y_new % 1

    return np.array([x_new, y_new])


@njit
def unbounded_standard_map(
    u: NDArray[np.float64], parameters: Union[NDArray[np.float64], Sequence[float]]
) -> NDArray[np.float64]:
    if len(parameters) == 1:
        k = parameters[0]
        gamma = 0
    else:
        k, gamma = parameters

    x, y = u
    y_new = (1 - gamma) * y + k * np.sin(2 * np.pi * x) / (2 * np.pi)
    x_new = x + y_new

    x_new = x_new % 1

    return np.array([x_new, y_new])


@njit
def standard_map_backwards(
    u: NDArray[np.float64], parameters: Union[NDArray[np.float64], Sequence[float]]
) -> NDArray[np.float64]:
    k = parameters[0]
    x, y = u
    x_new = x - y
    y_new = y - k * np.sin(2 * np.pi * x_new) / (2 * np.pi)

    x_new = x_new % 1
    y_new = y_new % 1

    return np.array([x_new, y_new])


@njit
def standard_map_jacobian(
    u: NDArray[np.float64],
    parameters: Union[NDArray[np.float64], Sequence[float]],
    *args: Any,
) -> NDArray[np.float64]:
    k = parameters[0]
    x, y = u
    return np.array(
        [[1 + k * np.cos(2 * np.pi * x), 1], [k * np.cos(2 * np.pi * x), 1]]
    )


# ! ----------------------------- !
# ! --- Standard nontwist map --- !
# ! ----------------------------- !


@njit
def standard_nontwist_map(
    u: NDArray[np.float64], parameters: Union[NDArray[np.float64], Sequence[float]]
) -> NDArray[np.float64]:

    a, b = parameters
    x, y = u

    y_new = y - b * np.sin(2 * np.pi * x)
    x_new = (x + a * (1 - y_new**2)) % 1.0

    return np.array([x_new, y_new])


@njit
def standard_nontwist_map_backwards(
    u: NDArray[np.float64], parameters: Union[NDArray[np.float64], Sequence[float]]
) -> NDArray[np.float64]:

    a, b = parameters

    x, y = u

    x_new = (x - a * (1 - y**2)) % 1.0
    y_new = y + b * np.sin(2 * np.pi * x_new)

    return np.array([x_new, y_new])


@njit
def standard_nontwist_map_jacobian(
    u: NDArray[np.float64],
    parameters: Union[NDArray[np.float64], Sequence[float]],
    *args: Any,
) -> NDArray[np.float64]:

    a, b = parameters
    x, _ = u
    _, y_new = standard_nontwist_map(u, parameters)
    dy_newdx = -2 * np.pi * b * np.cos(2 * np.pi * x)
    dy_newdy = 1

    J = np.zeros((2, 2))
    J[0, 0] = 1 - 2 * a * y_new * dy_newdx
    J[0, 1] = -2 * a * y_new * dy_newdy
    J[1, 0] = dy_newdx
    J[1, 1] = dy_newdy

    return J


# ! -------------------------------------- !
# ! --- Extended standard nontwist map --- !
# ! -------------------------------------- !


@njit
def extended_standard_nontwist_map(
    u: NDArray[np.float64], parameters: Union[NDArray[np.float64], Sequence[float]]
) -> NDArray[np.float64]:

    a, b, c, m = parameters

    x, y = u

    y_new = y - b * np.sin(2 * np.pi * x) - c * np.sin(2 * np.pi * m * x)
    x_new = (x + a * (1 - y_new**2)) % 1.0

    return np.array([x_new, y_new])


@njit
def extended_standard_nontwist_map_backwards(
    u: NDArray[np.float64], parameters: Union[NDArray[np.float64], Sequence[float]]
) -> NDArray[np.float64]:
    a, b, c, m = parameters
    x, y = u

    x_new = (x - a * (1 - y**2)) % 1.0
    y_new = y + b * np.sin(2 * np.pi * x_new) + c * np.sin(2 * np.pi * m * x_new)

    return np.array([x_new, y_new])


@njit
def extended_standard_nontwist_map_jacobian(
    u: NDArray[np.float64],
    parameters: Union[NDArray[np.float64], Sequence[float]],
    *args: Any,
) -> NDArray[np.float64]:
    a, b, c, m = parameters
    x, _ = u
    _, y_new = extended_standard_nontwist_map(u, parameters)
    dy_newdx = -2 * np.pi * b * np.cos(2 * np.pi * x) - 2 * np.pi * c * m * np.cos(
        2 * np.pi * m * x
    )
    dy_newdy = 1

    J = np.zeros((2, 2))
    J[0, 0] = 1 - 2 * a * y_new * dy_newdx
    J[0, 1] = -2 * a * y_new * dy_newdy
    J[1, 0] = dy_newdx
    J[1, 1] = dy_newdy

    return J


# ! ------------------ !
# ! --- Leonel map --- !
# ! ------------------ !


@njit
def leonel_map(
    u: NDArray[np.float64], parameters: Union[NDArray[np.float64], Sequence[float]]
) -> NDArray[np.float64]:
    eps, gamma = parameters

    x, y = u
    y_new = y + eps * np.sin(x)
    x_new = (x + 1 / abs(y_new) ** gamma) % (2 * np.pi)

    return np.array([x_new, y_new])


@njit
def leonel_map_backwards(
    u: NDArray[np.float64], parameters: Union[NDArray[np.float64], Sequence[float]]
) -> NDArray[np.float64]:
    eps, gamma = parameters

    x, y = u

    x_new = (x - 1 / abs(y) ** gamma) % (2 * np.pi)
    y_new = y - eps * np.sin(x_new)

    return np.array([x_new, y_new])


@njit
def leonel_map_jacobian(
    u: NDArray[np.float64],
    parameters: Union[NDArray[np.float64], Sequence[float]],
    *args: Any,
) -> NDArray[np.float64]:
    eps, gamma = parameters

    x, y = u
    _, y_new = leonel_map(u, parameters)
    dy_newdx = eps * np.cos(x)
    dy_newdy = 1

    J = np.zeros((2, 2))
    J[0, 0] = 1 - (gamma / (y_new * abs(y_new) ** gamma)) * dy_newdx
    J[0, 1] = -(gamma / (y_new * abs(y_new) ** gamma)) * dy_newdy
    J[1, 0] = dy_newdx
    J[1, 1] = dy_newdy

    return J


# ! ------------------------- !
# ! --- 4D symplectic map --- !
# ! ------------------------- !


@njit
def symplectic_map_4D(
    u: NDArray[np.float64], parameters: Union[NDArray[np.float64], Sequence[float]]
) -> NDArray[np.float64]:
    eps1, eps2, xi = parameters
    x1, x2, x3, x4 = u

    x1_new = x1 + x2
    x2_new = x2 - eps1 * np.sin(x1 + x2) - xi * (1 - np.cos(x1 + x2 + x3 + x4))
    x3_new = x3 + x4
    x4_new = x4 - eps2 * np.sin(x3 + x4) - xi * (1 - np.cos(x1 + x2 + x3 + x4))

    x1_new = x1_new % (2 * np.pi)
    x2_new = x2_new % (2 * np.pi)
    x3_new = x3_new % (2 * np.pi)
    x4_new = x4_new % (2 * np.pi)

    return np.array([x1_new, x2_new, x3_new, x4_new])


@njit
def symplectic_map_4D_backwards(
    u: NDArray[np.float64], parameters: Union[NDArray[np.float64], Sequence[float]]
) -> NDArray[np.float64]:
    eps1, eps2, xi = parameters
    x1, x2, x3, x4 = u

    x1_new = x1 - x2
    x2_new = x2 + eps1 * np.sin(x1 - x2) + xi * (1 - np.cos(x1 - x2 + x3 + x4))
    x3_new = x3 - x4
    x4_new = x4 + eps2 * np.sin(x3 - x4) + xi * (1 - np.cos(x1 - x2 + x3 + x4))

    x1_new = x1_new % (2 * np.pi)
    x2_new = x2_new % (2 * np.pi)
    x3_new = x3_new % (2 * np.pi)
    x4_new = x4_new % (2 * np.pi)

    return np.array([x1_new, x2_new, x3_new, x4_new])


@njit
def symplectic_map_4D_jacobian(
    u: NDArray[np.float64],
    parameters: Union[NDArray[np.float64], Sequence[float]],
    *args: Any,
) -> NDArray[np.float64]:
    eps1, eps2, xi = parameters

    J = np.zeros((4, 4))
    J[0, 0] = 1
    J[0, 1] = 1
    J[1, 0] = -eps1 * np.cos(u[0] + u[1]) - xi * np.sin(u[0] + u[1] + u[2] + u[3])
    J[1, 1] = 1 - eps1 * np.cos(u[0] + u[1]) - xi * np.sin(u[0] + u[1] + u[2] + u[3])
    J[1, 2] = -xi * np.sin(u[0] + u[1] + u[2] + u[3])
    J[1, 3] = -xi * np.sin(u[0] + u[1] + u[2] + u[3])
    J[2, 2] = 1
    J[2, 3] = 1
    J[3, 0] = -xi * np.sin(u[0] + u[1] + u[2] + u[3])
    J[3, 1] = -xi * np.sin(u[0] + u[1] + u[2] + u[3])
    J[3, 2] = -eps2 * np.cos(u[2] + u[3]) - xi * np.sin(u[0] + u[1] + u[2] + u[3])
    J[3, 3] = 1 - eps2 * np.cos(u[2] + u[3]) - xi * np.sin(u[0] + u[1] + u[2] + u[3])

    return J


# ! ----------------- !
# ! --- Henon map --- !
# ! ----------------- !


@njit
def henon_map(
    u: NDArray[np.float64], parameters: Union[NDArray[np.float64], Sequence[float]]
) -> NDArray[np.float64]:
    a, b = parameters
    x, y = u
    x_new = 1 - a * x**2 + y
    y_new = b * x
    return np.array([x_new, y_new])


@njit
def henon_map_jacobian(
    u: NDArray[np.float64],
    parameters: Union[NDArray[np.float64], Sequence[float]],
    *args: Any,
) -> NDArray[np.float64]:
    a, b = parameters
    x, y = u
    return np.array([[-2 * a * x, 1], [b, 0]])


# ! ----------------- !
# ! --- Lozi map --- !
# ! ----------------- !


@njit
def lozi_map(
    u: NDArray[np.float64], parameters: Union[NDArray[np.float64], Sequence[float]]
) -> NDArray[np.float64]:
    a, b = parameters
    x, y = u
    x_new = 1 - a * abs(x) + y
    y_new = b * x
    return np.array([x_new, y_new])


@njit
def lozi_map_jacobian(
    u: NDArray[np.float64],
    parameters: Union[NDArray[np.float64], Sequence[float]],
    *args: Any,
) -> NDArray[np.float64]:
    a, b = parameters
    x, y = u
    return np.array([[a * np.sign(x), 1], [b, 0]])


# ! -------------------- !
# ! --- Logistic map --- !
# ! -------------------- !


@njit
def logistic_map(
    u: NDArray[np.float64], parameters: Union[NDArray[np.float64], Sequence[float]]
) -> NDArray[np.float64]:
    r = parameters[0]
    x = u[0]
    x_new = r * x * (1 - x)
    return np.array([x_new])


@njit
def logistic_map_jacobian(
    u: NDArray[np.float64],
    parameters: Union[NDArray[np.float64], Sequence[float]],
    *args: Any,
) -> NDArray[np.float64]:
    r = parameters[0]
    x = u[0]
    return np.array([[r * (1 - 2 * x)]])


# ! ------------------ !
# ! --- Rulkov map --- !
# ! ------------------ !


@njit
def rulkov_map(
    u: NDArray[np.float64], parameters: Union[NDArray[np.float64], Sequence[float]]
) -> NDArray[np.float64]:
    alpha, sigma, mu = parameters
    x, y = u
    x_new = alpha / (1 + x**2) + y
    y_new = y - mu * (x - sigma)
    return np.array([x_new, y_new])


@njit
def rulkov_map_jacobian(
    u: NDArray[np.float64],
    parameters: Union[NDArray[np.float64], Sequence[float]],
    *args: Any,
) -> NDArray[np.float64]:
    alpha, sigma, mu = parameters
    x, y = u
    J = np.zeros((2, 2))
    J[0, 0] = -2 * alpha * x / (1 + x**2) ** 2
    J[0, 1] = 1
    J[1, 0] = -mu
    J[1, 1] = 1
    return J
