# numerical_integrators.py

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

from typing import Optional, Callable  # , Union, Tuple, Dict, List, Any, Sequence
from numpy.typing import NDArray
import numpy as np
from numba import njit, prange

from pynamicalsys.continuous_time.models import variational_equations


@njit
def rk4_step(
    t: float,
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    equations_of_motion: Callable[
        [float, NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]
    ],
    time_step: float = 0.01,
) -> NDArray[np.float64]:

    k1 = equations_of_motion(t, u, parameters)
    k2 = equations_of_motion(t + 0.5 * time_step, u + 0.5 * time_step * k1, parameters)
    k3 = equations_of_motion(t + 0.5 * time_step, u + 0.5 * time_step * k2, parameters)
    k4 = equations_of_motion(t + time_step, u + time_step * k3, parameters)

    u_next = u + (time_step / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return u_next


@njit
def variational_rk4_step(
    t: float,
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    equations_of_motion: Callable[
        [float, NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]
    ],
    jacobian: Callable[
        [float, NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]
    ],
    time_step: float = 0.01,
    number_of_deviation_vectors: Optional[int] = None,
) -> NDArray[np.float64]:

    k1 = variational_equations(
        t,
        u,
        parameters,
        equations_of_motion,
        jacobian,
        number_of_deviation_vectors=number_of_deviation_vectors,
    )

    k2 = variational_equations(
        t + 0.5 * time_step,
        u + 0.5 * time_step * k1,
        parameters,
        equations_of_motion,
        jacobian,
        number_of_deviation_vectors=number_of_deviation_vectors,
    )
    k3 = variational_equations(
        t + 0.5 * time_step,
        u + 0.5 * time_step * k2,
        parameters,
        equations_of_motion,
        jacobian,
        number_of_deviation_vectors=number_of_deviation_vectors,
    )
    k4 = variational_equations(
        t + time_step,
        u + time_step * k3,
        parameters,
        equations_of_motion,
        jacobian,
        number_of_deviation_vectors=number_of_deviation_vectors,
    )

    u_next = u + (time_step / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return u_next


# RK45 Dormand–Prince method coefficients
RK45_C = np.array([0.0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0])
RK45_A = np.array(
    [
        [0, 0, 0, 0, 0, 0],
        [1 / 5, 0, 0, 0, 0, 0],
        [3 / 40, 9 / 40, 0, 0, 0, 0],
        [44 / 45, -56 / 15, 32 / 9, 0, 0, 0],
        [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0, 0],
        [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656, 0],
        [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
    ]
)
RK45_B5 = np.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0])
RK45_B4 = np.array(
    [5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40]
)


@njit
def rk45_step(t, u, parameters, equations_of_motion, time_step, atol=1e-6, rtol=1e-3):
    """Single adaptive step of RK45 (Dormand-Prince).

    Returns:
        u_new: next state
        t_new: next time
        h_new: adjusted step size
        accept: whether step was accepted
    """
    k = np.empty((7, u.size), dtype=np.float64)
    for i in range(7):
        ti = t + RK45_C[i] * time_step
        ui = u.copy()
        for j in range(i):
            ui += time_step * RK45_A[i, j] * k[j]
        k[i] = equations_of_motion(ti, ui, parameters)

    # Compute 5th and 4th order estimates
    u5 = u.copy()
    u4 = u.copy()
    for i in range(7):
        u5 += time_step * RK45_B5[i] * k[i]
        u4 += time_step * RK45_B4[i] * k[i]

    # Compute element-wise error estimate
    error = np.abs(u5 - u4)
    scale = atol + rtol * np.maximum(np.abs(u), np.abs(u5))
    error_ratio = error / scale
    err = np.max(error_ratio)

    # Adapt step size
    if err == 0:
        factor = 2.0
    else:
        factor = 0.9 * err**-0.25

    if factor < 0.1:
        factor = 0.1
    elif factor > 2.0:
        factor = 2.0

    time_step_new = time_step * factor

    accept = err < 1.0
    return u5, t + time_step, time_step_new, accept


@njit
def variational_rk45_step(
    t,
    u,
    parameters,
    equations_of_motion,
    jacobian,
    time_step,
    number_of_deviation_vectors=None,
    atol=1e-6,
    rtol=1e-3,
):
    """Single adaptive step of RK45 (Dormand-Prince).

    Returns:
        u_new: next state
        t_new: next time
        h_new: adjusted step size
        accept: whether step was accepted
    """
    k = np.empty((7, u.size), dtype=np.float64)
    for i in range(7):
        ti = t + RK45_C[i] * time_step
        ui = u.copy()
        for j in range(i):
            ui += time_step * RK45_A[i][j] * k[j]
        k[i] = variational_equations(
            ti,
            ui,
            parameters,
            equations_of_motion,
            jacobian,
            number_of_deviation_vectors=number_of_deviation_vectors,
        )

    # Compute 5th and 4th order estimates
    u5 = u.copy()
    u4 = u.copy()
    for i in range(7):
        u5 += time_step * RK45_B5[i] * k[i]
        u4 += time_step * RK45_B4[i] * k[i]

    # Compute element-wise error estimate
    error = np.abs(u5 - u4)
    scale = atol + rtol * np.maximum(np.abs(u), np.abs(u5))
    error_ratio = error / scale
    err = np.max(error_ratio)

    # Adapt step size
    if err == 0:
        factor = 2.0
    else:
        factor = 0.9 * err**-0.25
    if factor < 0.1:
        factor = 0.1
    elif factor > 2.0:
        factor = 2.0

    time_step_new = time_step * factor

    accept = err < 1.0
    return u5, t + time_step, time_step_new, accept


@njit
def rk4_step_wrapped(
    t: float,
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    equations_of_motion: Callable[
        [float, NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]
    ],
    jacobian=None,
    time_step: float = 0.01,
    number_of_deviation_vectors: Optional[int] = None,
    atol: float = 1e-6,  # unused, just to match signature
    rtol: float = 1e-3,  # unused, just to match signature
) -> tuple[NDArray[np.float64], float, float, bool]:
    """
    Wrapper around rk4_step to match rk45_step return format.
    Returns (u_next, t_next, h_next, accept) with accept always True.
    """

    if jacobian is None:
        u_next = rk4_step(t, u, parameters, equations_of_motion, time_step)
    else:
        u_next = variational_rk4_step(
            t,
            u,
            parameters,
            equations_of_motion,
            jacobian,
            time_step=time_step,
            number_of_deviation_vectors=number_of_deviation_vectors,
        )

    t_next = t + time_step
    h_next = time_step
    accept = True
    return u_next, t_next, h_next, accept


@njit
def rk45_step_wrapped(
    t: float,
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    equations_of_motion: Callable[
        [float, NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]
    ],
    jacobian=None,
    time_step: float = 0.01,
    number_of_deviation_vectors=None,
    atol: float = 1e-6,
    rtol: float = 1e-3,
) -> tuple[NDArray[np.float64], float, float, bool]:
    """
    Wrapper around rk4_step to match rk45_step return format.
    Returns (u_next, t_next, h_next, accept) with accept always True.
    """

    if jacobian is None:
        return rk45_step(
            t, u, parameters, equations_of_motion, time_step, atol=atol, rtol=rtol
        )
    else:
        return variational_rk45_step(
            t,
            u,
            parameters,
            equations_of_motion,
            jacobian,
            time_step,
            number_of_deviation_vectors=number_of_deviation_vectors,
            atol=atol,
            rtol=rtol,
        )


@njit
def estimate_initial_step(
    t0: float,
    u0: np.ndarray,
    parameters: np.ndarray,
    equations_of_motion: Callable[[float, np.ndarray, np.ndarray], np.ndarray],
    order: int = 5,  # Dormand-Prince method is 5th order
    atol: float = 1e-6,
    rtol: float = 1e-3,
) -> float:
    """Estimate a good initial time step for adaptive integration."""
    f0 = equations_of_motion(t0, u0, parameters)

    scale = atol + rtol * np.abs(u0)
    d0 = np.linalg.norm(u0 / scale)
    d1 = np.linalg.norm(f0 / scale)

    if d0 < 1e-5 or d1 < 1e-5:
        h0 = 1e-6
    else:
        h0 = 0.01 * d0 / d1

    # Take one Euler step to estimate second derivative
    u1 = u0 + h0 * f0
    f1 = equations_of_motion(t0 + h0, u1, parameters)
    d2 = np.linalg.norm((f1 - f0) / scale) / h0

    if d1 <= 1e-15 and d2 <= 1e-15:
        h1 = max(1e-6, h0 * 1e-3)
    else:
        h1 = (0.01 / max(d1, d2)) ** (1.0 / (order + 1))

    return min(100 * h0, h1)
