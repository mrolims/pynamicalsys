# adaptive_step.py

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

from pynamicalsys.common.types import flow_t, flow_jacobian_t
from pynamicalsys.continuous_time.variational import variational_equations


# RK45 Dormand–Prince method coefficients
_RK45_C = np.array([0.0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0], dtype=np.float64)
_RK45_A = np.array(
    [
        [0, 0, 0, 0, 0, 0],
        [1 / 5, 0, 0, 0, 0, 0],
        [3 / 40, 9 / 40, 0, 0, 0, 0],
        [44 / 45, -56 / 15, 32 / 9, 0, 0, 0],
        [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0, 0],
        [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656, 0],
        [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
    ],
    dtype=np.float64,
)
_RK45_B5 = np.array(
    [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0],
    dtype=np.float64,
)
_RK45_B4 = np.array(
    [5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40],
    dtype=np.float64,
)


@njit
def rk45_step(
    t: np.float64,
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    equations_of_motion: flow_t,
    time_step: np.float64,
    atol: np.float64 = np.float64(1e-6),
    rtol: np.float64 = np.float64(1e-3),
) -> tuple[NDArray[np.float64], np.float64, np.float64, bool]:
    """
    Perform one adaptive Dormand-Prince RK45 step.

    Parameters
    ----------
    t : np.float64
        Current integration time.
    u : NDArray[np.float64]
        Current state vector.
    parameters : NDArray[np.float64]
        System parameters passed to the equations of motion.
    equations_of_motion : flow_t
        Continuous-time vector field with signature
        `(time, u, parameters) -> dudt`.
    time_step : np.float64
        Current integration step size.
    atol : np.float64, optional
        Absolute tolerance used in the local error control.
    rtol : np.float64, optional
        Relative tolerance used in the local error control.

    Returns
    -------
    tuple[NDArray[np.float64], np.float64, np.float64, bool]
        Tuple containing:
        - the fifth-order solution estimate
        - the advanced time
        - the suggested next step size
        - whether the step was accepted
    """
    k = np.empty((7, u.size), dtype=np.float64)

    for i in range(7):
        ti = t + _RK45_C[i] * time_step
        ui = u.copy()
        for j in range(i):
            ui += time_step * _RK45_A[i, j] * k[j]
        k[i] = equations_of_motion(ti, ui, parameters)

    u5 = u.copy()
    u4 = u.copy()
    for i in range(7):
        u5 += time_step * _RK45_B5[i] * k[i]
        u4 += time_step * _RK45_B4[i] * k[i]

    error = np.abs(u5 - u4)
    scale = atol + rtol * np.maximum(np.abs(u), np.abs(u5))
    error_ratio = error / scale
    err = np.max(error_ratio)

    if err == 0.0:
        factor = 2.0
    else:
        factor = 0.9 * err ** (-0.25)

    if factor < 0.1:
        factor = 0.1
    elif factor > 2.0:
        factor = 2.0

    time_step_new = time_step * factor
    accept = bool(err < 1.0)

    return u5, t + time_step, time_step_new, accept


@njit
def variational_rk45_step(
    t: np.float64,
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    equations_of_motion: flow_t,
    jacobian: flow_jacobian_t,
    time_step: np.float64,
    number_of_deviation_vectors: int | None = None,
    atol: np.float64 = np.float64(1e-6),
    rtol: np.float64 = np.float64(1e-3),
) -> tuple[NDArray[np.float64], np.float64, np.float64, bool]:
    """
    Perform one adaptive Dormand-Prince RK45 step for the extended variational system.

    Parameters
    ----------
    t : np.float64
        Current integration time.
    u : NDArray[np.float64]
        Current extended state vector containing the physical state and the
        flattened deviation vectors.
    parameters : NDArray[np.float64]
        System parameters passed to the equations of motion and Jacobian.
    equations_of_motion : flow_t
        Continuous-time vector field with signature
        `(time, u, parameters) -> dudt`.
    jacobian : flow_jacobian_t
        Jacobian of the vector field with signature
        `(time, u, parameters) -> J`.
    time_step : np.float64
        Current integration step size.
    number_of_deviation_vectors : int | None, optional
        Number of deviation vectors carried in the extended state.
    atol : np.float64, optional
        Absolute tolerance used in the local error control.
    rtol : np.float64, optional
        Relative tolerance used in the local error control.

    Returns
    -------
    tuple[NDArray[np.float64], np.float64, np.float64, bool]
        Tuple containing:
        - the fifth-order solution estimate for the extended state
        - the advanced time
        - the suggested next step size
        - whether the step was accepted
    """
    k = np.empty((7, u.size), dtype=np.float64)

    for i in range(7):
        ti = t + _RK45_C[i] * time_step
        ui = u.copy()
        for j in range(i):
            ui += time_step * _RK45_A[i, j] * k[j]
        k[i] = variational_equations(
            ti,
            ui,
            parameters,
            equations_of_motion,
            jacobian,
            number_of_deviation_vectors=number_of_deviation_vectors,
        )

    u5 = u.copy()
    u4 = u.copy()
    for i in range(7):
        u5 += time_step * _RK45_B5[i] * k[i]
        u4 += time_step * _RK45_B4[i] * k[i]

    error = np.abs(u5 - u4)
    scale = atol + rtol * np.maximum(np.abs(u), np.abs(u5))
    error_ratio = error / scale
    err = np.max(error_ratio)

    if err == 0.0:
        factor = 2.0
    else:
        factor = 0.9 * err ** (-0.25)

    if factor < 0.1:
        factor = 0.1
    elif factor > 2.0:
        factor = 2.0

    time_step_new = time_step * factor
    accept = bool(err < 1.0)

    return u5, t + time_step, time_step_new, accept
