# chaotic_indicators.py

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

from typing import Optional, Callable, Union, Tuple, Dict, List, Any, Sequence
from numpy.typing import NDArray
import numpy as np
from numba import njit, prange

from pynamicalsys.common.utils import qr
from pynamicalsys.continuous_time.trajectory_analysis import evolve_system
from pynamicalsys.continuous_time.numerical_integrators import rk4_step_wrapped


@njit(cache=True)
def lyapunov_exponents(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: float,
    equations_of_motion: Callable[
        [np.float64, NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]
    ],
    jacobian: Callable[
        [np.float64, NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]
    ],
    transient_time: Optional[float] = None,
    time_step: float = 0.01,
    atol: float = 1e-6,
    rtol: float = 1e-3,
    integrator=rk4_step_wrapped,
    return_history: bool = False,
    seed: int = 13,
    log_base: float = np.e,
    QR: Callable[
        [NDArray[np.float64]], Tuple[NDArray[np.float64], NDArray[np.float64]]
    ] = qr,
) -> NDArray[np.float64]:

    neq = len(u)  # Number of equations of the system
    nt = neq + neq**2  # Total number of equations including variational equations

    u = u.copy()

    # Handle transient time
    if transient_time is not None:
        u = evolve_system(
            u,
            parameters,
            transient_time,
            equations_of_motion,
            time_step=time_step,
            atol=atol,
            rtol=rtol,
            integrator=integrator,
        )
        sample_time = total_time - transient_time
        time = transient_time
    else:
        sample_time = total_time
        time = 0

    # State + deviation vectors
    uv = np.zeros(nt)
    uv[:neq] = u.copy()

    # Randomly define the deviation vectors and orthonormalize them
    np.random.seed(seed)
    uv[neq:] = -1 + 2 * np.random.rand(nt - neq)
    v = uv[neq:].reshape(neq, neq)
    v, _ = QR(v)
    uv[neq:] = v.reshape(neq**2)

    exponents = np.zeros(neq, dtype=np.float64)
    history = []

    while time < total_time:
        if time + time_step > total_time:
            time_step = total_time - time

        uv_new, time_new, time_step_new, accept = integrator(
            time,
            uv,
            parameters,
            equations_of_motion,
            jacobian=jacobian,
            time_step=time_step,
        )

        if accept:
            time = time_new
            uv = uv_new.copy()
            #  Reshape the deviation vectors into a neq x neq matrix
            v = uv[neq:].reshape(neq, neq).copy()

            # Perform the QR decomposition
            v, R = QR(v)

            # Accumulate the log
            exponents += np.log(np.abs(np.diag(R))) / np.log(log_base)

            if return_history:
                result = [time]
                for i in range(neq):
                    result.append(
                        exponents[i]
                        / (time - (transient_time if transient_time is not None else 0))
                    )
                history.append(result)

            # Reshape v back to uv
            uv[neq:] = v.reshape(neq**2)

        time_step = time_step_new

    if return_history:
        return history
    else:
        result = []
        for i in range(neq):
            result.append(
                exponents[i]
                / (time - (transient_time if transient_time is not None else 0))
            )
        return [result]


@njit(cache=True)
def SALI(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: float,
    equations_of_motion: Callable[
        [np.float64, NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]
    ],
    jacobian: Callable[
        [np.float64, NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]
    ],
    transient_time: Optional[float] = None,
    time_step: float = 0.01,
    atol: float = 1e-6,
    rtol: float = 1e-3,
    integrator=rk4_step_wrapped,
    return_history: bool = False,
    seed: int = 13,
    threshold: float = 1e-16,
) -> NDArray[np.float64]:

    neq = len(u)  # Number of equations of the system
    ndv = 2  # Number of deviation vectors
    nt = neq + neq * ndv  # Total number of equations including variational equations

    u = u.copy()

    # Handle transient time
    if transient_time is not None:
        u = evolve_system(
            u,
            parameters,
            transient_time,
            equations_of_motion,
            time_step=time_step,
            atol=atol,
            rtol=rtol,
            integrator=integrator,
        )
        time = transient_time
    else:
        time = 0

    # State + deviation vectors
    uv = np.zeros(nt)
    uv[:neq] = u.copy()

    # Randomly define the deviation vectors and orthonormalize them
    np.random.seed(seed)
    uv[neq:] = -1 + 2 * np.random.rand(nt - neq)
    v = uv[neq:].reshape(neq, ndv)
    v, _ = qr(v)
    uv[neq:] = v.reshape(neq * ndv)

    history = []

    while time < total_time:
        if time + time_step > total_time:
            time_step = total_time - time

        uv_new, time_new, time_step_new, accept = integrator(
            time,
            uv,
            parameters,
            equations_of_motion,
            jacobian=jacobian,
            time_step=time_step,
            number_of_deviation_vectors=ndv,
        )

        if accept:
            time = time_new
            uv = uv_new.copy()

            # Reshape the deviation vectors into a neq x ndv matrix
            v = uv[neq:].reshape(neq, ndv)

            # Normalize the deviation vectors
            v[:, 0] /= np.linalg.norm(v[:, 0])
            v[:, 1] /= np.linalg.norm(v[:, 1])

            # Calculate the aligment indexes and SALI
            PAI = np.linalg.norm(v[:, 0] + v[:, 1])
            AAI = np.linalg.norm(v[:, 0] - v[:, 1])
            sali = min(PAI, AAI)

            if return_history:
                result = [time, sali]
                history.append(result)

            # Early termination
            if sali <= threshold:
                break

            # Reshape v back to uv
            uv[neq:] = v.reshape(neq * ndv)

        time_step = time_step_new

    if return_history:
        return history
    else:
        return [[time, sali]]


# @njit(cache=True)
def LDI(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: float,
    equations_of_motion: Callable[
        [np.float64, NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]
    ],
    jacobian: Callable[
        [np.float64, NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]
    ],
    number_deviation_vectors: int,
    transient_time: Optional[float] = None,
    time_step: float = 0.01,
    atol: float = 1e-6,
    rtol: float = 1e-3,
    integrator=rk4_step_wrapped,
    return_history: bool = False,
    seed: int = 13,
    threshold: float = 1e-16,
) -> NDArray[np.float64]:

    neq = len(u)  # Number of equations of the system
    ndv = number_deviation_vectors  # Number of deviation vectors
    nt = neq + neq * ndv  # Total number of equations including variational equations

    u = u.copy()

    # Handle transient time
    if transient_time is not None:
        u = evolve_system(
            u,
            parameters,
            transient_time,
            equations_of_motion,
            time_step=time_step,
            atol=atol,
            rtol=rtol,
            integrator=integrator,
        )
        time = transient_time
    else:
        time = 0

    # State + deviation vectors
    uv = np.zeros(nt)
    uv[:neq] = u.copy()

    # Randomly define the deviation vectors and orthonormalize them
    np.random.seed(seed)
    uv[neq:] = -1 + 2 * np.random.rand(nt - neq)
    v = uv[neq:].reshape(neq, ndv)
    v, _ = qr(v)
    uv[neq:] = v.reshape(neq * ndv)

    history = []

    while time < total_time:
        if time + time_step > total_time:
            time_step = total_time - time

        uv_new, time_new, time_step_new, accept = integrator(
            time,
            uv,
            parameters,
            equations_of_motion,
            jacobian=jacobian,
            time_step=time_step,
            number_of_deviation_vectors=ndv,
        )

        if accept:
            time = time_new
            uv = uv_new.copy()

            # Reshape the deviation vectors into a neq x ndv matrix
            v = uv[neq:].reshape(neq, ndv)

            # Normalize the deviation vectors
            for i in range(ndv):
                v[:, i] /= np.linalg.norm(v[:, i])

            # Calculate the singular values
            S = np.linalg.svd(v, full_matrices=False, compute_uv=False)
            ldi = np.prod(S)

            if return_history:
                result = [time, ldi]
                history.append(result)

            # Early termination
            if ldi <= threshold:
                break

            # Reshape v back to uv
            uv[neq:] = v.reshape(neq * ndv)

        time_step = time_step_new

    if return_history:
        return history
    else:
        return [[time, ldi]]
