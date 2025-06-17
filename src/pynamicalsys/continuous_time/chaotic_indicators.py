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
from pynamicalsys.continuous_time.numerical_integrators import (
    rk4_step,
    variational_rk4_step,
)


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
    return_history: bool = False,
    sample_times: Optional[NDArray[np.float64]] = None,
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
            u, parameters, transient_time, equations_of_motion, time_step=time_step
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

    number_of_steps = round(sample_time / time_step)
    exponents = np.zeros(neq, dtype=np.float64)

    if return_history:
        if sample_times is not None:
            history = np.zeros((len(sample_times), neq + 1))
            count = 0
        else:
            history = np.zeros((number_of_steps, neq + 1))

    for i in range(number_of_steps):
        uv = variational_rk4_step(
            time, uv, parameters, equations_of_motion, jacobian, time_step=time_step
        )

        # Reshape the deviation vectors into a neq x neq matrix
        v = uv[neq:].reshape(neq, neq)

        # Perform the QR decomposition
        v, R = QR(v)

        # Accumulate the log
        exponents += np.log(np.abs(np.diag(R))) / np.log(log_base)

        if return_history:
            if sample_times is None:
                history[i, 0] = time
                history[i, 1:] = exponents / (i + 1)
            elif i in sample_times:
                history[count, 0] = time
                history[count, 1:] = exponents / (i + 1)
                count += 1

        # Reshape v back to uv
        uv[neq:] = v.reshape(neq**2)

        # Update time
        time += time_step

    if return_history:
        return history
    else:
        aux_exponents = np.zeros((neq, 1))
        aux_exponents[:, 0] = exponents / (number_of_steps * time_step)
        return aux_exponents


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
    return_history: bool = False,
    sample_times: Optional[NDArray[np.float64]] = None,
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
            u, parameters, transient_time, equations_of_motion, time_step=time_step
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
    v = uv[neq:].reshape(neq, ndv)
    v, _ = qr(v)
    uv[neq:] = v.reshape(neq * ndv)

    number_of_steps = round(sample_time / time_step)

    if return_history:
        if sample_times is not None:
            history = np.zeros((len(sample_times), 2))
            count = 0
        else:
            history = np.zeros((number_of_steps, 2))

    for i in range(number_of_steps):
        uv = variational_rk4_step(
            time,
            uv,
            parameters,
            equations_of_motion,
            jacobian,
            time_step=time_step,
            number_of_deviation_vectors=ndv,
        )

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
            if sample_times is None:
                history[i, 0] = time
                history[i, 1] = sali
            elif i in sample_times:
                history[count, 0] = time
                history[count, 1] = sali
                count += 1

        # Early termination
        if sali <= threshold:
            break

        # Reshape v back to uv
        uv[neq:] = v.reshape(neq * ndv)

        # Update time
        time += time_step

    if return_history:
        if sample_times is not None:
            return history[:count]
        else:
            return history[:i]
    else:
        aux_sali = np.zeros((2, 1), dtype=np.float64)
        aux_sali[0, 0] = time
        aux_sali[1, 0] = sali

        return aux_sali


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
    return_history: bool = False,
    sample_times: Optional[NDArray[np.float64]] = None,
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
            u, parameters, transient_time, equations_of_motion, time_step=time_step
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
    v = uv[neq:].reshape(neq, ndv)
    v, _ = qr(v)
    uv[neq:] = v.reshape(neq * ndv)

    number_of_steps = round(sample_time / time_step)

    if return_history:
        if sample_times is not None:
            history = np.zeros((len(sample_times), 2))
            count = 0
        else:
            history = np.zeros((number_of_steps, 2))

    for i in range(number_of_steps):
        uv = variational_rk4_step(
            time,
            uv,
            parameters,
            equations_of_motion,
            jacobian,
            time_step=time_step,
            number_of_deviation_vectors=ndv,
        )

        # Reshape the deviation vectors into a neq x ndv matrix
        v = uv[neq:].reshape(neq, ndv)

        # Normalize the deviation vectors
        for j in range(ndv):
            v[:, j] /= np.linalg.norm(v[:, j])

        # Calculate the singular value decomposition
        S = np.linalg.svd(v, full_matrices=False, compute_uv=False)
        ldi = np.prod(S)

        if return_history:
            if sample_times is None:
                history[i, 0] = time
                history[i, 1] = ldi
            elif i in sample_times:
                history[count, 0] = time
                history[count, 1] = ldi
                count += 1

        # Early termination
        if ldi <= threshold:
            break

        # Reshape v back to uv
        uv[neq:] = v.reshape(neq * ndv)

        # Update time
        time += time_step

    if return_history:
        if sample_times is not None:
            return history[:count]
        else:
            return history[:i]
    else:
        aux_ldi = np.zeros((2, 1), dtype=np.float64)
        aux_ldi[0, 0] = time
        aux_ldi[1, 0] = ldi

        return aux_ldi


if __name__ == "__main__":
    from pynamicalsys.continuous_time.models import (
        lorenz_system,
        lorenz_jacobian,
    )

    neq = 3
    u = np.array([0.1, 0.1, 0.1])
    total_time = 1000
    transient_time = 500
    parameters = np.array([16.0, 45.92, 4.0])
    sali = SALI(
        u,
        parameters,
        total_time,
        lorenz_system,
        lorenz_jacobian,
        transient_time=transient_time,
    )
    print(sali, sali.shape)
    print(sali[0, 0])
