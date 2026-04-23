# sali.py

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
from pynamicalsys.common.utils import qr
from pynamicalsys.continuous_time.step_methods import rk4_step_wrapped
from pynamicalsys.continuous_time.step import evolve_system, step


@njit
def sali(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: np.float64,
    equations_of_motion: flow_t,
    jacobian: flow_jacobian_t,
    transient_time: np.float64 | None = None,
    time_step: np.float64 = np.float64(0.01),
    atol: np.float64 = np.float64(1e-6),
    rtol: np.float64 = np.float64(1e-3),
    integrator=rk4_step_wrapped,
    return_history: bool = False,
    seed: int = 13,
    threshold: np.float64 = np.float64(1e-16),
) -> NDArray[np.float64]:
    """
    Compute the Smaller Alignment Index (SALI) for a continuous-time dynamical
    system.

    Parameters
    ----------
    u : NDArray[np.float64]
        Initial state vector.
    parameters : NDArray[np.float64]
        System parameters passed to the equations of motion and Jacobian.
    total_time : np.float64
        Final integration time.
    equations_of_motion : flow_t
        Continuous-time vector field with signature
        `(time, u, parameters) -> dudt`.
    jacobian : flow_jacobian_t
        Jacobian of the vector field with signature
        `(time, u, parameters) -> J`.
    transient_time : np.float64 | None, optional
        Initial integration time discarded before SALI accumulation.
    time_step : np.float64, optional
        Initial integration step size.
    atol : np.float64, optional
        Absolute tolerance used by adaptive integrators.
    rtol : np.float64, optional
        Relative tolerance used by adaptive integrators.
    integrator : callable, optional
        Step method with unified signature returning
        `(u_next, time_next, time_step_next, accept)`.
    return_history : bool, optional
        If True, return the time evolution of SALI.
    seed : int, optional
        Random seed used to initialize the deviation vectors.
    threshold : np.float64, optional
        Early stopping threshold. Integration stops when SALI falls below this
        value.

    Returns
    -------
    NDArray[np.float64]
        - If `return_history=False`, returns an array of shape `(1, 2)`
          containing `[time, SALI]`.
        - If `return_history=True`, returns an array of shape `(n_samples, 2)`
          whose first column contains time and whose second column contains SALI.

    Notes
    -----
    SALI is computed from two normalized deviation vectors `v_1` and `v_2` as

    - `SALI = min(||v_1 + v_2||, ||v_1 - v_2||)`

    Chaotic trajectories typically drive SALI rapidly toward zero, whereas
    regular trajectories keep it away from zero.
    """
    neq = len(u)
    ndv = 2
    nt = neq + neq * ndv

    u = u.copy()

    if transient_time is not None:
        u = evolve_system(
            u=u,
            parameters=parameters,
            total_time=transient_time,
            equations_of_motion=equations_of_motion,
            time_step=time_step,
            atol=atol,
            rtol=rtol,
            integrator=integrator,
        )
        time = transient_time
    else:
        time = np.float64(0.0)

    uv = np.zeros(nt, dtype=np.float64)
    uv[:neq] = u.copy()

    np.random.seed(seed)
    uv[neq:] = -1.0 + 2.0 * np.random.rand(nt - neq)
    v = uv[neq:].reshape(neq, ndv)
    v, _ = qr(v)
    uv[neq:] = v.reshape(neq * ndv)

    history = []
    sali = np.float64(0.0)

    while time < total_time:
        if time + time_step > total_time:
            time_step = total_time - time

        uv, time, time_step = step(
            time=time,
            u=uv,
            parameters=parameters,
            equations_of_motion=equations_of_motion,
            jacobian=jacobian,
            time_step=time_step,
            atol=atol,
            rtol=rtol,
            integrator=integrator,
            number_of_deviation_vectors=ndv,
        )

        v = uv[neq:].reshape(neq, ndv)
        v[:, 0] /= np.linalg.norm(v[:, 0])
        v[:, 1] /= np.linalg.norm(v[:, 1])

        pai = np.linalg.norm(v[:, 0] + v[:, 1])
        aai = np.linalg.norm(v[:, 0] - v[:, 1])
        sali = min(pai, aai)

        if return_history:
            history.append([time, sali])

        if sali <= threshold:
            break

        uv[neq:] = v.reshape(neq * ndv)

    if return_history:
        return np.asarray(history, dtype=np.float64)

    result = np.empty((1, 2), dtype=np.float64)
    result[0, 0] = time
    result[0, 1] = sali
    return result
