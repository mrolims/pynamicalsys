# gali.py

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

from numba import njit
from numpy.typing import NDArray
import numpy as np

from pynamicalsys.common.types import flow_t, flow_jacobian_t
from pynamicalsys.common.utils import qr
from pynamicalsys.continuous_time.step_methods import rk4_step_wrapped
from pynamicalsys.continuous_time.step import evolve_system, step


@njit
def gali_k(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: np.float64,
    equations_of_motion: flow_t,
    jacobian: flow_jacobian_t,
    number_deviation_vectors: int,
    transient_time: np.float64 | None = None,
    time_step: np.float64 = np.float64(0.01),
    atol: np.float64 = np.float64(1e-6),
    rtol: np.float64 = np.float64(1e-3),
    integrator=rk4_step_wrapped,
    return_history: bool = False,
    method: str = "QR",
    seed: int = 13,
    threshold: np.float64 = np.float64(1e-16),
) -> NDArray[np.float64]:
    """
    Compute the Generalized Alignment Index of order `k` for a continuous-time
    dynamical system.

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
    number_deviation_vectors : int
        Number of deviation vectors used in the computation.
    transient_time : np.float64 | None, optional
        Initial integration time discarded before GALI accumulation.
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
        If True, return the time evolution of GALI.
    method : str, optional
        Method used to compute GALI. Supported options are:

        - `"DET"`:
          Compute GALI from the Gram matrix `G = V^T V`, where `V` is the
          deviation-vector matrix. In this case,

          `GALI_k = sqrt(det(G))`.

        - `"QR"`:
          Compute GALI from the diagonal of the triangular factor returned by the
          custom QR routine `qr(V)`. If `V = Q R`, then

          `GALI_k = prod_i |R_ii|`.

        - `"QR_HH"`:
          Compute GALI from the diagonal of the triangular factor returned by
          `numpy.linalg.qr(V)`, again through

          `GALI_k = prod_i |R_ii|`.

        Default is `"QR"`.

    seed : int, optional
        Random seed used to initialize the deviation vectors.
    threshold : np.float64, optional
        Early stopping threshold. Integration stops when GALI falls below this
        value.

    Returns
    -------
    NDArray[np.float64]
        - If `return_history=False`, returns an array of shape `(1, 2)`
          containing `[time, GALI_k]`.
        - If `return_history=True`, returns an array of shape `(n_samples, 2)`
          whose first column contains time and whose second column contains
          `GALI_k`.

    Notes
    -----
    Let `V` be the matrix whose columns are the normalized deviation vectors.
    GALI measures the `k`-dimensional volume spanned by these columns and can be
    written as

    `GALI_k = sqrt(det(V^T V))`

    or, if `V = Q R` is a QR decomposition, as

    `GALI_k = |det(R)| = prod_i |R_ii|`.

    For chaotic trajectories, GALI typically decays rapidly toward zero due to
    the alignment of deviation vectors. For regular trajectories, the decay is
    slower or GALI may remain bounded away from zero, depending on the dimension
    of the underlying invariant object and on `k`.
    """
    neq = len(u)
    ndv = number_deviation_vectors
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
    gali = np.float64(0.0)

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

        for i in range(ndv):
            v[:, i] /= np.linalg.norm(v[:, i])

        if method == "DET":
            G = v.T @ v
            gali = np.sqrt(np.linalg.det(G))
        elif method == "QR":
            _, R = qr(v)
            gali = np.exp(np.sum(np.log(np.abs(np.diag(R)))))
        elif method == "QR_HH":
            _, R = np.linalg.qr(v)
            gali = np.exp(np.sum(np.log(np.abs(np.diag(R)))))
        else:
            raise ValueError("method must be 'DET', 'QR', or 'QR_HH'")

        if return_history:
            history.append([time, gali])

        if gali <= threshold:
            break

        uv[neq:] = v.reshape(neq * ndv)

    if return_history:
        return np.asarray(history, dtype=np.float64)

    result = np.empty((1, 2), dtype=np.float64)
    result[0, 0] = time
    result[0, 1] = gali
    return result
