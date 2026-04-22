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

import numpy as np
from numba import njit
from numpy.typing import NDArray
from typing import Tuple
from pynamicalsys.common.types import int_t, numeric_t, map_t, jacobian_t
from pynamicalsys.common.utils import qr


@njit
def gali_k(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: int_t,
    mapping: map_t,
    jacobian: jacobian_t,
    k: int,
    sample_times: NDArray[np.integer] | None = None,
    method: str = "QR",
    return_history: bool = False,
    tol: numeric_t = 1e-16,
    transient_time: int_t | None = None,
    seed: int = 13,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute the Generalized Alignment Index (GALI_k) for a discrete dynamical system.

    GALI_k quantifies the degree of alignment of `k` deviation vectors evolved in
    tangent space. It measures the contraction of the `k`-dimensional volume
    spanned by these vectors and is widely used to distinguish regular and chaotic
    motion.

    Parameters
    ----------
    u : NDArray[np.float64]
        Initial condition of shape `(neq,)`, where `neq` is the system dimension.
    parameters : NDArray[np.float64]
        Parameter array passed to `mapping` and `jacobian`.
    total_time : int_t
        Total number of iterations used in the computation.
    mapping : map_t
        Function defining the time evolution of the system.
    jacobian : jacobian_t
        Function returning the Jacobian matrix of the map.
    k : int
        Number of deviation vectors used in the computation.
    sample_times : NDArray[np.integer] | None, optional
        Array of iteration times at which GALI_k is recorded when
        `return_history=True`.
    method : str, optional
        Method used to compute GALI_k. Supported options are:

        - `"DET"`:
          Compute GALI_k from the Gram matrix `G = V^T V`, where `V` is the
          deviation-vector matrix. In this case,

          `GALI_k = sqrt(det(G))`.

          This approach is direct and compact, but it is usually less numerically
          stable when the deviation vectors become strongly aligned.

        - `"QR"`:
          Compute GALI_k from a reduced QR decomposition using the custom
          modified Gram-Schmidt routine `qr(V) = Q R`. Since the `k`-volume
          spanned by the columns of `V` is given by `|det(R)|`, GALI_k is
          computed as

          `GALI_k = prod_i |R_ii|`.

          This avoids forming the Gram matrix explicitly and is often more stable
          than the determinant-based approach.

        - `"QR_HH"`:
          Compute GALI_k from a QR decomposition based on Householder reflections
          using `numpy.linalg.qr`. As in the `"QR"` case, GALI_k is obtained from
          the diagonal of the triangular factor `R` through

          `GALI_k = prod_i |R_ii|`.

          This is typically the most numerically stable option among the three. It
          also is the slowest.

        Default is `"QR"`.

    return_history : bool, optional
        If True, return GALI_k evaluated at `sample_times`. Otherwise, return
        only the final GALI_k value. Default is False.
    tol : numeric_t, optional
        Early stopping threshold. If `GALI_k < tol`, the computation is
        interrupted. Default is `1e-16`.
    transient_time : int_t | None, optional
        Number of initial iterations to discard as transient before starting
        the GALI_k computation. If None, no transient is discarded.
    seed : int, optional
        Seed used to initialize the deviation vectors. Default is 13.

    Returns
    -------
    Tuple[NDArray[np.float64], NDArray[np.float64]]
        - If `return_history=False`, returns:
          - a 1D array with shape `(1,)` containing the final GALI_k value
          - the final state of shape `(neq,)`
        - If `return_history=True`, returns:
          - a 1D array with shape `(len(sample_times),)` containing GALI_k
            evaluated at the requested sampling times
          - the final state of shape `(neq,)`

    Notes
    -----
    A set of `k` initially orthonormal deviation vectors is evolved with the
    Jacobian and renormalized at each step. Let `V` be the matrix whose columns
    are the evolved deviation vectors. GALI_k measures the `k`-dimensional volume
    spanned by these columns and can be written equivalently as

    `GALI_k = sqrt(det(V^T V))`

    or, if `V = Q R` is a QR decomposition, as

    `GALI_k = |det(R)| = prod_i |R_ii|`.

    For chaotic trajectories, GALI_k typically decays rapidly toward zero due to
    the progressive alignment of deviation vectors. For regular trajectories, the
    decay is slower or GALI_k may remain bounded away from zero, depending on the
    dimension of the underlying invariant object and on `k`.

    The computation is terminated early if `GALI_k < tol`.

    This function is compiled with Numba via `@njit`.
    """
    np.random.seed(seed)  # For reproducibility
    num_samples = len(sample_times) if sample_times is not None else 0
    neq = len(u)

    # Generate random orthonormal deviation vectors
    v = np.random.rand(neq, k)
    v, _ = np.linalg.qr(v)

    sample_size = total_time - (transient_time if transient_time is not None else 0)
    if transient_time is not None:
        for _ in range(transient_time):
            u = mapping(u, parameters)

    history = np.empty(0, dtype=np.float64)
    if return_history and sample_times is not None:
        history = np.zeros(num_samples, dtype=np.float64)

    gali_val = np.sqrt(k)
    sample_idx = 0
    for n in range(sample_size):
        u = mapping(u, parameters)
        J = np.ascontiguousarray(jacobian(u, parameters, mapping))

        for i in range(k):
            v[:, i] = J @ np.ascontiguousarray(v[:, i])
            v[:, i] = v[:, i] / np.linalg.norm(v[:, i])

        v = np.ascontiguousarray(v)
        if method == "DET":
            G = v.T @ v
            gali_val = np.sqrt(np.linalg.det(G))
        elif method == "QR":
            _, R = qr(v)
            gali_val = np.exp(np.sum(np.log(np.abs(np.diag(R)))))
        elif method == "QR_HH":
            _, R = np.linalg.qr(v)
            gali_val = np.exp(np.sum(np.log(np.abs(np.diag(R)))))
        if (
            return_history
            and sample_times is not None
            and sample_idx < num_samples
            and n + 1 == sample_times[sample_idx]
        ):
            history[sample_idx] = gali_val
            sample_idx += 1

        if gali_val < tol:
            break

    if return_history:
        return history, u

    return np.array([gali_val]), u
