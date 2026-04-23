# ldi.py

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

import numpy as np
from numba import njit
from numpy.typing import NDArray
from typing import Tuple
from pynamicalsys.common.types import int_t, numeric_t, map_t, jacobian_t
from pynamicalsys.common.linalg import qr


@njit
def ldi_k(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: int_t,
    mapping: map_t,
    jacobian: jacobian_t,
    k: int,
    sample_times: NDArray[np.integer] | None = None,
    return_history: bool = False,
    tol: numeric_t = 1e-16,
    transient_time: int_t | None = None,
    seed: int = 13,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute the Linear Dependence Index (LDI_k) for a discrete dynamical system.

    LDI_k measures the contraction of a set of `k` deviation vectors evolved in
    tangent space. It is computed as the product of the singular values of the
    matrix formed by the evolved deviation vectors.

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
        Array of iteration times at which LDI_k is recorded when
        `return_history=True`.
    return_history : bool, optional
        If True, return LDI_k evaluated at `sample_times`. Otherwise, return
        only the final LDI_k value. Default is False.
    tol : numeric_t, optional
        Early stopping threshold. If `LDI_k < tol`, the computation is
        interrupted. Default is `1e-16`.
    transient_time : int_t | None, optional
        Number of initial iterations to discard as transient before starting
        the LDI_k computation. If None, no transient is discarded.
    seed : int, optional
        Seed used to initialize the deviation vectors. Default is 13.

    Returns
    -------
    Tuple[NDArray[np.float64], NDArray[np.float64]]
        - If `return_history=False`, returns:
          - a 1D array with shape `(1,)` containing the final LDI_k value
          - the final state of shape `(neq,)`
        - If `return_history=True`, returns:
          - a 1D array with shape `(len(sample_times),)` containing LDI_k
            evaluated at the requested sampling times
          - the final state of shape `(neq,)`

    Notes
    -----
    - A set of `k` initially orthonormal deviation vectors is evolved with the
    Jacobian and renormalized at each step. LDI_k is computed as the product
    of the singular values of the deviation-vector matrix.

    - The computation is terminated early if `LDI_k < tol`.

    - This function is compiled with Numba via `@njit`.
    """

    np.random.seed(seed)  # For reproducibility
    num_samples = len(sample_times) if sample_times is not None else 0
    neq = len(u)

    # Generate random orthonormal deviation vectors
    v = np.random.rand(neq, k)
    v, _ = qr(v)

    sample_size = total_time - (transient_time if transient_time is not None else 0)
    if transient_time is not None:
        for _ in range(transient_time):
            u = mapping(u, parameters)

    history = np.empty(0, dtype=np.float64)
    if return_history and sample_times is not None:
        history = np.zeros(num_samples, dtype=np.float64)

    ldi_val = np.sqrt(k)
    sample_idx = 0
    for n in range(sample_size):
        u = mapping(u, parameters)
        J = np.ascontiguousarray(jacobian(u, parameters, mapping))

        for i in range(k):
            v[:, i] = J @ np.ascontiguousarray(v[:, i])
            v[:, i] = v[:, i] / np.linalg.norm(v[:, i])

        _, S, _ = np.linalg.svd(v, full_matrices=False)
        ldi_val = np.exp(np.sum(np.log(S)))
        if (
            return_history
            and sample_times is not None
            and sample_idx < num_samples
            and n + 1 == sample_times[sample_idx]
        ):
            history[sample_idx] = ldi_val
            sample_idx += 1

        if ldi_val < tol:
            break

    if return_history:
        return history, u

    return np.array([ldi_val]), u
