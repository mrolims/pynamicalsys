# sali.py

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


from numba import njit
import numpy as np
from numpy.typing import NDArray
from typing import Optional, Tuple
from pynamicalsys.common.types import int_t, numeric_t, map_t, jacobian_t
from pynamicalsys.common.utils import qr


@njit
def sali(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: int_t,
    mapping: map_t,
    jacobian: jacobian_t,
    sample_times: NDArray[np.integer] | None,
    return_history: bool = False,
    tol: numeric_t = 1e-16,
    transient_time: Optional[int_t] = None,
    seed: int_t = 1312,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute the Smallest Alignment Index (SALI) for a discrete dynamical system.

    SALI quantifies the alignment of two deviation vectors evolved in tangent
    space. For chaotic trajectories, SALI typically decays rapidly toward zero,
    whereas for regular trajectories it remains bounded away from zero.

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
    sample_times : Optional[NDArray[np.integer]]
        Array of iteration times at which SALI is recorded when
        `return_history=True`.
    return_history : bool, optional
        If True, return SALI evaluated at `sample_times`. Otherwise, return only
        the final SALI value. Default is False.
    tol : numeric_t, optional
        Tolerance for early stopping. If `SALI < tol`, the computation is
        interrupted. Default is `1e-16`.
    transient_time : Optional[int_t], optional
        Number of initial iterations to discard as transient before starting the
        SALI computation. If None, no transient is discarded.
    seed : int, optional
        Seed used to initialize the deviation vectors. Default is 1312.

    Returns
    -------
    Tuple[NDArray[np.float64], NDArray[np.float64]]
        - If `return_history=False`, returns:
          - a 1D array with shape `(1,)` containing the final SALI value
          - the final state of shape `(neq,)`
        - If `return_history=True`, returns:
          - a 1D array with shape `(len(sample_times),)` containing SALI
            evaluated at the requested sampling times
          - the final state of shape `(neq,)`

    Notes
    -----
    Two initially orthonormal deviation vectors are evolved with the Jacobian and
    renormalized at each step. SALI is defined as

        min(||v_1 + v_2||, ||v_1 - v_2||).

    This function is compiled with Numba via `@njit`.
    """
    np.random.seed(seed)  # For reproducibility
    num_samples = len(sample_times) if sample_times is not None else 0
    neq = len(u)

    # Only need 2 vectors for SALI
    v = np.ascontiguousarray(np.random.rand(neq, 2))
    v, _ = qr(v)

    sample_size = total_time - (transient_time if transient_time is not None else 0)
    if transient_time is not None:
        for _ in range(transient_time):
            u = mapping(u, parameters)

    history = np.empty(0, dtype=np.float64)
    if return_history and sample_times is not None:
        history = np.zeros(num_samples, dtype=np.float64)

    sali_val = np.sqrt(2)
    sample_idx = 0
    for n in range(sample_size):
        u = mapping(u, parameters)
        J = np.ascontiguousarray(jacobian(u, parameters, mapping))

        for i in range(2):
            v[:, i] = J @ np.ascontiguousarray(v[:, i])
            v[:, i] /= np.linalg.norm(v[:, i])

        # Compute SALI
        PAI = np.linalg.norm(v[:, 0] + v[:, 1])
        AAI = np.linalg.norm(v[:, 0] - v[:, 1])
        sali_val = min(PAI, AAI)

        if (
            return_history
            and sample_times is not None
            and sample_idx < num_samples
            and n + 1 == sample_times[sample_idx]
        ):
            history[sample_idx] = sali_val
            sample_idx += 1

        if sali_val < tol:
            break

    if return_history:
        return history, u

    return np.array([sali_val]), u
