# manifolds.py

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

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from pynamicalsys.common.types import map_t, jacobian_t, numeric_t
from pynamicalsys.discrete_time.stability import classify_stability
from pynamicalsys.discrete_time.trajectory import ensemble_trajectories


def calculate_manifolds(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    forward_mapping: map_t,
    backward_mapping: map_t,
    jacobian: jacobian_t,
    period: int,
    delta: numeric_t = 1e-4,
    n_points: int | tuple[int, int] = 100,
    iter_time: int | tuple[int, int] = 100,
    stability: Literal["stable", "unstable"] = "unstable",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute the two branches of the stable or unstable manifold of a 2D saddle
    periodic orbit.

    Parameters
    ----------
    u : NDArray[np.float64]
        Initial condition on the periodic orbit, with shape `(2,)`.
    parameters : NDArray[np.float64]
        System parameters.
    forward_mapping : map_t
        Forward map.
    backward_mapping : map_t
        Backward map.
    jacobian : jacobian_t
        Jacobian of the forward map.
    period : int_t
        Period of the orbit.
    delta : numeric_t, optional
        Initial displacement magnitude used to seed the manifold branches.
    n_points : int | tuple[int, int], optional
        Number of seed points on each branch. If an integer is given, the same
        value is used for both branches.
    iter_time : int_t | tuple[int_t, int_t], optional
        Number of iterations used to evolve each branch. If an integer is given,
        the same value is used for both branches.
    stability : {"stable", "unstable"}, optional
        Which invariant manifold to compute.

    Returns
    -------
    tuple[NDArray[np.float64], NDArray[np.float64]]
        Two arrays containing the `+` and `-` manifold branches. Each array has
        shape `(n_points_branch * iter_time_branch, 2)`.

    Raises
    ------
    ValueError
        - If `u` is not 2-dimensional.
        - If the orbit is not classified as a saddle.
        - If `n_points` or `iter_time` are invalid.
        - If `delta` is not positive.
        - If `stability` is invalid.

    Notes
    -----
    The manifold is seeded by small displacements along the corresponding
    eigendirection of the monodromy matrix. The `"unstable"` manifold is evolved
    with the forward map, and the `"stable"` manifold is evolved with the
    backward map.
    """
    if u.shape != (2,):
        raise ValueError("calculate_manifolds is only implemented for 2D systems")

    if period < 1:
        raise ValueError("period must be positive")

    if delta <= 0:
        raise ValueError("delta must be positive")

    if isinstance(n_points, int):
        n_points_tuple = (n_points, n_points)
    else:
        if len(n_points) != 2:
            raise ValueError("n_points must be an int or a tuple of length 2")
        n_points_tuple = (int(n_points[0]), int(n_points[1]))

    if n_points_tuple[0] < 1 or n_points_tuple[1] < 1:
        raise ValueError("all n_points values must be positive")

    if isinstance(iter_time, int):
        iter_time_tuple = (iter_time, iter_time)
    else:
        if len(iter_time) != 2:
            raise ValueError("iter_time must be an int or a tuple of length 2")
        iter_time_tuple = (int(iter_time[0]), int(iter_time[1]))

    if iter_time_tuple[0] < 1 or iter_time_tuple[1] < 1:
        raise ValueError("all iter_time values must be positive")

    if stability not in ("stable", "unstable"):
        raise ValueError("stability must be either 'stable' or 'unstable'")

    stability_info = classify_stability(
        u=u,
        parameters=parameters,
        mapping=forward_mapping,
        jacobian=jacobian,
        period=period,
    )

    if stability_info["classification"] != "saddle":
        raise ValueError("calculate_manifolds requires a saddle periodic orbit")

    eigenvectors = stability_info["eigenvectors"]
    if not isinstance(eigenvectors, np.ndarray):
        raise ValueError("invalid eigenvector data returned by classify_stability")

    eigenvectors = np.asarray(eigenvectors, dtype=np.complex128)

    vu = eigenvectors[:, 0]
    vs = eigenvectors[:, 1]

    if stability == "unstable":
        direction = np.asarray(vu.real, dtype=np.float64)
        mapping = forward_mapping
    else:
        direction = np.asarray(vs.real, dtype=np.float64)
        mapping = backward_mapping

    norm = np.linalg.norm(direction)
    if norm == 0.0:
        raise ValueError("selected eigendirection has zero norm")

    direction /= norm

    def _calculate_branch(sign: int, branch_index: int) -> NDArray[np.float64]:
        seeds = np.empty((n_points_tuple[branch_index], 2), dtype=np.float64)

        scales = np.linspace(0.0, delta, n_points_tuple[branch_index])
        for i in range(n_points_tuple[branch_index]):
            seeds[i] = u + sign * scales[i] * direction

        return ensemble_trajectories(
            u=seeds,
            parameters=parameters,
            total_time=iter_time_tuple[branch_index],
            mapping=mapping,
        )

    branch_plus = _calculate_branch(sign=1, branch_index=0)
    branch_minus = _calculate_branch(sign=-1, branch_index=1)

    return branch_plus, branch_minus
