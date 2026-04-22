# birkhoff.py

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
from numpy.typing import NDArray

from pynamicalsys.common.types import int_t, map_t, observable_t
from pynamicalsys.discrete_time.trajectory_analysis import generate_trajectory


def dig(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: int_t,
    mapping: map_t,
    func: observable_t,
    transient_time: int_t | None = None,
) -> float:
    """
    Compute the `dig` indicator from weighted Birkhoff averages.

    The `dig` indicator measures the agreement between two weighted Birkhoff
    averages computed over consecutive halves of the same trajectory. It is
    defined as

        `dig = -log10(|WB_0 - WB_1|)`

    where `WB_0` and `WB_1` are weighted Birkhoff averages of the observable
    over the first and second halves of the sampled trajectory, respectively.

    Larger values of `dig` indicate better convergence of the weighted Birkhoff
    average and are typically associated with more regular dynamics. Smaller
    values indicate poorer convergence.

    Parameters
    ----------
    u : NDArray[np.float64]
        Initial condition of shape `(d,)`, where `d` is the system dimension.
    parameters : NDArray[np.float64]
        Parameter array passed to `mapping`.
    total_time : int_t
        Total number of iterations used in the computation.
    mapping : map_t
        Function defining the time evolution of the system.
    func : observable_t
        Observable evaluated along the trajectory. It must accept the trajectory
        array returned by `generate_trajectory` and return values compatible with
        weighted summation.
    transient_time : int_t | None, optional
        Number of initial iterations discarded before computing the weighted
        Birkhoff averages. If None, no transient is discarded.

    Returns
    -------
    float
        Value of the `dig` indicator.

    Raises
    ------
    ValueError
        If the effective sample size after transient removal is too small to be
        split into two nontrivial halves.

    Notes
    -----
    The computation proceeds as follows:

    1. Discard `transient_time` iterations, if requested.
    2. Split the remaining sample into two halves of length `N`.
    3. Compute the weighted Birkhoff average on each half.
    4. Return `-log10(|WB_0 - WB_1|)`.

    The weighted Birkhoff average uses weights

        `w_n propto exp(-1 / (t_n (1 - t_n)))`

    with `t_n = n / N`, for `n = 1, ..., N - 1`.

    This function assumes that input validation is handled by the wrapper.
    """

    u = u.copy()

    sample_size = total_time - (transient_time if transient_time is not None else 0)

    if transient_time is not None:
        for _ in range(transient_time):
            u = mapping(u, parameters)

    N = sample_size // 2
    if N < 2:
        raise ValueError("Effective sample size too small after transient removal")

    t = np.arange(1, N, dtype=np.float64) / N
    weights = np.exp(-1.0 / (t * (1.0 - t)))
    weights /= weights.sum()

    time_series = generate_trajectory(u, parameters, N, mapping)
    wb0 = np.sum(weights * func(time_series[:-1, :]))

    u = time_series[-1, :].copy()
    time_series = generate_trajectory(u, parameters, N, mapping)
    wb1 = np.sum(weights * func(time_series[:-1, :]))

    return float(-np.log10(abs(wb0 - wb1)))
