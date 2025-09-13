from typing import Optional, Tuple, Union, Callable
from numpy.typing import NDArray
import numpy as np
from numba import njit, prange

from pynamicalsys.common.utils import fit_poly


@njit(parallel=True)
def hurst_exponent(
    time_series: NDArray[np.float64],
    wmin: int = 2,
) -> NDArray[np.float64]:

    time_series = time_series.copy()

    sample_size, neq = time_series.shape

    H = np.zeros(neq)

    ells = np.arange(wmin, sample_size // 2)
    log_ells = np.log(ells)
    RS = np.empty((ells.shape[0], neq))

    for j in prange(neq):
        series = time_series[:, j]

        # Precompute cumulative sums and cumulative sums of squares
        cum_sum = np.zeros(sample_size)
        cum_sum_sq = np.zeros(sample_size)
        cum_sum[0] = series[0]
        cum_sum_sq[0] = series[0] ** 2
        for t in range(1, sample_size):
            cum_sum[t] = cum_sum[t - 1] + series[t]
            cum_sum_sq[t] = cum_sum_sq[t - 1] + series[t] ** 2

        for i, ell in enumerate(ells):
            num_blocks = sample_size // ell
            R_over_S = np.zeros(num_blocks)

            for block in range(num_blocks):
                start = block * ell
                end = start + ell

                # Mean using cumulative sums
                block_sum = cum_sum[end - 1] - (cum_sum[start - 1] if start > 0 else 0)
                block_mean = block_sum / ell

                # Variance using cumulative sums of squares
                block_sum_sq = cum_sum_sq[end - 1] - (
                    cum_sum_sq[start - 1] if start > 0 else 0
                )
                var = block_sum_sq / ell - block_mean**2
                S = np.sqrt(var) if var > 0 else 0

                # Cumulative sum of mean-adjusted series for range
                max_Z = 0.0
                min_Z = 0.0
                cumsum = 0.0
                for k in range(start, end):
                    cumsum += series[k] - block_mean
                    if cumsum > max_Z:
                        max_Z = cumsum
                    if cumsum < min_Z:
                        min_Z = cumsum
                R = max_Z - min_Z

                R_over_S[block] = R / S if S > 0 else 0.0

            positive_mask = R_over_S > 0
            RS[i, j] = (
                np.mean(R_over_S[positive_mask]) if np.any(positive_mask) else 0.0
            )

        # Linear regression in log-log space
        positive_inds = np.where(RS[:, j] > 0)[0]
        if positive_inds.size == 0:
            H[j] = 0.0
        else:
            x_fit = log_ells[positive_inds]
            y_fit = np.log(RS[positive_inds, j])
            fitting = fit_poly(x_fit, y_fit, 1)
            H[j] = fitting[0]

    return H
