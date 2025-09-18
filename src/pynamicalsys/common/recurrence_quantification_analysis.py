# recurrence_quantification_analysis.py

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

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numba import njit
from numpy.typing import NDArray


@dataclass
class RTEConfig:
    """
    Configuration class for Recurrence Time Entropy (RTE) analysis.

    Attributes
    ----------
    metric : {'supremum', 'euclidean', 'manhattan'}, default='supremum'
        Distance metric used for phase space reconstruction.
    std_metric : {'supremum', 'euclidean', 'manhattan'}, default='supremum'
        Distance metric used for standard deviation calculation.
    lmin : int, default=1
        Minimum line length to consider in recurrence quantification.
    threshold : float, default=0.1
        Recurrence threshold (relative to data range).
    threshold_std : bool, default=True
        Whether to scale threshold by data standard deviation.
    return_final_state : bool, default=False
        Whether to return the final system state in results.
    return_recmat : bool, default=False
        Whether to return the recurrence matrix.
    return_p : bool, default=False
        Whether to return white vertical line length distribution.

    Notes
    -----
    - The 'supremum' metric (default) is computationally efficient and often sufficient for RTE.
    - Typical threshold values range from 0.05 to 0.3 depending on data noise levels.
    - Set lmin=2 to exclude single-point recurrences from analysis.
    """

    metric: Literal["supremum", "euclidean", "manhattan"] = "supremum"
    std_metric: Literal["supremum", "euclidean", "manhattan"] = "supremum"
    lmin: int = 1
    threshold: float = 0.1
    threshold_std: bool = True
    return_final_state: bool = False
    return_recmat: bool = False
    return_p: bool = False

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.lmin < 1:
            raise ValueError("lmin must be ≥ 1")

        if not isinstance(self.lmin, int):
            raise TypeError("lmin must be an integer")

        if not isinstance(self.threshold, float):
            raise TypeError("threshold must be a float")

        if not 0 < self.threshold < 1:
            raise ValueError("threshold must be in (0, 1)")

        if not isinstance(self.std_metric, str):
            raise TypeError("std_metric must be a string")

        if not isinstance(self.metric, str):
            raise TypeError("metric must be a string")

        if self.std_metric not in {"supremum", "euclidean", "manhattan"}:
            raise ValueError(
                "std_metric must be 'supremum', 'euclidean' or 'manhattan'"
            )

        if self.metric not in {"supremum", "euclidean", "manhattan"}:
            raise ValueError("metric must be 'supremum', 'euclidean' or 'manhattan'")


@njit
def _recurrence_matrix(
    arr: NDArray[np.float64], threshold: float, metric_id: int
) -> NDArray[np.uint8]:
    """
    Compute the binary recurrence matrix of a time series using a specified norm.

    Parameters
    ----------
    arr : NDarray of shape (N, d)
        The input time series or phase-space trajectory, where N is the number of time points
        and d is the embedding dimension (or feature dimension).

    threshold : float
        Distance threshold for determining recurrence. A recurrence is detected
        when the distance between two points is less than this threshold.

    metric_id : int
        Identifier for the norm to be used:
            - 0: Supremum (infinity) norm
            - 1: Euclidean (L2) norm
            - 2: Manhattan (L1) norm

    Returns
    -------
    recmat : NDarray of shape (N, N), dtype=np.uint8
        Binary recurrence matrix where 1 indicates recurrence and 0 indicates no recurrence.
    """
    N, d = arr.shape
    recmat = np.zeros((N, N), dtype=np.uint8)

    for i in range(N):
        for j in range(i, N):
            if metric_id == 0:  # Supremum norm
                max_diff = 0.0
                for k in range(d):
                    diff = abs(arr[i, k] - arr[j, k])
                    if diff > max_diff:
                        max_diff = diff
                dist = max_diff
            elif metric_id == 1:  # Manhattan norm
                sum_abs = 0.0
                for k in range(d):
                    sum_abs += abs(arr[i, k] - arr[j, k])
                dist = sum_abs
            elif metric_id == 2:  # Euclidean norm
                sq_sum = 0.0
                for k in range(d):
                    diff = arr[i, k] - arr[j, k]
                    sq_sum += diff * diff
                dist = np.sqrt(sq_sum)
            else:
                # Fallback: shouldn't happen
                dist = 0.0

            if dist < threshold:
                recmat[i, j] = 1
                recmat[j, i] = 1  # enforce symmetry

    return recmat


def recurrence_matrix(
    arr: NDArray[np.float64], threshold: float, metric: str = "supremum"
) -> NDArray[np.uint8]:
    """
    Compute the recurrence matrix of a univariate or multivariate time series.

    Parameters
    ----------
    u : NDArray
        Time series data. Can be 1D (shape: (N,)) or 2D (shape: (N, d)).
        If 1D, the array is reshaped to (N, 1) automatically.

    threshold : float
        Distance threshold for recurrence. A recurrence is detected when the
        distance between two points is less than this threshold.

    metric : str, optional, default="supremum"
        Distance metric to use. Supported values are:
            - "supremum"  : infinity norm (L-infinity)
            - "euclidean" : L2 norm
            - "manhattan" : L1 norm

    Returns
    -------
    recmat : NDArray of shape (N, N), dtype=np.uint8
        Binary recurrence matrix indicating whether each pair of points
        are within the threshold distance.

    Raises
    ------
    ValueError
        If the specified metric is invalid.
    """
    metrics = {"supremum": 0, "euclidean": 1, "manhattan": 2}
    if metric not in metrics:
        raise ValueError("Metric must be 'supremum', 'euclidean', or 'manhattan'")
    metric_id = metrics[metric]

    if threshold <= 0:
        print(threshold)
        raise ValueError("Threshold must be positive")

    if not isinstance(arr, np.ndarray):
        raise TypeError("Input 'arr' must be a NumPy array")
    if arr.ndim not in (1, 2):
        raise ValueError("Input 'arr' must be 1D or 2D array")

    arr = np.atleast_2d(arr).astype(np.float64)
    if arr.shape[0] == 1:
        arr = arr.T

    return _recurrence_matrix(arr, threshold, metric_id)


@njit
def white_vertline_distr(
    recmat: NDArray[np.uint8], wmin: int = 1
) -> NDArray[np.float64]:
    """
    Calculate the distribution of white vertical line lengths in a binary recurrence matrix.

    This function counts occurrences of consecutive vertical white (0) pixels, excluding
    lines touching the matrix borders, as defined in recurrence quantification analysis.

    Parameters
    ----------
    recmat : NDArray[np.uint8]
        A 2D binary matrix (0s and 1s) representing a recurrence matrix.
        Expected shape: (N, N) where N is the matrix dimension.

    Returns
    -------
    NDArray[np.float64]
        Array where index represents line length and value represents count.
        (Note: Index 0 is unused since minimum line length is 1)

    Raises
    ------
    ValueError
        If input is not 2D or not square.

    Notes
    -----
    - Border lines (touching matrix edges) are excluded from counts [1]
    - Complexity: O(N^2) for N x N matrix
    - Optimized with Numba's @njit decorator for performance

    References
    ----------
    [1] K. H. Kraemer & N. Marwan, "Border effect corrections for diagonal line based
        recurrence quantification analysis measures", Physics Letters A 383, 125977 (2019)
    """
    # Input validation
    if recmat.ndim != 2 or recmat.shape[0] != recmat.shape[1]:
        raise ValueError("Input must be a square 2D array")

    N = recmat.shape[0]
    P = np.zeros(N + 1)  # Index 0 unused, max possible length is N

    for i in range(N):
        current_length = 0
        border_flag = False  # Tracks if we're in a border region

        for j in range(N):
            if recmat[i, j] == 0:
                if border_flag:  # Only count after first black pixel
                    current_length += 1
            else:
                border_flag = True  # Mark that we've passed the border
                if current_length > 0 and j < N - 1:
                    P[current_length] += 1
                    current_length = 0

    P = P[wmin:]

    return P
