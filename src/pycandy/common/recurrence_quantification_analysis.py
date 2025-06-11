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

import numpy as np
from numba import njit
from dataclasses import dataclass
from typing import Literal
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


@njit(cache=True)
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
def white_vertline_distr(recmat: NDArray[np.uint8]) -> NDArray[np.float64]:
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
                if current_length > 0:
                    P[current_length] += 1
                    current_length = 0

        # Handle line continuing to matrix edge
        if current_length > 0 and border_flag:
            P[current_length] += 1

    P = P[1:]  # Exclude unused 0 index

    return P


# def RTE(
#     u: NDArray[np.float64],
#     parameters: NDArray[np.float64],
#     total_time: int,
#     mapping: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
#     transient_time: Optional[int] = None,
#     **kwargs
# ) -> Union[float, Tuple]:
#     """
#     Calculate Recurrence Time Entropy (RTE) for a dynamical system.

#     RTE quantifies the complexity of a system by analyzing the distribution
#     of white vertical lines, i.e., the gap between two diagonal lines.
#     Higher entropy indicates more complex dynamics.

#     Parameters
#     ----------
#     u : NDArray[np.float64]
#         Initial state vector (shape: (neq,))
#     parameters : NDArray[np.float64]
#         System parameters passed to mapping function
#     total_time : int
#         Number of iterations to simulate
#     mapping : Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]
#         System evolution function: u_next = mapping(u, parameters)
#     transient_time : Optional[int], default=None
#         Time to wait before starting RTE calculation.
#     **kwargs
#         Configuration parameters (see RTEConfig)

#     Returns
#     -------
#     Union[float, Tuple]
#         - Base case: RTE value (float)
#         - With optional returns: List containing [RTE, *requested_additional_data]

#     Raises
#     ------
#     ValueError
#         - If invalid metric specified
#         - If trajectory generation fails

#     Notes
#     -----
#     - Implements the method described in [1]
#     - For optimal results:
#         - Use total_time > 1000 for reliable statistics
#         - Typical threshold values: 0.05-0.3
#         - Set lmin=1 to include single-point recurrences

#     References
#     ----------
#     [1] M. R. Sales, M. Mugnaine, J. Szezech, José D., R. L. Viana, I. L. Caldas, N. Marwan, and J. Kurths, Stickiness and recurrence plots: An entropy-based approach, Chaos: An Interdisciplinary Journal of Nonlinear Science 33, 033140 (2023)
#     """

#     u = u.copy()

#     # Configuration handling
#     config = RTEConfig(**kwargs)

#     # Metric setup
#     metric_map = {
#         "supremum": np.inf,
#         "euclidean": 2,
#         "manhattan": 1
#     }

#     try:
#         ord = metric_map[config.std_metric.lower()]
#     except KeyError:
#         raise ValueError(
#             f"Invalid std_metric: {config.std_metric}. Must be {list(metric_map.keys())}")

#     if transient_time is not None:
#         u = iterate_mapping(u, parameters, transient_time, mapping)
#         total_time -= transient_time

#     # Generate trajectory
#     try:
#         time_series = generate_trajectory(u, parameters, total_time, mapping)
#     except Exception as e:
#         raise ValueError(f"Trajectory generation failed: {str(e)}")

#     # Threshold calculation
#     if config.threshold_std:
#         std = np.std(time_series, axis=0)
#         eps = config.threshold * np.linalg.norm(std, ord=ord)
#         if eps <= 0:
#             eps = 0.1
#     else:
#         eps = config.threshold

#     # Recurrence matrix calculation
#     recmat = recurrence_matrix(time_series, float(eps), metric=config.metric)

#     # White line distribution
#     P = white_vertline_distr(recmat)[config.lmin:]
#     P = P[P > 0]  # Remove zeros
#     P /= P.sum()   # Normalize

#     # Entropy calculation
#     rte = -np.sum(P * np.log(P))

#     # Prepare output
#     result = [rte]
#     if config.return_final_state:
#         result.append(time_series[-1])
#     if config.return_recmat:
#         result.append(recmat)
#     if config.return_p:
#         result.append(P)

#     return result[0] if len(result) == 1 else tuple(result)


# def finite_time_RTE(
#     u: NDArray[np.float64],
#     parameters: NDArray[np.float64],
#     total_time: int,
#     finite_time: int,
#     mapping: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
#     return_points: bool = False,
#     **kwargs
# ) -> Union[NDArray[np.float64], Tuple[NDArray[np.float64], NDArray[np.float64]]]:
#     # Validate window size
#     if finite_time > total_time:
#         raise ValueError(
#             f"finite_time ({finite_time}) exceeds available samples ({total_time})")

#     num_windows = total_time // finite_time
#     RTE_values = np.zeros(num_windows)
#     phase_space_points = np.zeros((num_windows, u.shape[0]))

#     for i in range(num_windows):
#         result = RTE(
#             u,
#             parameters,
#             finite_time,
#             mapping,
#             return_final_state=True,
#             **kwargs
#         )
#         if isinstance(result, tuple):
#             RTE_values[i], u_new = result
#             phase_space_points[i] = u
#             u = u_new.copy()

#     if return_points:
#         return RTE_values, phase_space_points
#     else:
#         return RTE_values
