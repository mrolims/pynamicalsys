# recurrence_quantification_analysis.py

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

from dataclasses import dataclass
from typing import Literal, Union, Callable, Optional

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

import warnings


PairwiseMetric = Union[
    Literal["supremum", "euclidean", "manhattan"],
    Callable[[NDArray[np.float64], NDArray[np.float64]], float],
]

StdMetric = Union[
    Literal["supremum", "euclidean", "manhattan"],
    Callable[[NDArray[np.float64]], float],
]


@dataclass
class RTEConfig:
    metric: PairwiseMetric = "supremum"
    std_metric: StdMetric = "supremum"

    threshold: float = 0.1

    # New API
    threshold_mode: Optional[Literal["direct", "std", "rr"]] = None

    # Deprecated legacy API
    threshold_std: bool = True

    lmin: int = 1
    return_final_state: bool = False
    return_recmat: bool = False
    return_p: bool = False

    def __post_init__(self):
        allowed_modes = {"direct", "std", "rr"}
        allowed_named_metrics = {"supremum", "euclidean", "manhattan"}

        if self.threshold_mode is not None and self.threshold_mode not in allowed_modes:
            raise ValueError("threshold_mode must be 'direct', 'std', or 'rr'.")

        if self.threshold_mode is None:
            if self.threshold_std:
                warnings.warn(
                    "`threshold_std` is deprecated and will be removed in a future "
                    "release. Use `threshold_mode='std'` instead.",
                    FutureWarning,
                    stacklevel=2,
                )
                self.threshold_mode = "std"
            else:
                warnings.warn(
                    "`threshold_std` is deprecated and will be removed in a future "
                    "release. Use `threshold_mode='direct'` or `threshold_mode='rr'` instead.",
                    FutureWarning,
                    stacklevel=2,
                )
                self.threshold_mode = "direct"
        else:
            if self.threshold_std is not True:
                warnings.warn(
                    "`threshold_std` is deprecated and ignored when `threshold_mode` "
                    "is explicitly provided.",
                    FutureWarning,
                    stacklevel=2,
                )

        if not isinstance(self.threshold, (int, float)):
            raise TypeError("threshold must be a real number")
        self.threshold = float(self.threshold)

        if self.threshold_mode == "rr":
            if not 0 < self.threshold < 1:
                raise ValueError(
                    "For threshold_mode='rr', threshold must be in (0, 1)."
                )
        else:
            if self.threshold <= 0:
                raise ValueError("threshold must be positive.")

        if not isinstance(self.lmin, int):
            raise TypeError("lmin must be an integer")
        if self.lmin < 1:
            raise ValueError("lmin must be >= 1")

        self._validate_named_or_callable_metric(
            self.metric, "metric", allowed_named_metrics
        )
        self._validate_named_or_callable_metric(
            self.std_metric, "std_metric", allowed_named_metrics
        )

    @staticmethod
    def _validate_named_or_callable_metric(value, name, allowed_named_metrics):
        if isinstance(value, str):
            if value.lower() not in allowed_named_metrics:
                raise ValueError(
                    f"{name} must be one of {sorted(allowed_named_metrics)} or a callable"
                )
        elif not callable(value):
            raise TypeError(f"{name} must be a string or a callable")


def _as_2d_float_array(time_series: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Convert the input time series to a 2D float array of shape (N, d).
    """
    X = np.asarray(time_series, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    elif X.ndim != 2:
        raise ValueError("time_series must be a 1D or 2D array.")
    return X


def _threshold_from_std(
    X: NDArray[np.float64],
    scale: float,
    metric: StdMetric = "euclidean",
) -> float:
    sigma = np.std(X, axis=0)

    if isinstance(metric, str):
        metric_map = {
            "manhattan": 1,
            "euclidean": 2,
            "supremum": np.inf,
        }
        norm_order = metric_map[metric.lower()]
        eps = scale * np.linalg.norm(sigma, ord=norm_order)
    else:
        eps = scale * float(metric(sigma))

    if eps == 0:
        eps = 0.1

    if eps < 0:
        raise ValueError(
            "Computed threshold eps < 0. This usually indicates a constant "
            "time series, a zero scaling factor, or a non-positive custom std_metric."
        )

    return float(eps)



# The integer codes `_recurrence_matrix` branches on. Defined once and used by
# every caller: the threshold and the matrix must agree on what "euclidean"
# means, and keeping two copies of this mapping is exactly how they came to
# disagree before.
METRIC_IDS = {"supremum": 0, "manhattan": 1, "euclidean": 2}


def metric_id_of(metric: str) -> int:
    """Return the integer code for a named metric."""
    key = metric.lower()
    if key not in METRIC_IDS:
        raise ValueError(
            f"Metric must be one of {sorted(METRIC_IDS)}, or a callable; got {metric!r}"
        )
    return METRIC_IDS[key]


@njit(inline="always")
def _pair_distance(
    arr: NDArray[np.float64], i: int, j: int, metric_id: int
) -> float:
    """Distance between rows `i` and `j` under the selected built-in metric."""
    d = arr.shape[1]
    if metric_id == 0:  # supremum
        dist = 0.0
        for k in range(d):
            value = abs(arr[i, k] - arr[j, k])
            if value > dist:
                dist = value
    elif metric_id == 1:  # manhattan
        dist = 0.0
        for k in range(d):
            dist += abs(arr[i, k] - arr[j, k])
    else:  # euclidean
        dist = 0.0
        for k in range(d):
            value = arr[i, k] - arr[j, k]
            dist += value * value
        dist = np.sqrt(dist)
    return dist


@njit(parallel=True)
def _distance_bounds(
    arr: NDArray[np.float64], metric_id: int, n_blocks: int
) -> tuple[float, float]:
    """Smallest and largest off-diagonal pairwise distance, without storing any."""
    N = arr.shape[0]
    lows = np.full(n_blocks, np.inf, dtype=np.float64)
    highs = np.full(n_blocks, -np.inf, dtype=np.float64)
    for block in prange(n_blocks):
        low = np.inf
        high = -np.inf
        for i in range(block, N, n_blocks):
            for j in range(i + 1, N):
                dist = _pair_distance(arr, i, j, metric_id)
                if dist < low:
                    low = dist
                if dist > high:
                    high = dist
        lows[block] = low
        highs[block] = high
    return lows.min(), highs.max()


@njit(parallel=True)
def _distance_histogram(
    arr: NDArray[np.float64],
    metric_id: int,
    low: float,
    high: float,
    n_bins: int,
    n_blocks: int,
) -> tuple[NDArray[np.int64], int]:
    """
    Histogram of the off-diagonal pairwise distances over `[low, high]`.

    Returns the per-bin counts and the number of distances falling strictly
    below `low`. Nothing proportional to the number of pairs is ever stored, so
    the memory cost is the number of bins rather than N squared.
    """
    N = arr.shape[0]
    counts = np.zeros((n_blocks, n_bins), dtype=np.int64)
    below = np.zeros(n_blocks, dtype=np.int64)
    width = (high - low) / n_bins
    for block in prange(n_blocks):
        for i in range(block, N, n_blocks):
            for j in range(i + 1, N):
                dist = _pair_distance(arr, i, j, metric_id)
                if dist < low:
                    below[block] += 1
                elif dist <= high:
                    index = int((dist - low) / width)
                    if index >= n_bins:
                        index = n_bins - 1
                    counts[block, index] += 1
    total = np.zeros(n_bins, dtype=np.int64)
    for block in range(n_blocks):
        for b in range(n_bins):
            total[b] += counts[block, b]
    return total, below.sum()


@njit
def _distances_in_range(
    arr: NDArray[np.float64],
    metric_id: int,
    low: float,
    high: float,
    capacity: int,
) -> NDArray[np.float64]:
    """Collect the off-diagonal distances lying in `[low, high]`."""
    N = arr.shape[0]
    out = np.empty(capacity, dtype=np.float64)
    n = 0
    for i in range(N):
        for j in range(i + 1, N):
            dist = _pair_distance(arr, i, j, metric_id)
            if low <= dist <= high and n < capacity:
                out[n] = dist
                n += 1
    return out[:n]


def _threshold_from_rr(
    X: NDArray[np.float64],
    recurrence_rate: float,
    metric: PairwiseMetric = "supremum",
    n_bins: int = 4096,
    max_collect: int = 1 << 20,
) -> float:
    """
    Compute the recurrence threshold epsilon for a fixed recurrence rate.

    The threshold is the `recurrence_rate`-quantile of the off-diagonal pairwise
    distance distribution, matching `numpy.quantile` with linear interpolation.

    Parameters
    ----------
    X : NDArray[np.float64]
        Time series of shape `(N, d)`.
    recurrence_rate : float
        Target recurrence rate in `[0, 1]`.
    metric : PairwiseMetric, optional
        Named metric or a callable taking two points.
    n_bins : int, optional
        Number of histogram bins used per refinement round.
    max_collect : int, optional
        Largest number of distances materialised in the final exact step.

    Returns
    -------
    float
        The requested quantile of the pairwise distance distribution.

    Notes
    -----
    The quantile is found without ever building the distance matrix. Forming it
    outright costs order `N**2` doubles for the distances, plus the same again
    for the coordinate differences and for the pair of index arrays that extract
    the upper triangle, so the peak was several times the size of the recurrence
    matrix the threshold is being computed for.

    Instead the distances are histogrammed in a streaming pass, the bin holding
    the wanted order statistic is located, and the search is repeated inside that
    bin until few enough distances remain to sort exactly. Each round narrows the
    bracket by a factor of `n_bins`, so two or three passes suffice, and the
    memory used is the number of bins rather than the number of pairs. The result
    is exact, not an approximation: only the last, tiny bracket is materialised.
    """
    if not 0.0 <= recurrence_rate <= 1.0:
        raise ValueError("recurrence_rate must be between 0 and 1.")

    N = X.shape[0]
    if N < 2:
        raise ValueError("time_series must contain at least two samples.")

    n_pairs = N * (N - 1) // 2
    n_blocks = min(64, max(1, N))

    if not isinstance(metric, str):
        # A Python callable cannot be compiled, so fall back to evaluating it
        # row by row. This is slow, but it still never holds more than one row
        # of distances at a time.
        distances = np.empty(n_pairs, dtype=np.float64)
        pos = 0
        for i in range(N):
            for j in range(i + 1, N):
                distances[pos] = metric(X[i], X[j])
                pos += 1
        return float(np.quantile(distances, recurrence_rate))

    metric_id = metric_id_of(metric)

    low, high = _distance_bounds(X, metric_id, n_blocks)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return float(low)

    # numpy.quantile with linear interpolation reads between these two order
    # statistics of the ascending distances.
    position = (n_pairs - 1) * recurrence_rate
    lower_index = int(np.floor(position))
    fraction = position - lower_index
    upper_index = min(lower_index + 1, n_pairs - 1)

    n_below = 0
    while True:
        counts, below = _distance_histogram(X, metric_id, low, high, n_bins, n_blocks)
        n_below += below

        width = (high - low) / n_bins
        cumulative = n_below
        chosen = n_bins - 1
        for b in range(n_bins):
            if cumulative + counts[b] > lower_index:
                chosen = b
                break
            cumulative += counts[b]

        # Extend forward until the upper order statistic is inside the bracket
        # too, since the interpolation needs both. Stepping a single bin is not
        # enough: with many bins and few pairs most bins are empty, and stopping
        # early silently drops the interpolation and biases the result low.
        chosen_high = chosen
        running = cumulative + counts[chosen]
        while running <= upper_index and chosen_high + 1 < n_bins:
            chosen_high += 1
            running += counts[chosen_high]

        bracket_low = low + chosen * width
        bracket_high = low + (chosen_high + 1) * width
        if chosen_high == n_bins - 1:
            bracket_high = high
        span = running - cumulative

        low, high, n_below = bracket_low, bracket_high, cumulative
        if span <= max_collect or bracket_high - bracket_low <= 0.0:
            break

    values = _distances_in_range(X, metric_id, low, high, int(span) + 1)
    values.sort()

    offset = lower_index - n_below
    if offset < 0:
        offset = 0
    if offset >= values.size:
        offset = values.size - 1
    result = values[offset]
    if fraction > 0.0 and offset + 1 < values.size:
        result = result + fraction * (values[offset + 1] - result)
    return float(result)


def calculate_threshold(time_series: NDArray[np.float64], config) -> float:
    """
    Calculate the recurrence threshold according to the configuration.

    Behavior
    --------
    threshold_std=True  -> threshold computed from data standard deviation
    fixed_rr=True       -> threshold computed from fixed recurrence rate
    neither             -> threshold returned directly
    """

    X = _as_2d_float_array(time_series)

    if config.threshold_mode == "direct":
        return float(config.threshold)

    if config.threshold_mode == "std":
        return _threshold_from_std(X, config.threshold, config.std_metric)

    if config.threshold_mode == "rr":
        return _threshold_from_rr(X, config.threshold, config.metric)

    raise ValueError("Invalid threshold_mode.")


@njit
def _recurrence_matrix(
    arr: NDArray[np.float64], threshold: float, metric_id: int
) -> NDArray[np.uint8]:
    """
    Compute the binary recurrence matrix for a 2D array using a built-in metric.

    Parameters
    ----------
    arr : NDArray[np.float64]
        Input array of shape (N, d), where N is the number of samples and
        d is the state-space dimension.
    threshold : float
        Recurrence threshold. A recurrence is detected when the distance
        between two points is strictly less than this threshold.
    metric_id : int
        Built-in metric selector:
            - 0 : supremum norm
            - 1 : manhattan norm
            - 2 : euclidean norm

    Returns
    -------
    NDArray[np.uint8]
        Binary recurrence matrix of shape (N, N).
    """
    N, d = arr.shape
    recmat = np.zeros((N, N), dtype=np.uint8)

    for i in range(N):
        for j in range(i, N):
            if metric_id == 0:  # supremum
                dist = 0.0
                for k in range(d):
                    diff = abs(arr[i, k] - arr[j, k])
                    if diff > dist:
                        dist = diff

            elif metric_id == 1:  # manhattan
                dist = 0.0
                for k in range(d):
                    dist += abs(arr[i, k] - arr[j, k])

            else:  # metric_id == 2, euclidean
                dist = 0.0
                for k in range(d):
                    diff = arr[i, k] - arr[j, k]
                    dist += diff * diff
                dist = np.sqrt(dist)

            if dist < threshold:
                recmat[i, j] = 1
                recmat[j, i] = 1

    return recmat


@njit
def _recurrence_matrix_callable(
    arr: NDArray[np.float64], threshold: float, metric
) -> NDArray[np.uint8]:
    N = arr.shape[0]
    recmat = np.zeros((N, N), dtype=np.uint8)

    for i in range(N):
        for j in range(i, N):
            dist = metric(arr[i], arr[j])
            if dist < threshold:
                recmat[i, j] = 1
                recmat[j, i] = 1

    return recmat


def build_recurrence_matrix(
    arr: NDArray[np.float64], threshold: float, metric: PairwiseMetric = "supremum"
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
    if isinstance(metric, str):
        metric_id = metric_id_of(metric)

        return _recurrence_matrix(arr, threshold, metric_id)

    return _recurrence_matrix_callable(arr, threshold, metric)


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
