# time_series_metrics.py

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
from numpy.typing import NDArray

from numbers import Integral, Real

from pynamicalsys.common.recurrence_quantification_analysis import (
    recurrence_matrix,
    RTEConfig,
    white_vertline_distr,
)

from pynamicalsys.discrete_time.validators import validate_positive

from pynamicalsys.common.time_series_metrics import hurst_exponent


class TimeSeriesMetrics:
    def __init__(self, time_series: NDArray[np.float64]) -> None:
        """
        Initialize the TimeSeriesMetrics class.

        This class provides methods to compute metrics related to time series analysis,
        such as survival probability and in a future release, the autocorrelation function.
        """
        self.time_series = time_series

        # The time series can be either 1D (shape(N,)) or 2D (shape(N, dim))
        if time_series.ndim not in {1, 2}:
            raise ValueError("time_series must be 1D or 2D array")

    def recurrence_matrix(
        self, compute_white_vert_distr=False, **kwargs
    ) -> NDArray[np.float64]:
        """
        Compute the recurrence matrix for the time series.

        Parameters
        ----------
        metric : {'supremum', 'euclidean', 'manhattan'}, default='supremum'
            Distance metric used for phase space reconstruction.
        std_metric : {'supremum', 'euclidean', 'manhattan'}, default='supremum'
            Distance metric used for standard deviation calculation.
        threshold : float, default=0.1
            Recurrence threshold (relative to data range).
        threshold_std : bool, default=True
            Whether to scale threshold by data standard deviation.

        Returns
        -------
        NDArray[np.float64]
            The recurrence matrix of the time series.
        """
        config = RTEConfig(**kwargs)
        # Metric setup
        metric_map = {"supremum": np.inf, "euclidean": 2, "manhattan": 1}
        try:
            ord = metric_map[config.std_metric.lower()]
        except KeyError:
            raise ValueError(
                f"Invalid std_metric: {config.std_metric}. Must be {list(metric_map.keys())}"
            )

        # Threshold calculation
        if config.threshold_std:
            std = np.std(self.time_series, axis=0)
            eps = config.threshold * np.linalg.norm(std, ord=ord)
            if eps <= 0:
                eps = 0.1
        else:
            eps = config.threshold

        if compute_white_vert_distr:
            recmat = recurrence_matrix(
                self.time_series, float(eps), metric=config.metric
            )
            P = white_vertline_distr(recmat)[config.lmin :]

            return recmat, P
        else:
            return recurrence_matrix(self.time_series, float(eps), metric=config.metric)

    def recurrence_time_entropy(self, **kwargs):
        """
        Compute the recurrence time entropy of the time series.

        Parameters
        ----------
        metric : {'supremum', 'euclidean', 'manhattan'}, default='supremum'
            Distance metric used for phase space reconstruction.
        std_metric : {'supremum', 'euclidean', 'manhattan'},                default='supremum'
            Distance metric used for standard deviation calculation.
        lmin : int, default=1
            Minimum line length to consider in recurrence quantification.
        threshold : float, default=0.1
            Recurrence threshold (relative to data range).
        threshold_std : bool, default=True
            Whether to scale threshold by data standard deviation.
        return_recmat : bool, default=False
            Whether to return the recurrence matrix.
        return_p : bool, default=False
            Whether to return white vertical line length distribution.

        Returns
        -------
        Tuple[float, float]
            The recurrence time entropy and its standard deviation.
        """

        config = RTEConfig(**kwargs)

        # Compute the recurrence matrix
        rec_matrix = self.recurrence_matrix(**kwargs)

        # Calculate the white vertical line distribution
        P = white_vertline_distr(rec_matrix)[config.lmin :]
        P = P[P > 0]  # Filter out zero probabilities
        P /= P.sum()  # Normalize the distribution

        # Calculate the recurrence time entropy
        rte = -np.sum(P * np.log(P))

        result = [rte]
        if config.return_recmat:
            result.append(rec_matrix)
        if config.return_p:
            result.append(P)

        return result[0] if len(result) == 1 else tuple(result)

    def hurst_exponent(self, wmin: int = 2):
        """
        Estimate the Hurst exponent for a system trajectory using the rescaled range (R/S) method.

        Parameters
        ----------
        u : NDArray[np.float64]
            Initial condition vector of shape (n,).
        parameters : NDArray[np.float64]
            Parameters passed to the mapping function.
        total_time : int
            Total number of iterations used to generate the trajectory.
        mapping : Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]
            A function that defines the system dynamics, i.e., how `u` evolves over time given `parameters`.
        wmin : int, optional
            Minimum window size for the rescaled range calculation. Default is 2.
        transient_time : Optional[int], optional
            Number of initial iterations to discard as transient. If `None`, no transient is removed. Default is `None`.

        Returns
        -------
        NDArray[np.float64]
            Estimated Hurst exponents for each dimension of the input vector `u`, of shape (n,).

        Notes
        -----
        The Hurst exponent is a measure of the long-term memory of a time series:

        - H = 0.5 indicates a random walk (no memory).
        - H > 0.5 indicates persistent behavior (positive autocorrelation).
        - H < 0.5 indicates anti-persistent behavior (negative autocorrelation).

        This implementation computes the rescaled range (R/S) for various window sizes and
        performs a linear regression in log-log space to estimate the exponent.

        The function supports multivariate time series, estimating one Hurst exponent per dimension.
        """

        sample_size = self.time_series.shape[0]

        validate_positive(wmin, "wmin", Integral)

        if wmin < 2 or wmin >= sample_size // 2:
            raise ValueError(
                f"`wmin` must be an integer >= 2 and <= len(time_series) / 2. Got {wmin}."
            )

        if self.time_series.ndim == 1:
            time_series = self.time_series.reshape(sample_size, 1)
        else:
            time_series = self.time_series

        result = hurst_exponent(time_series, wmin=wmin)

        if self.time_series.ndim == 1:
            return result[0]
        else:
            return result
