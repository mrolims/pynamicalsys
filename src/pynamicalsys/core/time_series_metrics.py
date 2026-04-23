# time_series_metrics.py

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

from typing import Union, Tuple

import numpy as np
from numpy.typing import NDArray

from numbers import Integral

from pynamicalsys.common.recurrence_quantification_analysis import (
    calculate_threshold,
    build_recurrence_matrix,
    RTEConfig,
    white_vertline_distr,
)

from pynamicalsys.common.validators import validate_positive

from pynamicalsys.common.hurst import hurst_exponent


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

        if time_series.shape[0] < 2:
            raise ValueError("time_series must contain at least two points.")

    def recurrence_matrix(
        self, compute_white_vert_distr=False, return_eps=False, **kwargs
    ) -> Union[
        NDArray[np.uint8],
        Tuple[NDArray[np.uint8], NDArray[np.float64]],
        Tuple[NDArray[np.uint8], float],
        Tuple[NDArray[np.uint8], NDArray[np.float64], float],
    ]:
        """
        Compute the recurrence matrix for the time series.

        The recurrence threshold can be determined in three different ways,
        controlled by ``threshold_mode``:

        - ``threshold_mode="direct"``: ``threshold`` is used directly as the
          recurrence threshold.

        - ``threshold_mode="std"``: the threshold is computed from the standard
          deviation of the data as

              eps = threshold * ||sigma||_p

          where ``sigma`` is the vector of component-wise standard deviations.

        - ``threshold_mode="rr"``: the threshold is chosen such that the recurrence
          matrix achieves the desired recurrence rate, where ``threshold`` is
          interpreted as the target recurrence rate.

        For backward compatibility, the deprecated parameter ``threshold_std`` may
        still be used. If ``threshold_std=True``, the threshold strategy is treated
        as ``threshold_mode="std"``. A warning will be issued and the parameter will
        be removed in a future release.

        Parameters
        ----------
        compute_white_vert_distr : bool, default=False
            If True, also return the white vertical line length distribution.

        return_eps : bool, default=False
            If True, also return the threshold used in the recurrence matrix
            construction.

        metric : {'supremum', 'euclidean', 'manhattan'} or callable, default='supremum'
            Distance metric used to compute pairwise distances between state vectors.

            If a callable is provided, it must have signature ``metric(x, y) -> float``,
            where ``x`` and ``y`` are 1D NumPy arrays representing state vectors.

            The callable must be **Numba-compatible**, since it will be executed inside
            a Numba-compiled routine. For best performance and reliability, the metric
            should be decorated with ``@numba.njit``.

        std_metric : {'supremum', 'euclidean', 'manhattan'} or callable, default='supremum'
            Metric used in the standard-deviation-based threshold calculation.

            If a callable is provided, it must take the vector of component-wise
            standard deviations and return a scalar with signature
            ``std_metric(sigma) -> float``. It must be **Numba-compatible**
            (ideally decorated with ``@numba.njit``).

        threshold : float, default=0.1
            Threshold parameter. Its meaning depends on the selected strategy:

            - ``threshold_mode="direct"``: direct recurrence threshold
            - ``threshold_mode="std"``: scaling factor for the standard-deviation
              threshold
            - ``threshold_mode="rr"``: target recurrence rate

        threshold_mode : {'direct', 'std', 'rr'}, optional
            Strategy used to determine the recurrence threshold.

        threshold_std : bool, optional
            **Deprecated.** If set to True, the threshold will be computed using
            the standard deviation of the data (equivalent to
            ``threshold_mode="std"``). This parameter will be removed in a future
            release.

        lmin : int, default=1
            Minimum white vertical line length considered when computing the
            white vertical line length distribution.

        Returns
        -------
        NDArray[np.uint8] or tuple
            The returned value depends on the optional flags:

            - If ``compute_white_vert_distr=False`` and ``return_eps=False``:
              returns ``recmat``.

            - If ``compute_white_vert_distr=True`` and ``return_eps=False``:
              returns ``(recmat, distr)``, where ``distr`` is the white vertical
              line length distribution.

            - If ``compute_white_vert_distr=False`` and ``return_eps=True``:
              returns ``(recmat, eps)``, where ``eps`` is the recurrence threshold
              used to construct the matrix.

            - If ``compute_white_vert_distr=True`` and ``return_eps=True``:
              returns ``(recmat, distr, eps)``.
        """
        config = RTEConfig(**kwargs)

        eps = calculate_threshold(self.time_series, config)

        recmat = build_recurrence_matrix(
            self.time_series, float(eps), metric=config.metric
        )

        if compute_white_vert_distr:
            distr = white_vertline_distr(recmat)[config.lmin :]

            if return_eps:
                return recmat, distr, eps
            return recmat, distr

        if return_eps:
            return recmat, eps

        return recmat

    def recurrence_time_entropy(self, **kwargs):
        """
        Compute the recurrence time entropy (RTE) of the time series.

        The recurrence time entropy is computed from the distribution of white
        vertical line lengths in the recurrence matrix.

        Parameters
        ----------
        metric : {'supremum', 'euclidean', 'manhattan'} or callable, default='supremum'
            Distance metric used to compute pairwise distances between state vectors.

            If a callable is provided, it must have signature ``metric(x, y) -> float``.
            The callable must be **Numba-compatible**, since it will be executed inside
            a Numba-compiled routine. For best performance, decorate it with
            ``@numba.njit``.

        std_metric : {'supremum', 'euclidean', 'manhattan'} or callable, default='supremum'
            Metric used in the standard-deviation-based threshold calculation.

            If a callable is provided, it must take the vector of component-wise
            standard deviations and return a scalar with signature
            ``std_metric(sigma) -> float``. It must be **Numba-compatible**
            (ideally decorated with ``@numba.njit``).

        threshold : float, default=0.1
            Threshold parameter. Its interpretation depends on ``threshold_mode``.

        threshold_mode : {'direct', 'std', 'rr'}, optional
            Strategy used to determine the recurrence threshold.

            - ``direct``: use ``threshold`` directly as the recurrence threshold
            - ``std``: compute ``eps = threshold * ||sigma||_p``
            - ``rr``: choose the threshold so that the recurrence matrix achieves
              the desired recurrence rate

        threshold_std : bool, optional
            **Deprecated.** If set to True, the threshold will be computed using
            the standard deviation of the data (equivalent to
            ``threshold_mode="std"``). This parameter will be removed in a future
            release.

        lmin : int, default=1
            Minimum white vertical line length considered in the entropy
            calculation.

        return_recmat : bool, default=False
            If True, also return the recurrence matrix.

        return_p : bool, default=False
            If True, also return the normalized white vertical line length
            distribution used to compute the entropy.

        Returns
        -------
        float or tuple
            - If ``return_recmat=False`` and ``return_p=False``:
              returns the recurrence time entropy.

            - If ``return_recmat=True``:
              returns ``(rte, recmat)``.

            - If ``return_p=True``:
              returns ``(rte, P)``.

            - If both are True:
              returns ``(rte, recmat, P)``.
        """
        config = RTEConfig(**kwargs)

        # Compute the recurrence matrix
        eps = calculate_threshold(self.time_series, config)
        recmat = build_recurrence_matrix(self.time_series, eps, config.metric)

        # Calculate the white vertical line distribution
        P = white_vertline_distr(recmat)[config.lmin :]
        P = P[P > 0]  # Filter out zero probabilities
        P /= P.sum()  # Normalize the distribution

        # Calculate the recurrence time entropy
        rte = -np.sum(P * np.log(P))

        result = [rte]
        if config.return_recmat:
            result.append(recmat)
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

        return result
