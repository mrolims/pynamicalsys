# basin_metrics.py

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
from numbers import Integral, Real
from typing import Optional, Tuple
from numpy.typing import NDArray
from pynamicalsys.common.basin_analysis import basin_entropy, uncertainty_fraction


class BasinMetrics:
    """A class for computing metrics related to basin of attraction analysis, such as basin entropy and uncertainty fraction.

    This class provides methods to quantify the unpredictability and complexity of basins of attraction in dynamical systems. It supports calculation of basin entropy, boundary basin entropy, and the uncertainty fraction, which are useful for characterizing the structure and boundaries of basins.

    Parameters
    ----------
    basin : NDArray[np.float64]
        A 2D array representing the basin of attraction, where each element indicates
        the final state (attractor) for that initial condition (shape: (Nx, Ny)).

    Raises
    ------
    ValueError
        If `basin` is not a 2-dimensional array.

    Notes
    -----
    The basin should be a 2D array where each element represents the final state
    (attractor) for that initial condition. The shape of the basin should be (Nx, Ny),
    where Nx is the number of rows and Ny is the number of columns.

    Examples
    --------
    >>> import numpy as np
    >>> from pynamicalsys import BasinMetrics
    >>> basin = np.array([[0, 1], [1, 0]])
    >>> metrics = BasinMetrics(basin)
    """

    def __init__(self, basin: NDArray[np.float64]) -> None:
        self.basin = basin

        if isinstance(self.basin, list):
            self.basin = np.array(self.basin, dtype=np.float64)

        if basin.ndim != 2:
            raise ValueError("basin must be 2-dimensional")

        pass

    def basin_entropy(
        self,
        n: int,
        log_base: float = np.e,
        nx: Optional[int] = None,
        ny: Optional[int] = None,
    ) -> Tuple[float, float]:
        """Calculate the basin entropy (Sb) and boundary basin entropy (Sbb) of a 2D basin.

        The basin entropy quantifies the uncertainty in final state prediction, while the boundary
        entropy specifically measures uncertainty at basin boundaries where multiple attractors coexist.

        Parameters
        ----------
        n : int
            Default size of square sub-boxes for partitioning (must be positive).
        log : float, optional
            Logarithm base for entropy calculation (default: np.e, which is natural logarithm).

        Returns
        -------
        Tuple[float, float]
            A tuple containing:

            - Sb: Basin entropy
            - Sbb: Boundary basin entropy

        Raises
        ------
        ValueError
            If `n`, is not positive integer, or if `log_base` is not positive.

        Notes
        -----
        The basin entropy is calculated by partitioning the basin into sub-boxes of size `n` and computing the entropy of each sub-box. The boundary basin entropy is computed similarly but focuses on the sub-boxes that lie on the boundaries of the basin where multiple attractors coexist.

        Examples
        --------
        >>> import numpy as np
        >>> np.random.seed(13)
        >>> basin = np.random.randint(1, 4, size=(1000, 1000))
        >>> from pynamicalsys import BasinMetrics
        >>> metrics = BasinMetrics(basin)
        >>> metrics.basin_entropy(n=5, log_base=2)
        (1.5251876046167432, 1.5251876046167432)
        """

        if not isinstance(n, Integral) or n <= 0:
            raise ValueError("n must be positive integer")

        if log_base <= 0:
            raise ValueError("log_base must be positive")

        return basin_entropy(basin=self.basin, n=n, log_base=log_base)

    def uncertainty_fraction(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        epsilon_max: float = 0.1,
        n_eps: int = 100,
        epsilon_min: Optional[int] = None,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Calculate the uncertainty fraction for a given basin.

        This method computes the uncertainty fraction for each point in the basin
        based on the provided parameters.

        Parameters
        ----------
        x : NDArray[np.float64]
            2D array of the basin's x-coordinates.
        y : NDArray[np.float64]
            2D array of the basin's y-coordinates.
        epsilon_max : float, optional
            Maximum epsilon value (default: 0.1).
        n_eps : int, optional
            Number of epsilon values to consider (default: 100).
        epsilon_min : int, optional
            Minimum epsilon value (default: None).

        Returns
        -------
        Tuple[NDArray[np.float64], NDArray[np.float64]]
            A tuple containing:

            - epsilons: Array of epsilon values.
            - uncertainty_fraction: Array of uncertainty fractions corresponding to each epsilon.

        Notes
        -----
        - The uncertainty fraction scales with ε as a power law: f(ε) ~ ε^{⍺}, where ⍺ is the uncertainty exponent.
        - For D-dimensional basins, the dimension d of the basin boundary is given by d = D - ⍺.

        Examples
        --------
        >>> # Create a basin of 0's and 1's, where the 1's form a rectangle, i.e., d = 1
        >>> grid_size = 10000
        >>> x_range = (0, 1, grid_size)
        >>> y_range = (0, 1, grid_size)
        >>> x = np.linspace(*x_range)
        >>> y = np.linspace(*y_range)
        >>> X, Y = np.meshgrid(x, y, indexing='ij')
        >>> obj = [[0.2, 0.6],
            [0.2, 0.6]]
        >>> basin = np.zeros((grid_size, grid_size), dtype=int)
        >>> basin[mask] = 1
        >>> bm = BasinMetrics(basin)
        >>> eps, f = bm.uncertainty_fraction(X, Y, epsilon_max=0.1)
        """

        if isinstance(x, list):
            x = np.array(x, dtype=np.float64)
        if isinstance(y, list):
            y = np.array(y, dtype=np.float64)

        if x.ndim != 2 or y.ndim != 2:
            raise ValueError("x, y, and basin must be 2-dimensional arrays")
        if x.shape != y.shape or x.shape != self.basin.shape:
            raise ValueError("x, y, and basin must have the same shape")

        if not isinstance(epsilon_max, Real) or epsilon_max < 0:
            raise ValueError("epsilon_min must be a non-negative real number")

        if not isinstance(n_eps, Integral) or n_eps <= 0:
            raise ValueError("n_eps must be a positive integer")

        if epsilon_min is not None:
            if not isinstance(epsilon_min, Real) and epsilon_min < 0:
                raise ValueError("epsilon_min must be a non-negative real number")
        else:
            epsilon_min = 0.0

        return uncertainty_fraction(
            x=x,
            y=y,
            basin=self.basin,
            epsilon_max=epsilon_max,
            n_eps=n_eps,
            epsilon_min=epsilon_min,
        )
