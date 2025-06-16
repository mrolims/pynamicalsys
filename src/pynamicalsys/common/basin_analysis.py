# basin_analysis.py

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
from numba import njit, prange
from typing import Optional, Tuple
from numpy.typing import NDArray


def uncertainty_fraction(x, y, basin, epsilon_max, n_eps=100, epsilon_min=0):
    """
    Wrapper to compute uncertainty fractions using a njit-compatible core function.
    """

    # Estimate dx and dy (uniform spacing assumed)
    dx = np.mean(np.abs(x[1:, :] - x[:-1, :]))
    dy = np.mean(np.abs(y[:, 1:] - y[:, :-1]))
    if epsilon_min != 0:
        min_epsilon = epsilon_min
    else:
        min_epsilon = max(dx, dy)

    # Log-spaced epsilon values (outside JIT)
    epsilons = np.logspace(np.log10(min_epsilon), np.log10(epsilon_max), n_eps)

    return epsilons, _uncertainty_fraction_core(basin, dx, dy, epsilons)


@njit(cache=True, parallel=True)
def _uncertainty_fraction_core(basin, dx, dy, epsilons):
    """
    Numba-compatible core function for computing uncertainty fractions.
    """
    nx, ny = basin.shape
    f_epsilons = np.zeros(len(epsilons))

    for k in prange(len(epsilons)):
        eps = epsilons[k]
        dx_pix = int(round(eps / dx))
        dy_pix = int(round(eps / dy))

        if dx_pix < 1 or dy_pix < 1:
            f_epsilons[k] = 0.0
            continue

        uncertain = 0
        total = 0

        for i in range(dx_pix, nx - dx_pix):
            for j in range(dy_pix, ny - dy_pix):
                center = basin[i, j]
                neighbors = (
                    basin[i + dx_pix, j],
                    basin[i - dx_pix, j],
                    basin[i, j + dy_pix],
                    basin[i, j - dy_pix],
                )
                for nb in neighbors:
                    if nb != center:
                        uncertain += 1
                        break  # once marked uncertain, no need to check other neighbors
                total += 1

        if total > 0:
            f_epsilons[k] = uncertain / total

    return f_epsilons


def basin_entropy(
    basin: NDArray[np.float64], n: int, log_base: float = np.e
) -> Tuple[float, float]:
    """
    Calculate the basin entropy (Sb) and boundary basin entropy (Sbb) of a 2D attraction basin.

    The basin entropy quantifies the uncertainty in final state prediction, while the boundary
    entropy specifically measures uncertainty at basin boundaries where multiple attractors coexist.

    Parameters
    ----------
    basin : NDArray[np.float64]
        2D array representing the basin of attraction, where each element indicates
        the final state (attractor) for that initial condition (shape: (Nx, Ny)).
    n : int
        Default size of square sub-boxes for partitioning (must be positive).
    log : {'2', 'e', '10'} or Callable, optional
        Logarithm base for entropy calculation:
        - '2' : bits (default)
        - 'e' : nats
        - '10' : hartleys
        Alternatively, a custom log function can be provided.

    Returns
    -------
    Sb : float
        Average entropy across all sub-boxes (basin entropy).
    Sbb : float
        Average entropy across boundary sub-boxes (boundary basin entropy).

    Raises
    ------
    ValueError
        - If `basin` is not 2D
        - If `n` ≤ 0
        - If invalid `log` specification

    Notes
    -----
    - **Entropy Calculation**:
        For each sub-box: S = -Σ(p_i * log(p_i)), where p_i is the probability of state i.
    - **Boundary Detection**:
        Sub-boxes with >1 unique state are considered boundaries.
    - **Performance**:
        Uses vectorized operations where possible for efficiency.

    Examples
    --------
    >>> basin = np.random.randint(0, 2, (100, 100))
    >>> Sb, Sbb = boundary_entropy(basin, n=10, log='2')
    >>> print(f"Basin entropy: {Sb:.3f}, Boundary entropy: {Sbb:.3f}")
    """

    Nx, Ny = basin.shape

    if Nx % n != 0 or Ny % n != 0:
        raise ValueError(
            f"Sub-box sizes ({n}, {n}) must divide basin dimensions ({Nx}, {Ny})"
        )

    # Initialize
    Mx = Nx // n
    My = Ny // n
    S = np.zeros((Mx, My))
    boundary_mask = np.zeros((Mx, My), dtype=bool)

    # Process each sub-box
    for i in range(Mx):
        for j in range(My):
            # Extract sub-box
            box = basin[i * n : (i + 1) * n, j * n : (j + 1) * n]
            unique_states, counts = np.unique(box, return_counts=True)
            num_unique = len(unique_states)

            # Mark boundary boxes
            if num_unique > 1:
                boundary_mask[i, j] = True

            # Calculate entropy
            probs = counts / counts.sum()
            S[i, j] = -np.sum(probs * np.log(probs) / np.log(log_base))

    # Compute averages
    Sb = np.mean(S)
    Sbb = np.mean(S[boundary_mask]) if boundary_mask.any() else 0.0

    return float(Sb), float(Sbb)
