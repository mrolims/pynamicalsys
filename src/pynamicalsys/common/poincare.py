# poincare.py

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

from numba import njit
import numpy as np


@njit
def wrap_period(x: np.float64, period: np.float64) -> np.float64:
    """Wrap x into (-period/2, period/2)."""
    half = 0.5 * period
    x = x % period
    if x >= half:
        x -= period
    return x


@njit
def detect_crossing(g_old: np.float64, g_new: np.float64, crossing: int) -> bool:
    if crossing == 0:
        return bool(
            (g_old < 0.0 and g_new >= 0.0)
            or (g_old > 0.0 and g_new <= 0.0)
            or (g_old == 0.0 and g_new != 0.0)
        )
    if crossing == 1:
        return bool((g_old < 0.0 and g_new >= 0.0) or (g_old == 0.0 and g_new > 0.0))
    return bool((g_old > 0.0 and g_new <= 0.0) or (g_old == 0.0 and g_new < 0.0))
