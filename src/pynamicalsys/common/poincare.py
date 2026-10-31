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
from numpy.typing import NDArray


@njit
def wrap_period(x: np.float64, period: np.float64) -> np.float64:
    """Wrap x into (-period/2, period/2)."""
    half = 0.5 * period
    x = x % period
    if x >= half:
        x -= period
    return x


@njit
def detect_crossing(
    g_old: np.float64, g_new: np.float64, crossing: int, first_step: bool
) -> bool:
    # A point exactly on the section is counted once, at the step that arrives
    # on it, and not again at the step that leaves it. The initial point is
    # counted when it lies on the section and the trajectory leaves it in the
    # requested direction.
    if g_old == 0.0:
        if not first_step:
            return False
        if crossing == 0:
            return bool(g_new != 0.0)
        if crossing == 1:
            return bool(g_new > 0.0)
        return bool(g_new < 0.0)
    if crossing == 0:
        return bool((g_old < 0.0 and g_new >= 0.0) or (g_old > 0.0 and g_new <= 0.0))
    if crossing == 1:
        return bool(g_old < 0.0 and g_new >= 0.0)
    return bool(g_old > 0.0 and g_new <= 0.0)


@njit
def hermite_crossing(
    g_old: np.float64,
    g_new: np.float64,
    dg_old: np.float64,
    dg_new: np.float64,
    time_step: np.float64,
) -> np.float64:
    """
    Fraction `s` of the step at which the section function crosses zero.

    The section function is interpolated by the cubic Hermite polynomial that
    matches its values `g_old` and `g_new` and its time derivatives `dg_old`
    and `dg_new` at the two ends of a step of length `time_step`. The two
    values must have opposite signs, or one of them must be zero, so the
    polynomial has a zero in [0, 1]. It is found by Newton's method kept
    inside the bracket where the sign changes.
    """
    if g_old == 0.0:
        return np.float64(0.0)
    if g_new == 0.0:
        return np.float64(1.0)

    m_old = time_step * dg_old
    m_new = time_step * dg_new
    lower = np.float64(0.0)
    upper = np.float64(1.0)
    s = g_old / (g_old - g_new)

    for _ in range(50):
        s2 = s * s
        s3 = s2 * s
        g = (
            (2.0 * s3 - 3.0 * s2 + 1.0) * g_old
            + (s3 - 2.0 * s2 + s) * m_old
            + (-2.0 * s3 + 3.0 * s2) * g_new
            + (s3 - s2) * m_new
        )
        if g == 0.0:
            return s
        if (g > 0.0) == (g_old > 0.0):
            lower = s
        else:
            upper = s

        dg = (
            (6.0 * s2 - 6.0 * s) * g_old
            + (3.0 * s2 - 4.0 * s + 1.0) * m_old
            + (-6.0 * s2 + 6.0 * s) * g_new
            + (3.0 * s2 - 2.0 * s) * m_new
        )
        s_next = s - g / dg if dg != 0.0 else 0.5 * (lower + upper)
        if not (lower < s_next < upper):
            s_next = 0.5 * (lower + upper)
        if abs(s_next - s) <= 1e-15:
            return s_next
        s = s_next

    return s


@njit
def hermite_point(
    u_old: NDArray[np.float64],
    u_new: NDArray[np.float64],
    du_old: NDArray[np.float64],
    du_new: NDArray[np.float64],
    time_step: np.float64,
    s: np.float64,
) -> NDArray[np.float64]:
    """
    Cubic Hermite interpolation of a state at the fraction `s` of a step.

    It matches the states `u_old` and `u_new` and their time derivatives
    `du_old` and `du_new` at the ends of a step of length `time_step`, and
    returns the end states exactly at `s = 0` and `s = 1`.
    """
    if s == 0.0:
        return u_old.copy()
    if s == 1.0:
        return u_new.copy()
    s2 = s * s
    s3 = s2 * s
    return (
        (2.0 * s3 - 3.0 * s2 + 1.0) * u_old
        + (s3 - 2.0 * s2 + s) * time_step * du_old
        + (-2.0 * s3 + 3.0 * s2) * u_new
        + (s3 - s2) * time_step * du_new
    )
