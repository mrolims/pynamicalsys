"""
White vertical line distribution of a recurrence matrix.

Lines touching the left or right border are excluded (Kraemer & Marwan, 2019),
but a line closed by a recurrence point in the last column does not touch the
border and must be counted.
"""

import numpy as np
import pytest

from pynamicalsys import DiscreteDynamicalSystem as DDS
from pynamicalsys import TimeSeriesMetrics
from pynamicalsys.common.recurrence_quantification_analysis import (
    white_vertline_distr,
)


def _matrix_with_line(N, start, length):
    """All-recurrent matrix with one white line in row 0, kept symmetric."""
    M = np.ones((N, N), dtype=np.uint8)
    M[0, start : start + length] = 0
    M[start : start + length, 0] = 0
    return M


def test_line_closed_in_last_column_is_counted():
    # Row 0 is [1 0 0 0 1]: the line is closed at column N - 1.
    M = _matrix_with_line(5, start=1, length=3)
    P = white_vertline_distr(M)
    assert P[3 - 1] == 1
    assert P.sum() == 1


def test_line_closed_before_last_column_is_counted():
    M = _matrix_with_line(6, start=1, length=3)
    P = white_vertline_distr(M)
    assert P[3 - 1] == 1
    assert P.sum() == 1


def test_border_lines_are_excluded():
    # Row 0 is [1 1 0 0 0]: the line runs into the right border.
    M = np.ones((5, 5), dtype=np.uint8)
    M[0, 2:] = 0
    M[2:, 0] = 0
    assert white_vertline_distr(M).sum() == 0


@pytest.mark.parametrize("wmin", [1, 2, 3])
def test_index_k_is_length_k_plus_wmin(wmin):
    M = np.ones((10, 10), dtype=np.uint8)
    for row, length in [(0, 2), (4, 3)]:
        M[row, row + 1 : row + 1 + length] = 0
        M[row + 1 : row + 1 + length, row] = 0
    full = white_vertline_distr(M, wmin=1)
    np.testing.assert_array_equal(white_vertline_distr(M, wmin=wmin), full[wmin - 1 :])


@pytest.mark.parametrize("lmin", [1, 2, 3])
def test_time_series_rte_matches_system_rte(lmin):
    ds = DDS(model="standard map")
    u = np.array([0.05, 0.05])
    k = 1.5
    N = 2000

    rte_sys, P_sys = ds.recurrence_time_entropy(
        u, N, parameters=k, lmin=lmin, return_p=True
    )
    ts = ds.trajectory(u, N, parameters=k)
    rte_ts, P_ts = TimeSeriesMetrics(ts).recurrence_time_entropy(
        lmin=lmin, return_p=True
    )

    np.testing.assert_allclose(P_ts, P_sys)
    assert rte_ts == pytest.approx(rte_sys)
