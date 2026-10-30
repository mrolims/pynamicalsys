"""
ContinuousDynamicalSystem.CLV: the warmup.

With rk4, a QR interval that is not a multiple of the time step ends with a
shortened step that lands on the block boundary. The step size must return to
`time_step` afterwards, so that every block of the warmup takes the same steps
as the blocks of the stored stage, which restore it.

The stored samples cover `[transient_time + warmup_time, total_time]`, so with
a warmup they are the samples of a run without warmup from the end of the
warmup on.
"""

import numpy as np
import pytest

from pynamicalsys import ContinuousDynamicalSystem as CDS


@pytest.fixture
def lorenz():
    ds = CDS(model="lorenz system")
    ds.set_parameters([10.0, 28.0, 8.0 / 3.0])
    return ds


def test_warmup_restores_the_step_size_after_each_block(lorenz):
    lorenz.integrator("rk4", time_step=0.03)
    u = [1.0, 1.0, 1.0]

    # Without warmup, the stored states at t = 0.1 and 0.2 come from blocks
    # of steps 0.03, 0.03, 0.03, 0.01.
    _, stored = lorenz.CLV(u, 3.0, qr_time_step=0.1)
    # A warmup of two blocks must take the same steps, so the state at the
    # end of the warmup is the stored state at t = 0.2.
    _, warmed = lorenz.CLV(u, 3.0, qr_time_step=0.1, warmup_time=0.2)

    np.testing.assert_array_equal(warmed[0, 1:], stored[2, 1:])


@pytest.mark.parametrize("transient_time", [None, 1.0])
def test_stored_window_starts_after_the_warmup(lorenz, transient_time):
    # Binary fractions, so the block counts are free of round-off.
    lorenz.integrator("rk4", time_step=0.03125)
    u = [1.0, 1.0, 1.0]
    t0 = 0.0 if transient_time is None else transient_time
    options = dict(qr_time_step=0.125, transient_time=transient_time)

    _, stored = lorenz.CLV(u, t0 + 3.0, **options)
    _, warmed = lorenz.CLV(u, t0 + 3.0, warmup_time=0.5, **options)

    np.testing.assert_array_equal(warmed[:, 0], t0 + 0.5 + 0.125 * np.arange(21))
    np.testing.assert_allclose(warmed, stored[4:], rtol=1e-12)


def test_warmup_must_end_before_total_time(lorenz):
    lorenz.integrator("rk4", time_step=0.01)
    with pytest.raises(ValueError, match="warmup_time"):
        lorenz.CLV([1.0, 1.0, 1.0], 1.0, transient_time=0.5, warmup_time=0.5)
    with pytest.raises(ValueError, match="warmup_time"):
        lorenz.CLV_angles(
            [1.0, 1.0, 1.0], 1.0, pairs=(0, 2), transient_time=0.5, warmup_time=0.5
        )
