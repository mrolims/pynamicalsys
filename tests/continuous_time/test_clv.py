"""
ContinuousDynamicalSystem.CLV: the warmup.

With rk4, a QR interval that is not a multiple of the time step ends with a
shortened step that lands on the block boundary. The step size must return to
`time_step` afterwards, so that every block of the warmup takes the same steps
as the blocks of the stored stage, which restore it.
"""

import numpy as np

from pynamicalsys import ContinuousDynamicalSystem as CDS


def test_warmup_restores_the_step_size_after_each_block():
    ds = CDS(model="lorenz system")
    ds.set_parameters([10.0, 28.0, 8.0 / 3.0])
    ds.integrator("rk4", time_step=0.03)
    u = [1.0, 1.0, 1.0]

    # Without warmup, the stored states at t = 0.1 and 0.2 come from blocks
    # of steps 0.03, 0.03, 0.03, 0.01.
    _, stored = ds.CLV(u, 3.0, qr_time_step=0.1)
    # A warmup of two blocks must take the same steps, so the state at the
    # end of the warmup is the stored state at t = 0.2.
    _, warmed = ds.CLV(u, 3.0, qr_time_step=0.1, warmup_time=0.2)

    np.testing.assert_array_equal(warmed[0, 1:], stored[2, 1:])
