"""
HamiltonianSystem.CLV: qr_time_step that is not a multiple of the time step.

`qr_time_step` is rounded to a whole number of integration steps, and
`warmup_time` and `tail_time` are counted in blocks of that rounded spacing.
With the time step 0.01, `qr_time_step=0.025` is therefore the same as
`qr_time_step=0.02`, and must give the same vectors and samples.
"""

import numpy as np

from pynamicalsys import HamiltonianSystem as HS


def test_rounded_qr_time_step_keeps_warmup_and_tail_times():
    hs = HS(model="henon heiles")
    hs.integrator("svy4", time_step=0.01)
    q, p = [0.0, 0.1], [0.3, 0.0]
    options = dict(total_time=5.0, warmup_time=10.0, tail_time=10.0)

    clvs, trajectory = hs.CLV(q, p, qr_time_step=0.025, **options)
    clvs_ref, trajectory_ref = hs.CLV(q, p, qr_time_step=0.02, **options)

    assert trajectory[0, 0] == 10.0
    np.testing.assert_array_equal(trajectory, trajectory_ref)
    np.testing.assert_array_equal(clvs, clvs_ref)
