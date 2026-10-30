"""
ContinuousDynamicalSystem.stroboscopic_map: the samples are taken at the
sampling times themselves.

The integration lands exactly on every sampling time, so each sample has the
accuracy of the integrator, and a step longer than `sampling_time` is split
instead of skipping samples. The reference is the trajectory sampled at the
same times with a tight rk45, which also lands exactly on each of them.
"""

import numpy as np
import pytest

from pynamicalsys import ContinuousDynamicalSystem as CDS

U0 = [1.0, 1.0, 1.0]
NUM_SAMPLES = 20


@pytest.fixture
def lorenz():
    ds = CDS(model="lorenz system")
    ds.set_parameters([10.0, 28.0, 8.0 / 3.0])
    return ds


def reference(ds, sampling_time):
    times = sampling_time * np.arange(1, NUM_SAMPLES + 1)
    ds.integrator("rk45", atol=1e-13, rtol=1e-13)
    return ds.trajectory(U0, times[-1], sample_times=times)


@pytest.mark.parametrize(
    "integrator, options, sampling_time, tolerance",
    [
        ("rk4", dict(time_step=0.01), 0.105, 1e-3),
        ("rk4", dict(time_step=0.01), 0.004, 1e-6),
        ("rk45", dict(atol=1e-8, rtol=1e-8), 0.105, 1e-5),
    ],
)
def test_samples_are_taken_at_the_sampling_times(
    lorenz, integrator, options, sampling_time, tolerance
):
    expected = reference(lorenz, sampling_time)
    lorenz.integrator(integrator, **options)
    strobe = lorenz.stroboscopic_map(U0, NUM_SAMPLES, sampling_time=sampling_time)

    np.testing.assert_array_equal(strobe[:, 0], expected[:, 0])
    np.testing.assert_allclose(strobe[:, 1:], expected[:, 1:], rtol=0, atol=tolerance)
