"""
The optimized continuous kernels against straightforward reference versions.

The references are the earlier implementations, which build every array with
array expressions and store the trajectory in a list. The RK45 step and the
trajectory must give the same bits. The variational equations compute
`J @ v` with an explicit loop instead of BLAS, so they agree to round-off.
"""

import numpy as np
import pytest
from numba import njit

from pynamicalsys import ContinuousDynamicalSystem as CDS

from pynamicalsys.continuous_time.adaptive_step import (
    _RK45_A,
    _RK45_B4,
    _RK45_B5,
    _RK45_C,
    rk45_step,
)
from pynamicalsys.continuous_time.models import (
    henon_heiles,
    henon_heiles_jacobian,
    lorenz_jacobian,
    lorenz_system,
)
from pynamicalsys.continuous_time.step import step_to
from pynamicalsys.continuous_time.step_methods import (
    rk4_step_wrapped,
    rk45_step_wrapped,
)
from pynamicalsys.continuous_time.trajectory import (
    ensemble_trajectories,
    generate_trajectory,
)
from pynamicalsys.continuous_time.variational import variational_equations

LORENZ_PARAMETERS = np.array([10.0, 28.0, 8.0 / 3.0])
U_LORENZ = np.array([0.1, 0.1, 0.1])


@njit
def reference_rk45_step(t, u, parameters, equations_of_motion, time_step, atol, rtol):
    k = np.empty((7, u.size), dtype=np.float64)
    for i in range(7):
        ti = t + _RK45_C[i] * time_step
        ui = u.copy()
        for j in range(i):
            ui += time_step * _RK45_A[i, j] * k[j]
        k[i] = equations_of_motion(ti, ui, parameters)
    u5 = u.copy()
    u4 = u.copy()
    for i in range(7):
        u5 += time_step * _RK45_B5[i] * k[i]
        u4 += time_step * _RK45_B4[i] * k[i]
    error = np.abs(u5 - u4)
    scale = atol + rtol * np.maximum(np.abs(u), np.abs(u5))
    return u5, np.max(error / scale)


@njit
def reference_trajectory(u, parameters, total_time, equations_of_motion, time_step, atol, rtol, integrator):
    time = 0.0
    trajectory = []
    while time < total_time:
        u, time, time_step, _ = step_to(
            time=time, u=u, target_time=total_time, parameters=parameters,
            equations_of_motion=equations_of_motion, time_step=time_step,
            atol=atol, rtol=rtol, integrator=integrator,
        )
        result = [time]
        for i in range(len(u)):
            result.append(u[i])
        trajectory.append(result)
    return np.asarray(trajectory, dtype=np.float64)


@njit
def reference_variational(time, state, parameters, equations_of_motion, jacobian, ndv):
    nt = len(state)
    neq = nt // (1 + ndv)
    u = state[:neq].copy()
    v = state[neq:].reshape(neq, ndv).copy()
    out = np.empty(nt)
    out[:neq] = equations_of_motion(time, u, parameters)
    out[neq:] = (jacobian(time, u, parameters) @ v).reshape(neq * ndv)
    return out


@pytest.mark.parametrize("time_step", [1e-3, 0.05, 0.5])
def test_rk45_step_is_bit_identical(time_step):
    u = np.array([1.5, -2.0, 20.0])
    for atol, rtol in [(1e-6, 1e-3), (1e-12, 1e-12)]:
        u5, _, h_new, accept = rk45_step(
            0.3, u, LORENZ_PARAMETERS, lorenz_system, time_step, atol, rtol
        )
        expected_u5, err = reference_rk45_step(
            0.3, u, LORENZ_PARAMETERS, lorenz_system, time_step, atol, rtol
        )
        np.testing.assert_array_equal(u5, expected_u5)
        assert accept == (err < 1.0)


def test_rk45_step_propagates_nan_to_the_error():
    @njit
    def blows_up(t, u, parameters):
        return np.array([np.nan, 1.0])

    _, _, h_new, accept = rk45_step(0.0, np.array([1.0, 1.0]), np.zeros(1), blows_up, 0.1)
    assert not accept
    assert np.isnan(h_new)


@pytest.mark.parametrize(
    "integrator, atol, rtol",
    [(rk4_step_wrapped, 1e-6, 1e-3), (rk45_step_wrapped, 1e-10, 1e-10)],
)
def test_trajectory_is_bit_identical(integrator, atol, rtol):
    # Long enough to grow the row buffer several times.
    result = generate_trajectory(
        U_LORENZ, LORENZ_PARAMETERS, 20.0, lorenz_system,
        time_step=0.01, atol=atol, rtol=rtol, integrator=integrator,
    )
    expected = reference_trajectory(
        U_LORENZ.copy(), LORENZ_PARAMETERS, 20.0, lorenz_system,
        0.01, atol, rtol, integrator,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "integrator, atol, rtol",
    [(rk4_step_wrapped, 1e-6, 1e-3), (rk45_step_wrapped, 1e-9, 1e-9)],
)
def test_ensemble_trajectories_match_single_trajectories(integrator, atol, rtol):
    # The ensemble runs in threads; each trajectory must equal the one
    # computed alone with the same arguments.
    ensemble = np.random.default_rng(0).uniform(-10, 10, size=(6, 3)) + [0, 0, 25]
    kwargs = dict(
        transient_time=0.5, time_step=0.01, atol=atol, rtol=rtol,
        integrator=integrator, sample_times=None,
    )
    trajectories = ensemble_trajectories(
        ensemble, LORENZ_PARAMETERS, 5.0, lorenz_system, **kwargs
    )
    assert len(trajectories) == len(ensemble)
    for u, trajectory in zip(ensemble, trajectories):
        expected = generate_trajectory(u, LORENZ_PARAMETERS, 5.0, lorenz_system, **kwargs)
        np.testing.assert_array_equal(trajectory, expected)


@pytest.mark.parametrize(
    "f, jac, u, parameters",
    [
        (lorenz_system, lorenz_jacobian, U_LORENZ, LORENZ_PARAMETERS),
        (henon_heiles, henon_heiles_jacobian, np.array([0.0, 0.1, 0.5, 0.0]), np.zeros(1)),
    ],
)
@pytest.mark.parametrize("ndv", [1, 2])
def test_variational_equations_agree_to_round_off(f, jac, u, parameters, ndv):
    rng = np.random.default_rng(4)
    neq = len(u)
    state = np.concatenate([u, rng.normal(size=neq * ndv)])
    result = variational_equations(0.2, state, parameters, f, jac, ndv)
    expected = reference_variational(0.2, state, parameters, f, jac, ndv)
    np.testing.assert_array_equal(result[:neq], expected[:neq])
    np.testing.assert_allclose(result[neq:], expected[neq:], rtol=1e-14, atol=1e-14)


def test_lyapunov_history_matches_the_final_value():
    ds = CDS(model="lorenz system")
    ds.set_parameters(LORENZ_PARAMETERS)
    ds.integrator("rk4", time_step=0.01)
    history = ds.lyapunov(U_LORENZ, 20.0, return_history=True)
    final = ds.lyapunov(U_LORENZ, 20.0)
    assert history.shape == (2000, 4)
    np.testing.assert_array_equal(history[-1, 1:], final)
    # The QR and Householder methods share the integration, so they agree
    # to round-off on the exponents.
    np.testing.assert_allclose(ds.lyapunov(U_LORENZ, 20.0, method="QR_HH"), final, atol=1e-12)
