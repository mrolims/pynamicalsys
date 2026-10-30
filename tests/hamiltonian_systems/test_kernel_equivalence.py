"""
The optimized Hamiltonian kernels against straightforward reference versions.

The references are the earlier implementations, which build every array with
array expressions. The symplectic and implicit midpoint steps must give the
same bits. The tangent Verlet step multiplies by the Hessians with an explicit
loop instead of BLAS, so it agrees to round-off.
"""

import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest
from numba import njit

from pynamicalsys import HamiltonianSystem as HS
from pynamicalsys.hamiltonian_systems.coefficients import ALPHA, BETA
from pynamicalsys.hamiltonian_systems.fixed_step import (
    _symplectic_J_inv_matmul,
    implicit_midpoint_step,
    velocity_verlet_2nd_step,
    yoshida_4th_step,
)
from pynamicalsys.hamiltonian_systems.models import (
    henon_heiles_eom,
    henon_heiles_grad_T,
    henon_heiles_grad_V,
    henon_heiles_hess_H,
    henon_heiles_hess_T,
    henon_heiles_hess_V,
)
from pynamicalsys.hamiltonian_systems.tangent import (
    implicit_midpoint_step_traj_tan,
    yoshida_4th_step_traj_tan,
)

PARAMETERS = np.zeros(1)
ORBITS = [
    (np.array([0.0, 0.1]), np.array([0.5, 0.0])),
    (np.array([0.0, -0.25]), np.array([0.5, 0.0])),
]


@njit
def reference_verlet(q, p, h, grad_T, grad_V, parameters):
    q_new = q.copy()
    p_new = p.copy()
    p_new -= np.float64(0.5) * h * grad_V(q, parameters)
    q_new += h * grad_T(p_new, parameters)
    p_new -= np.float64(0.5) * h * grad_V(q_new, parameters)
    return q_new, p_new


@njit
def reference_tangent_verlet(q, p, dv, h, grad_T, grad_V, hess_T, hess_V, parameters):
    q_new = q.copy()
    p_new = p.copy()
    dv_new = dv.copy()
    dof = len(q)
    p_new -= np.float64(0.5) * h * grad_V(q_new, parameters)
    HV = hess_V(q_new, parameters)
    dv_new[dof:, :] -= np.float64(0.5) * h * (HV @ np.ascontiguousarray(dv_new[:dof, :]))
    q_new += h * grad_T(p_new, parameters)
    HT = hess_T(p_new, parameters)
    dv_new[:dof, :] += h * (HT @ np.ascontiguousarray(dv_new[dof:, :]))
    p_new -= np.float64(0.5) * h * grad_V(q_new, parameters)
    HV = hess_V(q_new, parameters)
    dv_new[dof:, :] -= np.float64(0.5) * h * (HV @ np.ascontiguousarray(dv_new[:dof, :]))
    return q_new, p_new, dv_new


@njit
def reference_midpoint(q, p, h, eom, hess_H, parameters, tol, max_iter):
    n = q.shape[0]
    z_n = np.concatenate((q, p))
    q_bar = q.copy()
    p_bar = p.copy()
    identity = np.eye(2 * n)
    Jinv_Hmid = np.zeros((2 * n, 2 * n))
    DF = identity.copy()
    for _ in range(max_iter):
        qdot, pdot = eom(q_bar, p_bar, parameters)
        z_bar = np.concatenate((q_bar, p_bar))
        F = z_bar - z_n - np.float64(0.5) * h * np.concatenate((qdot, pdot))
        residual_norm = np.sqrt(np.sum(F * F))
        Jinv_Hmid = _symplectic_J_inv_matmul(n, hess_H(q_bar, p_bar, parameters))
        DF = identity - np.float64(0.5) * h * Jinv_Hmid
        if residual_norm < tol:
            break
        z_bar = z_bar + np.linalg.solve(DF, -F)
        q_bar = z_bar[:n]
        p_bar = z_bar[n:]
    return np.float64(2.0) * q_bar - q, np.float64(2.0) * p_bar - p, DF, Jinv_Hmid


@njit
def run_verlet(stepper, q, p, steps, h):
    for _ in range(steps):
        q, p = stepper(q, p, h, henon_heiles_grad_T, henon_heiles_grad_V, PARAMETERS)
    return q, p


@njit
def run_reference_yoshida(q, p, steps, h):
    for _ in range(steps):
        q, p = reference_verlet(q, p, np.float64(ALPHA) * h, henon_heiles_grad_T, henon_heiles_grad_V, PARAMETERS)
        q, p = reference_verlet(q, p, np.float64(BETA) * h, henon_heiles_grad_T, henon_heiles_grad_V, PARAMETERS)
        q, p = reference_verlet(q, p, np.float64(ALPHA) * h, henon_heiles_grad_T, henon_heiles_grad_V, PARAMETERS)
    return q, p


@njit
def run_yoshida(q, p, steps, h):
    for _ in range(steps):
        q, p = yoshida_4th_step(q, p, h, henon_heiles_grad_T, henon_heiles_grad_V, PARAMETERS)
    return q, p


@pytest.mark.parametrize("q, p", ORBITS)
def test_symplectic_steps_are_bit_identical(q, p):
    result = run_verlet(velocity_verlet_2nd_step, q, p, 2_000, 0.01)
    expected = run_verlet(reference_verlet, q, p, 2_000, 0.01)
    np.testing.assert_array_equal(result[0], expected[0])
    np.testing.assert_array_equal(result[1], expected[1])

    result = run_yoshida(q, p, 2_000, 0.01)
    expected = run_reference_yoshida(q, p, 2_000, 0.01)
    np.testing.assert_array_equal(result[0], expected[0])
    np.testing.assert_array_equal(result[1], expected[1])


@pytest.mark.parametrize("q, p", ORBITS)
def test_implicit_midpoint_steps_are_bit_identical(q, p):
    dv = np.linalg.qr(np.random.default_rng(1).uniform(-1, 1, (4, 3)))[0]
    dv = np.ascontiguousarray(dv)
    for _ in range(200):
        q_new, p_new = implicit_midpoint_step(
            q, p, 0.05, henon_heiles_eom, henon_heiles_hess_H, PARAMETERS
        )
        q_tan, p_tan, dv_new = implicit_midpoint_step_traj_tan(
            q, p, dv, 0.05, henon_heiles_eom, henon_heiles_hess_H, PARAMETERS
        )
        q_ref, p_ref, DF, Jinv_Hmid = reference_midpoint(
            q, p, 0.05, henon_heiles_eom, henon_heiles_hess_H, PARAMETERS, 1e-12, 50
        )
        dv_ref = np.linalg.solve(DF, (np.eye(4) + 0.5 * 0.05 * Jinv_Hmid) @ dv)
        for a, b in [(q_new, q_ref), (p_new, p_ref), (q_tan, q_ref), (p_tan, p_ref), (dv_new, dv_ref)]:
            np.testing.assert_array_equal(a, b)
        q, p, dv = q_new, p_new, np.ascontiguousarray(dv_new)


@pytest.mark.parametrize("q, p", ORBITS)
@pytest.mark.parametrize("k", [1, 2, 4])
def test_tangent_yoshida_step_agrees_to_round_off(q, p, k):
    dv = np.ascontiguousarray(np.random.default_rng(2).uniform(-1, 1, (4, k)))
    q_ref, p_ref, dv_ref = q, p, dv
    for _ in range(100):
        q, p, dv = yoshida_4th_step_traj_tan(
            q, p, dv, 0.01, henon_heiles_grad_T, henon_heiles_grad_V,
            henon_heiles_hess_T, henon_heiles_hess_V, PARAMETERS,
        )
        for factor in (ALPHA, BETA, ALPHA):
            q_ref, p_ref, dv_ref = reference_tangent_verlet(
                q_ref, p_ref, dv_ref, np.float64(factor) * 0.01,
                henon_heiles_grad_T, henon_heiles_grad_V,
                henon_heiles_hess_T, henon_heiles_hess_V, PARAMETERS,
            )
    # The state does not depend on the tangent vectors, so it stays identical.
    np.testing.assert_array_equal(q, q_ref)
    np.testing.assert_array_equal(p, p_ref)
    np.testing.assert_allclose(dv, dv_ref, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("integrator", ["svy4", "vv2", "imp"])
def test_ensemble_poincare_section_matches_single_sections(integrator):
    ds = HS(model="henon heiles")
    ds.integrator(integrator, time_step=0.01)
    q = np.zeros((5, 2))
    q[:, 1] = np.linspace(-0.2, 0.2, 5)
    p = np.zeros((5, 2))
    p[:, 0] = 0.3
    sections = ds.poincare_section(q, p, 20, max_workers=3)
    assert sections.shape == (5, 20, 5)
    for i in range(5):
        np.testing.assert_array_equal(sections[i], ds.poincare_section(q[i], p[i], 20))


def test_ensemble_poincare_section_runs_without_a_main_guard(tmp_path):
    # With processes started by spawn, an unguarded script used to fail.
    script = tmp_path / "unguarded.py"
    script.write_text(textwrap.dedent(
        """
        import numpy as np
        from pynamicalsys import HamiltonianSystem as HS
        ds = HS(model="henon heiles")
        q = np.zeros((3, 2)); q[:, 1] = [-0.1, 0.0, 0.1]
        p = np.zeros((3, 2)); p[:, 0] = 0.3
        print(ds.poincare_section(q, p, 5).shape)
        """
    ))
    src = os.path.join(os.path.dirname(__file__), "..", "..", "src")
    env = dict(os.environ, PYTHONPATH=os.path.abspath(src))
    result = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, env=env, timeout=600
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "(3, 5, 5)"
