"""
Structural properties of the built-in models in `pynamicalsys.discrete_time`.

Covers the discrete-time maps only. The continuous-time and Hamiltonian
models have their own directories alongside this one.

These tests do not check what a trajectory *is* -- for a chaotic map that is
not a reproducible quantity, since nearby orbits separate exponentially. They
check identities that must hold at every point in phase space, to machine
precision, whatever the dynamics does. Such properties are cheap to test and
catch the kind of algebra error a trajectory plot will happily hide.
"""

import numpy as np
import pytest

from pynamicalsys import DiscreteDynamicalSystem as DDS


# One table describing every model under test, so that adding a model is a
# single new entry rather than an edit to each test.
#
#   params      parameter values to test at
#   periods     wrapping period of each coordinate; np.inf where the
#               coordinate is unbounded. Taken from the `%` operations in
#               discrete_time/models.py.
#   box         sampling range per coordinate
#   det         analytic value of the Jacobian determinant
#   invertible  whether the model provides a backward mapping
TWO_PI = 2 * np.pi

MODELS = {
    "standard map": dict(
        params=[1.5], periods=[1.0, 1.0], box=[(0.0, 1.0), (0.0, 1.0)],
        det=1.0, invertible=True,
    ),
    "standard nontwist map": dict(
        params=[0.3, 0.6], periods=[1.0, np.inf], box=[(0.0, 1.0), (-1.0, 1.0)],
        det=1.0, invertible=True,
    ),
    "extended standard nontwist map": dict(
        params=[0.3, 0.6, 0.1, 3.0], periods=[1.0, np.inf],
        box=[(0.0, 1.0), (-1.0, 1.0)], det=1.0, invertible=True,
    ),
    "leonel map": dict(
        # y is kept away from zero: the map contains 1 / |y|**gamma.
        params=[0.5, 1.0], periods=[TWO_PI, np.inf],
        box=[(0.0, TWO_PI), (0.5, 2.0)], det=1.0, invertible=True,
    ),
    "4d symplectic map": dict(
        params=[0.5, 0.3, 0.1], periods=[TWO_PI] * 4, box=[(0.0, TWO_PI)] * 4,
        det=1.0, invertible=True,
    ),
    "henon map": dict(
        params=[1.4, 0.3], periods=[np.inf, np.inf], box=[(-1.0, 1.0), (-1.0, 1.0)],
        det=-0.3, invertible=False,
    ),
    "lozi map": dict(
        # x is kept away from zero: the map contains |x|, so its derivative
        # genuinely does not exist on x = 0 and a finite difference across the
        # kink is meaningless.
        params=[1.7, 0.5], periods=[np.inf, np.inf], box=[(0.1, 1.0), (-1.0, 1.0)],
        det=-0.5, invertible=False,
    ),
}

ALL_MODELS = sorted(MODELS)
INVERTIBLE_MODELS = sorted(m for m in MODELS if MODELS[m]["invertible"])


def circular_difference(a, b, periods):
    """
    Difference `a - b`, reduced to (-P/2, P/2] on coordinates of period P.

    Coordinates that wrap onto a circle have many equivalent representations:
    x and x + P are the same point. Subtracting them naively reports a
    difference of P where the true difference is zero, so wrapped coordinates
    must be compared on the circle. Coordinates with period `np.inf` do not
    wrap and are compared directly.
    """
    difference = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    for i, period in enumerate(periods):
        if np.isfinite(period):
            difference[i] = (difference[i] + period / 2) % period - period / 2
    return difference


def sample_states(rng, model, n=200):
    """Draw `n` states uniformly from the model's sampling box."""
    box = MODELS[model]["box"]
    low = np.array([lo for lo, _ in box])
    high = np.array([hi for _, hi in box])
    return low + rng.random((n, len(box))) * (high - low)


@pytest.mark.parametrize("model", INVERTIBLE_MODELS)
def test_backward_map_inverts_forward_map(rng, model):
    """
    The backward map applied to the image of the forward map must return the
    original state.

    This is the defining property of an inverse, so any deviation beyond
    round-off is an algebra error in one of the two functions.
    """
    spec = MODELS[model]
    p = np.asarray(spec["params"], dtype=float)
    forward = DDS(model=model).info["mapping"]
    backward = DDS(model=model).info["backwards_mapping"]

    worst = 0.0
    for u in sample_states(rng, model):
        recovered = backward(forward(u, p), p)
        worst = max(worst, np.max(np.abs(
            circular_difference(recovered, u, spec["periods"])
        )))

    assert worst < 1e-12, (
        f"{model}: backward map is not the inverse of the forward map "
        f"(worst round-trip error {worst:.3e})"
    )


@pytest.mark.parametrize("model", ALL_MODELS)
def test_jacobian_determinant_is_constant(rng, model):
    """
    The Jacobian determinant must equal its analytic value everywhere.

    For the area-preserving maps this is the symplectic condition det J = 1;
    the Henon and Lozi maps contract phase-space areas by exactly b. A
    determinant that drifts from the expected constant means the Jacobian does
    not describe the map it claims to differentiate.
    """
    spec = MODELS[model]
    p = np.asarray(spec["params"], dtype=float)
    jacobian = DDS(model=model).info["jacobian"]

    determinants = [np.linalg.det(jacobian(u, p)) for u in sample_states(rng, model)]

    np.testing.assert_allclose(
        determinants, spec["det"], atol=1e-12,
        err_msg=f"{model}: Jacobian determinant is not {spec['det']} everywhere",
    )


@pytest.mark.parametrize("model", ALL_MODELS)
def test_analytic_jacobian_matches_finite_differences(rng, model):
    """
    The hand-written Jacobian must agree with a numerical derivative of the map.

    The determinant test above would still pass if a Jacobian were wrong in a
    way that happened to preserve its determinant, so the two are checked
    directly against each other here. The tolerance is loose because the
    reference is a second-order central difference, not because the Jacobian is
    expected to be imprecise.
    """
    spec = MODELS[model]
    p = np.asarray(spec["params"], dtype=float)
    mapping = DDS(model=model).info["mapping"]
    jacobian = DDS(model=model).info["jacobian"]
    dim = len(spec["box"])
    h = 1e-6

    for u in sample_states(rng, model, n=20):
        numerical = np.empty((dim, dim))
        for j in range(dim):
            step = np.zeros(dim)
            step[j] = h
            # The map output wraps too, so the two evaluations can land on
            # opposite sides of a branch cut; difference them on the circle.
            numerical[:, j] = circular_difference(
                mapping(u + step, p), mapping(u - step, p), spec["periods"]
            ) / (2 * h)

        np.testing.assert_allclose(
            jacobian(u, p), numerical, rtol=1e-5, atol=1e-7,
            err_msg=f"{model}: analytic Jacobian disagrees with finite differences",
        )
