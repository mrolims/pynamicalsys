"""
Periodic orbits, their eigenvalues, and their stability classification.

Fixed points are the one place in a chaotic map where everything is available
in closed form: their position, the Jacobian there, and hence the eigenvalues
and the stability type. That makes this the sharpest test of the linearised
dynamics in the package, and the place where an error in a Jacobian shows up
most plainly -- the Lozi sign error inverted the eigenvalues of its fixed point
and reflected the unstable eigenvector.

Analytic values are written as expressions in the parameters rather than as
decimal literals, so the derivation stays visible and a change of parameters
does not silently invalidate the reference.
"""

import numpy as np
import pytest

from pynamicalsys import DiscreteDynamicalSystem as DDS


HENON = dict(a=1.4, b=0.3)
LOZI = dict(a=1.7, b=0.5)
STANDARD_K = 1.5


def henon_fixed_point(a, b):
    """
    Fixed point of x' = 1 - a x^2 + y, y' = b x with x > 0.

    Substituting y = b x into the first equation gives a x^2 + (1 - b) x - 1 = 0.
    """
    x = (-(1 - b) + np.sqrt((1 - b) ** 2 + 4 * a)) / (2 * a)
    return np.array([x, b * x])


def henon_eigenvalues(a, b):
    """
    Eigenvalues of J = [[-2 a x*, 1], [b, 0]], i.e. of L^2 + 2 a x* L - b = 0.
    """
    x = henon_fixed_point(a, b)[0]
    root = np.sqrt((a * x) ** 2 + b)
    return np.array([-a * x - root, -a * x + root])


def lozi_fixed_point(a, b):
    """
    Fixed point of x' = 1 - a|x| + y, y' = b x with x > 0, where the map is
    linear: x (1 + a - b) = 1.
    """
    x = 1 / (1 + a - b)
    return np.array([x, b * x])


def lozi_eigenvalues(a, b):
    """Eigenvalues of J = [[-a, 1], [b, 0]] for x > 0, i.e. of L^2 + a L - b = 0."""
    root = np.sqrt(a**2 + 4 * b)
    return np.array([(-a - root) / 2, (-a + root) / 2])


def standard_map_hyperbolic_eigenvalues(k):
    """
    At (0, 0) the standard map has J = [[1 + k, 1], [k, 1]], so the eigenvalues
    solve L^2 - (2 + k) L + 1 = 0. Both are real and positive for k > 0.
    """
    trace = 2 + k
    root = np.sqrt(trace**2 - 4)
    return np.array([(trace - root) / 2, (trace + root) / 2])


def search_grid(x_range, y_range, n=50):
    """Build the (n, n, 2) grid of starting points that the search expects."""
    x = np.linspace(*x_range, n)
    y = np.linspace(*y_range, n)
    grid_x, grid_y = np.meshgrid(x, y, indexing="ij")
    return np.stack([grid_x, grid_y], axis=-1)


def system(model, params):
    ds = DDS(model=model)
    ds.set_parameters(params)
    return ds


# --------------------------------------------------------------------------
# Locating periodic orbits
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "model, params, box, expected",
    [
        ("henon map", [HENON["a"], HENON["b"]], ((0.3, 0.9), (0.0, 0.4)),
         henon_fixed_point(**HENON)),
        ("lozi map", [LOZI["a"], LOZI["b"]], ((0.2, 0.7), (0.05, 0.4)),
         lozi_fixed_point(**LOZI)),
    ],
    ids=["henon map", "lozi map"],
)
def test_search_recovers_hyperbolic_fixed_point(model, params, box, expected):
    """
    The grid search must converge onto the analytically known fixed point.

    Note that `tolerance` is the radius used to decide whether a coarse grid
    point looks periodic at all, not the accuracy of the answer. It has to be
    loose enough that some grid point qualifies -- at 1e-3 the search reports
    that it found nothing -- while the accuracy of the result is governed by
    `convergence_threshold`. These converge to round-off.
    """
    found = np.ravel(system(model, params).find_periodic_orbit(
        search_grid(*box), period=1, tolerance=1e-2,
    ))

    np.testing.assert_allclose(
        found[:2], expected, atol=1e-12,
        err_msg=f"{model}: search did not converge on the analytic fixed point",
    )


def test_search_recovers_elliptic_fixed_point():
    """
    The elliptic fixed point of the standard map at (1/2, 0) is located far
    less precisely than a hyperbolic one, and the tolerance reflects that.

    A hyperbolic point is isolated in the sense that nearby orbits leave its
    neighbourhood, so the refinement can shrink onto it. Around an elliptic
    point orbits stay nearby and continue to look almost periodic, so the
    search box stops contracting once it reaches that scale. This converges to
    about 1e-09 rather than 1e-16, which is a property of the problem and not
    a defect of the implementation.
    """
    found = np.ravel(system("standard map", STANDARD_K).find_periodic_orbit(
        search_grid((0.35, 0.65), (-0.15, 0.15)), period=1, tolerance=1e-2,
    ))

    np.testing.assert_allclose(
        found[:2], [0.5, 0.0], atol=1e-7,
        err_msg="standard map: search did not converge on the elliptic fixed point",
    )


# --------------------------------------------------------------------------
# Eigenvalues
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "model, params, point, expected",
    [
        ("henon map", [HENON["a"], HENON["b"]], henon_fixed_point(**HENON),
         henon_eigenvalues(**HENON)),
        ("lozi map", [LOZI["a"], LOZI["b"]], lozi_fixed_point(**LOZI),
         lozi_eigenvalues(**LOZI)),
        ("standard map", [STANDARD_K], np.array([0.0, 0.0]),
         standard_map_hyperbolic_eigenvalues(STANDARD_K)),
    ],
    ids=["henon map", "lozi map", "standard map at origin"],
)
def test_eigenvalues_match_analytic_values(model, params, point, expected):
    """
    The eigenvalues at a fixed point must equal those of the analytic Jacobian.

    This is the check that would have failed on the Lozi map before its
    Jacobian sign was corrected: the computed pair was the negation of the
    correct one, [1.9557, -0.2557] instead of [-1.9557, 0.2557].
    """
    eigenvalues, _ = system(model, params).eigenvalues_and_eigenvectors(point, 1)

    np.testing.assert_allclose(
        np.sort(np.real(eigenvalues)), np.sort(expected), atol=1e-12,
        err_msg=f"{model}: eigenvalues at the fixed point are wrong",
    )
    np.testing.assert_allclose(
        np.imag(eigenvalues), 0.0, atol=1e-12,
        err_msg=f"{model}: eigenvalues at a saddle should be real",
    )


@pytest.mark.parametrize(
    "model, params, point, det, trace",
    [
        ("henon map", [HENON["a"], HENON["b"]], henon_fixed_point(**HENON),
         -HENON["b"], -2 * HENON["a"] * henon_fixed_point(**HENON)[0]),
        ("lozi map", [LOZI["a"], LOZI["b"]], lozi_fixed_point(**LOZI),
         -LOZI["b"], -LOZI["a"]),
        ("standard map", [STANDARD_K], np.array([0.0, 0.0]), 1.0, 2 + STANDARD_K),
        ("standard map", [STANDARD_K], np.array([0.5, 0.0]), 1.0, 2 - STANDARD_K),
    ],
    ids=["henon map", "lozi map", "standard map at origin", "standard map at centre"],
)
def test_eigenvalue_product_and_sum_match_determinant_and_trace(
    model, params, point, det, trace
):
    """
    The eigenvalues must multiply to det J and add to tr J.

    These follow from the characteristic polynomial alone, so they hold whatever
    the eigenvalues turn out to be, and they check the eigen-decomposition
    against the Jacobian independently of any reference value. Both hold to
    round-off.
    """
    eigenvalues, _ = system(model, params).eigenvalues_and_eigenvectors(point, 1)

    np.testing.assert_allclose(
        np.prod(eigenvalues).real, det, atol=1e-12,
        err_msg=f"{model} at {point}: product of eigenvalues should be det J",
    )
    np.testing.assert_allclose(
        np.sum(eigenvalues).real, trace, atol=1e-12,
        err_msg=f"{model} at {point}: sum of eigenvalues should be tr J",
    )


def test_elliptic_fixed_point_has_unit_modulus_eigenvalues():
    """
    At the elliptic fixed point of an area-preserving map the eigenvalues are a
    complex-conjugate pair on the unit circle.

    Area preservation forces their product to be 1, and a conjugate pair with
    product 1 must have modulus 1 exactly. This is a structural consequence, so
    the tolerance is tight.
    """
    eigenvalues, _ = system("standard map", STANDARD_K).eigenvalues_and_eigenvectors(
        [0.5, 0.0], 1
    )

    assert np.max(np.abs(np.imag(eigenvalues))) > 0.1, (
        "standard map at (1/2, 0): eigenvalues should be complex"
    )
    np.testing.assert_allclose(
        np.abs(eigenvalues), 1.0, atol=1e-12,
        err_msg="standard map at (1/2, 0): eigenvalues should lie on the unit circle",
    )


# --------------------------------------------------------------------------
# Stability classification
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "model, params, point, expected_kind",
    [
        ("henon map", [HENON["a"], HENON["b"]], henon_fixed_point(**HENON), "saddle"),
        ("lozi map", [LOZI["a"], LOZI["b"]], lozi_fixed_point(**LOZI), "saddle"),
        ("standard map", [STANDARD_K], np.array([0.0, 0.0]), "saddle"),
        ("standard map", [STANDARD_K], np.array([0.5, 0.0]), "elliptic"),
    ],
    ids=["henon map", "lozi map", "standard map at origin", "standard map at centre"],
)
def test_stability_classification(model, params, point, expected_kind):
    """
    The two fixed points of the standard map have genuinely different characters
    -- a saddle at the origin and an elliptic point at (1/2, 0) -- and the
    classifier must tell them apart rather than reporting the same thing twice.
    """
    result = system(model, params).classify_stability(point, 1)

    assert expected_kind in result["classification"], (
        f"{model} at {point}: expected a {expected_kind}, "
        f"got {result['classification']!r}"
    )


# --------------------------------------------------------------------------
# Periods
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "r, expected_period",
    [(2.5, 1), (3.2, 2), (3.5, 4), (3.55, 8)],
)
def test_period_follows_the_period_doubling_cascade(r, expected_period):
    """
    The logistic map's attracting cycle doubles in length at each bifurcation,
    giving periods 1, 2, 4 and 8 in the four windows sampled here. Detecting
    the wrong one is a far more legible failure than a numerical discrepancy.
    """
    detected = system("logistic map", r).period(
        [0.3], 20_000, transient_time=10_000,
    )

    assert detected == expected_period, (
        f"logistic map at r={r}: detected period {detected}, expected {expected_period}"
    )


def test_two_cycle_of_the_logistic_map_matches_the_analytic_values():
    """
    The period-2 orbit born at r = 3 satisfies r^2 x^2 - r(r + 1) x + (r + 1) = 0,
    so its two points are (r + 1 +/- sqrt((r - 3)(r + 1))) / 2r.

    This pins the location of the cycle, not merely its length.
    """
    r = 3.2
    root = np.sqrt((r - 3) * (r + 1))
    expected = np.sort([(r + 1 + root) / (2 * r), (r + 1 - root) / (2 * r)])

    tail = system("logistic map", r).trajectory([0.3], 20_000, transient_time=19_990)
    visited = np.sort(np.unique(np.round(np.ravel(tail), 9)))

    assert len(visited) == 2, f"expected a 2-cycle, visited {len(visited)} points"
    np.testing.assert_allclose(
        visited, expected, atol=1e-9,
        err_msg="logistic map at r=3.2: 2-cycle is not at the analytic locations",
    )


@pytest.mark.parametrize("r, period", [(2.5, 1), (3.2, 2), (3.5, 4)])
def test_is_periodic_agrees_with_period(r, period):
    """
    `is_periodic` and `period` are separate entry points and must not disagree:
    the period reported by one has to be accepted by the other.
    """
    ds = system("logistic map", r)

    assert ds.is_periodic([0.3], period, transient_time=10_000), (
        f"logistic map at r={r}: is_periodic rejected its own period {period}"
    )
