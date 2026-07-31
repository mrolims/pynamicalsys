"""
Lyapunov exponents of the built-in discrete-time maps.

Three different kinds of check appear here, and it is worth being explicit
about which is which, because they justify very different tolerances.

1. Exact analytic references. Where an orbit settles onto a fixed point the
   exponent is known in closed form and the computation reproduces it to
   round-off, so the tolerance can be tight.

2. Invariants. Quantities such as the sum of the spectrum are fixed by the
   structure of the map and hold to machine precision *regardless* of how well
   the individual exponents have converged. These are the strongest tests
   available for a chaotic system and the tolerance can be tight even when the
   exponents themselves are still drifting in the fourth decimal.

3. Reference values for chaotic orbits. These converge slowly and not
   monotonically, because a finite-time exponent is a fluctuating average.
   A longer run is not reliably a more accurate one, so the tolerance here is
   loose by necessity, not by sloppiness.
"""

import numpy as np
import pytest

from pynamicalsys import DiscreteDynamicalSystem as DDS


# (model, parameters, initial condition, analytic det J).
#
# The sum of the Lyapunov spectrum equals ln|det J| whenever the determinant is
# constant: the area-preserving maps have det J = 1 and therefore a spectrum
# summing to zero, while Henon and Lozi contract by exactly b every step.
#
# The determinants are written out rather than read from the model's own
# Jacobian on purpose. A reference that is computed from the thing under test
# is not a reference; test_models.py checks the Jacobians against finite
# differences separately.
CONSTANT_DET_MODELS = [
    ("standard map", [1.5], [0.1, 0.2], 1.0),
    ("standard nontwist map", [0.3, 0.6], [0.1, 0.2], 1.0),
    ("extended standard nontwist map", [0.3, 0.6, 0.1, 3.0], [0.1, 0.2], 1.0),
    ("leonel map", [0.5, 1.0], [0.1, 1.0], 1.0),
    ("4d symplectic map", [0.5, 0.3, 0.1], [0.1, 0.2, 0.3, 0.4], 1.0),
    ("henon map", [1.4, 0.3], [0.1, 0.1], -0.3),
    ("lozi map", [1.7, 0.5], [0.1, 0.1], -0.5),
]

# Two-dimensional maps, where the analytic Eckmann-Ruelle method is available
# alongside the two QR implementations.
TWO_DIMENSIONAL_MODELS = [
    ("standard map", [1.5], [0.1, 0.2]),
    ("standard nontwist map", [0.3, 0.6], [0.1, 0.2]),
    ("henon map", [1.4, 0.3], [0.1, 0.1]),
    ("lozi map", [1.7, 0.5], [0.1, 0.1]),
]


def exponents(model, params, u, total_time, **kwargs):
    """Run `lyapunov` and always return a 1D array, whatever the arity."""
    ds = DDS(model=model)
    ds.set_parameters(params)
    return np.atleast_1d(np.ravel(ds.lyapunov(u, total_time, **kwargs)))


# --------------------------------------------------------------------------
# 1. Exact analytic references
# --------------------------------------------------------------------------

@pytest.mark.parametrize("r", [1.5, 2.5, 2.9])
def test_logistic_map_fixed_point_exponent(r):
    """
    On 1 < r < 3 the logistic map has an attracting fixed point x* = 1 - 1/r,
    where the derivative is r(1 - 2x*) = 2 - r. The Lyapunov exponent of an
    orbit that settles there is therefore exactly ln|2 - r|.

    Because the orbit converges onto the fixed point, this is not a statistical
    estimate and the agreement is at round-off level, so the tolerance is tight.
    """
    got = exponents("logistic map", r, [0.3], 20_000, transient_time=10_000)

    np.testing.assert_allclose(
        got, [np.log(abs(2 - r))], atol=1e-9,
        err_msg=f"logistic map at r={r}: exponent should be ln|2-r|",
    )


# --------------------------------------------------------------------------
# 2. Invariants
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "model, params, u, det", CONSTANT_DET_MODELS,
    ids=[case[0] for case in CONSTANT_DET_MODELS],
)
def test_sum_of_exponents_equals_log_det_jacobian(model, params, u, det):
    """
    The Lyapunov spectrum sums to ln|det J|.

    Each step multiplies phase-space volume by det J, and the spectrum measures
    the exponential growth rate of that volume, so the identity is structural.
    It holds far more accurately than any individual exponent: for the standard
    map at these settings the exponents are still around 5e-05 and drifting,
    while their sum is zero to about 1e-19.

    That gap is the reason to test invariants rather than values. This test is
    tight and meaningful at a run length where a test of the exponents
    themselves would have to be too loose to catch anything.
    """
    spectrum = exponents(model, params, u, 100_000, transient_time=1_000)

    assert len(spectrum) == len(u), (
        f"{model}: expected {len(u)} exponents, got {len(spectrum)}"
    )
    np.testing.assert_allclose(
        spectrum.sum(), np.log(abs(det)), atol=1e-9,
        err_msg=f"{model}: sum of the spectrum should equal ln|det J| = ln{abs(det)}",
    )


@pytest.mark.parametrize(
    "model, params, u", TWO_DIMENSIONAL_MODELS,
    ids=[case[0] for case in TWO_DIMENSIONAL_MODELS],
)
def test_requesting_one_exponent_matches_the_full_spectrum(model, params, u):
    """
    Asking for a single exponent must give the same number as the largest of
    the full spectrum.

    These take different code paths internally -- `maximum_lyapunov_er` versus
    `lyapunov_er` -- so this is a consistency check between two independent
    implementations of the same quantity, both driven by the same seed. They
    agree exactly, so the tolerance is tight.
    """
    common = dict(method="ER", transient_time=1_000)
    largest = exponents(model, params, u, 50_000, num_exponents=1, **common)
    full = exponents(model, params, u, 50_000, **common)

    np.testing.assert_allclose(
        largest[0], full[0], atol=1e-12,
        err_msg=f"{model}: num_exponents=1 disagrees with the full spectrum",
    )


@pytest.mark.parametrize(
    "model, params, u", TWO_DIMENSIONAL_MODELS,
    ids=[case[0] for case in TWO_DIMENSIONAL_MODELS],
)
def test_lyapunov_methods_agree(model, params, u):
    """
    The three available methods must produce the same spectrum.

    "ER" is the analytic Eckmann-Ruelle scheme for two-dimensional maps, "QR"
    uses the package's own modified Gram-Schmidt routine, and "QR_HH" defers to
    the Householder QR in NumPy. They are independent implementations, so
    agreement is real evidence rather than a tautology.

    QR and QR_HH agree to round-off. ER accumulates its rounding differently and
    departs by up to a few times 1e-05 at this run length, which sets the
    tolerance.
    """
    results = {
        method: exponents(model, params, u, 50_000, method=method,
                          transient_time=1_000)
        for method in ("ER", "QR", "QR_HH")
    }

    np.testing.assert_allclose(
        results["QR"], results["QR_HH"], atol=1e-12,
        err_msg=f"{model}: the two QR implementations disagree",
    )
    np.testing.assert_allclose(
        results["ER"], results["QR"], atol=1e-3,
        err_msg=f"{model}: Eckmann-Ruelle disagrees with QR",
    )


def test_er_method_rejected_above_two_dimensions():
    """
    The Eckmann-Ruelle method is derived for two-dimensional maps, and asking
    for it on a larger system must fail loudly rather than return something
    plausible.
    """
    ds = DDS(model="4d symplectic map")
    ds.set_parameters([0.5, 0.3, 0.1])

    with pytest.raises(ValueError, match="only valid for 2 dimensional"):
        ds.lyapunov([0.1, 0.2, 0.3, 0.4], 1_000, method="ER")


# --------------------------------------------------------------------------
# 3. Reference values for chaotic orbits
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "model, params, u, reference",
    [
        # The logistic map at r = 4 is conjugate to the tent map, whose
        # exponent is ln 2 exactly. The orbit is chaotic, so the computed value
        # is a fluctuating average that only approaches it.
        ("logistic map", 4.0, [0.1234], [np.log(2)]),
        # Henon at the classical parameters; widely quoted values.
        ("henon map", [1.4, 0.3], [0.1, 0.1], [0.41922, -1.62319]),
    ],
    ids=["logistic map r=4", "henon map"],
)
def test_reference_values_for_chaotic_orbits(model, params, u, reference):
    """
    Chaotic orbits reproduce their published exponents to about three decimals.

    The tolerance is loose on purpose. A finite-time Lyapunov exponent is an
    average over a fluctuating quantity, so its error does not fall off
    smoothly with run length: for the Henon map the error here is around 6e-05
    at 100000 iterations but around 8e-04 at 400000. Tightening the tolerance
    would buy no extra sensitivity and would make the test fail at random.

    These are not marked slow. Almost all of this file's runtime is Numba
    compiling a fresh specialisation of the tangent-space kernels for each
    map, not the iteration itself, so raising or lowering the iteration count
    barely moves the total.
    """
    got = exponents(model, params, u, 100_000, transient_time=1_000)

    np.testing.assert_allclose(
        got, reference, atol=2e-3,
        err_msg=f"{model}: exponents differ from the published values",
    )
