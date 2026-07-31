"""
The Newton solver in `find_periodic_orbit`.

Passing a 1D initial guess with no symmetry line selects Newton's method, which
refines a single point rather than searching a region. It solves
`F^p(u) - u = 0` using the monodromy matrix as the derivative, so unlike the
grid refinement it places no restriction on the system dimension and converges
quadratically at elliptic and hyperbolic orbits alike.

The tests below fall into four groups: that it works in dimensions the grid
search cannot handle, that it agrees with the grid search where both apply,
that its documented behaviour on a torus and at lower-period solutions holds,
and that the paths that existed before are untouched.
"""

import numpy as np
import pytest

from pynamicalsys import DiscreteDynamicalSystem as DDS


TWO_PI = 2 * np.pi

HENON_A, HENON_B = 1.4, 0.3
LOZI_A, LOZI_B = 1.7, 0.5
STANDARD_K = 1.5


def system(model, params):
    ds = DDS(model=model)
    ds.set_parameters(params)
    return ds


def henon_fixed_point(a=HENON_A, b=HENON_B):
    """Root of a x^2 + (1 - b) x - 1 = 0 with x > 0, and y = b x."""
    x = (-(1 - b) + np.sqrt((1 - b) ** 2 + 4 * a)) / (2 * a)
    return np.array([x, b * x])


def lozi_fixed_point(a=LOZI_A, b=LOZI_B):
    """For x > 0 the Lozi map is linear, so x (1 + a - b) = 1."""
    x = 1 / (1 + a - b)
    return np.array([x, b * x])


def search_grid(x_range, y_range, n=50):
    x = np.linspace(*x_range, n)
    y = np.linspace(*y_range, n)
    grid_x, grid_y = np.meshgrid(x, y, indexing="ij")
    return np.stack([grid_x, grid_y], axis=-1)


def circular_difference(a, b, period):
    """Difference reduced onto (-P/2, P/2], for comparing points on a torus."""
    return (np.asarray(a) - np.asarray(b) + period / 2) % period - period / 2


# --------------------------------------------------------------------------
# Dimensions the grid search cannot reach
# --------------------------------------------------------------------------

@pytest.mark.parametrize("r", [2.5, 3.2])
def test_one_dimensional_system(r):
    """
    A one-dimensional map has a fixed point at x* = 1 - 1/r.

    The grid search rejects anything that is not two-dimensional, so before
    Newton there was no way to ask this question through the public API at all.
    """
    found = np.ravel(system("logistic map", r).find_periodic_orbit([0.4], 1))

    np.testing.assert_allclose(
        found, [1 - 1 / r], atol=1e-12,
        err_msg=f"logistic map at r={r}: wrong fixed point",
    )


def test_four_dimensional_system():
    """
    With the coupling switched off the 4D symplectic map reduces to two
    uncoupled standard maps, so the origin is exactly a fixed point.

    This is the case that motivated the work: four dimensions were previously
    out of reach entirely.
    """
    found = np.ravel(
        system("4d symplectic map", [0.5, 0.3, 0.0]).find_periodic_orbit(
            [0.08, 0.03, 0.08, 0.03], 1, periods=np.full(4, TWO_PI),
        )
    )

    assert found.shape == (4,), f"expected 4 coordinates, got {found.shape}"
    np.testing.assert_allclose(
        circular_difference(found, np.zeros(4), TWO_PI), 0.0, atol=1e-12,
        err_msg="4d symplectic map: origin should be a fixed point at xi = 0",
    )


def test_four_dimensional_system_with_coupling():
    """
    With the coupling on there is no closed form for the fixed point, so the
    check is that the point returned really is one: F(u) = u on the torus.

    A residual test is weaker than comparing against a known answer, but it is
    the honest thing to assert when no known answer exists.
    """
    parameters = [0.5, 0.3, 0.1]
    ds = system("4d symplectic map", parameters)

    found = np.ravel(ds.find_periodic_orbit(
        [0.1, 0.05, 0.1, 0.05], 1, periods=np.full(4, TWO_PI),
    ))
    image = np.ravel(ds.step(found))

    np.testing.assert_allclose(
        circular_difference(image, found, TWO_PI), 0.0, atol=1e-12,
        err_msg="4d symplectic map: the point returned is not a fixed point",
    )


def test_newton_works_without_an_analytic_jacobian():
    """
    A custom mapping supplied without a Jacobian falls back to finite
    differences, and Newton has to work with that too, since a user defining
    their own map is the most likely reason to need a dimension the grid search
    does not support.
    """
    from numba import njit

    @njit
    def henon(u, parameters):
        a, b = parameters
        x, y = u
        return np.array([1 - a * x * x + y, b * x])

    ds = DDS(
        mapping=henon, system_dimension=2, parameters=[HENON_A, HENON_B],
    )
    found = np.ravel(ds.find_periodic_orbit([0.5, 0.2], 1))

    np.testing.assert_allclose(
        found, henon_fixed_point(), atol=1e-10,
        err_msg="custom map with a finite-difference Jacobian: wrong fixed point",
    )


# --------------------------------------------------------------------------
# Agreement with the existing search, and accuracy
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "model, params, box, guess, expected",
    [
        ("henon map", [HENON_A, HENON_B], ((0.3, 0.9), (0.0, 0.4)),
         [0.5, 0.2], henon_fixed_point()),
        ("lozi map", [LOZI_A, LOZI_B], ((0.2, 0.7), (0.05, 0.4)),
         [0.4, 0.2], lozi_fixed_point()),
    ],
    ids=["henon map", "lozi map"],
)
def test_newton_and_grid_search_agree(model, params, box, guess, expected):
    """
    Where both solvers apply they must find the same orbit.

    They share no code beyond the mapping itself -- one refines a box by
    repeated scanning, the other solves a linear system against the monodromy
    matrix -- so agreement is evidence rather than a tautology.
    """
    ds = system(model, params)

    from_grid = np.ravel(ds.find_periodic_orbit(
        search_grid(*box), 1, tolerance=1e-2,
    ))
    from_newton = np.ravel(ds.find_periodic_orbit(guess, 1))

    np.testing.assert_allclose(
        from_newton, from_grid, atol=1e-10,
        err_msg=f"{model}: the two solvers disagree",
    )
    np.testing.assert_allclose(
        from_newton, expected, atol=1e-12,
        err_msg=f"{model}: Newton did not reach the analytic fixed point",
    )


def test_newton_is_far_more_accurate_at_an_elliptic_orbit():
    """
    At the elliptic fixed point of the standard map, Newton reaches round-off
    while the grid refinement stalls about seven orders of magnitude short.

    Near a hyperbolic point the refinement contracts because nearby orbits
    leave the neighbourhood. Near an elliptic point they stay close and keep
    looking almost periodic, so the search box stops shrinking. Newton is
    indifferent to the distinction, which is the practical reason to prefer it
    when a guess is available.

    The assertion is deliberately loose -- it demands three orders of magnitude,
    not the seven observed -- so that it documents the qualitative gap without
    breaking on an unrelated change to the refinement schedule.
    """
    ds = system("standard map", STANDARD_K)
    centre = np.array([0.5, 0.0])

    from_grid = np.ravel(ds.find_periodic_orbit(
        search_grid((0.35, 0.65), (-0.15, 0.15)), 1, tolerance=1e-2,
    ))
    from_newton = np.ravel(ds.find_periodic_orbit(
        [0.48, 0.02], 1, periods=[1.0, 1.0],
    ))

    grid_error = np.max(np.abs(circular_difference(from_grid, centre, 1.0)))
    newton_error = np.max(np.abs(circular_difference(from_newton, centre, 1.0)))

    assert newton_error < 1e-12, (
        f"Newton should reach round-off at the elliptic point, got {newton_error:.2e}"
    )
    assert newton_error < grid_error / 1e3, (
        f"Newton ({newton_error:.2e}) should be far more accurate than the grid "
        f"search ({grid_error:.2e}) at an elliptic point"
    )


def test_period_two_orbit_of_the_henon_map():
    """
    The two points of the Hénon 2-cycle satisfy x0 + x1 = (1 - b)/a and
    x0 x1 = ((1 - b)^2 - a)/a^2, obtained by eliminating y from the period-2
    conditions and dividing out the fixed-point solutions.

    Checking the symmetric functions rather than the roots individually avoids
    depending on which point of the cycle Newton happens to return.
    """
    a, b = HENON_A, HENON_B
    ds = system("henon map", [a, b])

    first = np.ravel(ds.find_periodic_orbit([-0.5, 0.3], 2))
    second = np.ravel(ds.step(first))

    assert np.max(np.abs(second - first)) > 1e-6, (
        "expected a genuine 2-cycle, got a fixed point"
    )
    np.testing.assert_allclose(
        np.ravel(ds.step(second)), first, atol=1e-12,
        err_msg="the orbit does not close after two steps",
    )
    np.testing.assert_allclose(
        [first[0] + second[0], first[0] * second[0]],
        [(1 - b) / a, ((1 - b) ** 2 - a) / a**2],
        atol=1e-12,
        err_msg="the 2-cycle is not at the analytic locations",
    )


# --------------------------------------------------------------------------
# Documented behaviour on a torus and at lower-period solutions
# --------------------------------------------------------------------------

def test_orbit_on_a_torus_is_returned_near_the_initial_guess():
    """
    Newton steps freely in R^n, so an iterate can leave the fundamental domain:
    this period-3 orbit converges to x = 3.0 before reduction. The point
    returned is the representative nearest the guess, so it comes back at
    x = 0.0.

    Reducing onto [0, 1) instead would be worse. The elliptic fixed point
    converges to y ~ -1e-17, and y % 1.0 reports that as 1 - 1e-17, at the far
    edge of the domain.
    """
    ds = system("standard map", STANDARD_K)

    found = np.ravel(ds.find_periodic_orbit([0.3, 0.35], 3, periods=[1.0, 1.0]))
    assert np.all(np.abs(found) < 1.5), (
        f"the point returned should sit near the guess, got {found}"
    )

    orbit = found.copy()
    for _ in range(3):
        orbit = np.ravel(ds.step(orbit))
    np.testing.assert_allclose(
        circular_difference(orbit, found, 1.0), 0.0, atol=1e-12,
        err_msg="the period-3 orbit does not close",
    )

    centre = np.ravel(ds.find_periodic_orbit([0.48, 0.02], 1, periods=[1.0, 1.0]))
    assert abs(centre[1]) < 1e-12, (
        f"a coordinate converging to ~0 should not be reported near 1, got {centre}"
    )


def test_lower_period_solutions_are_returned_unless_prime_period_is_set():
    """
    Every point of period dividing p solves F^p(u) = u, so a fixed point is a
    valid solution of the period-2 equation and Newton may converge to it from
    a nearby guess. That is not a failure of the solver, so it is returned by
    default; `prime_period=True` turns it into an error instead.
    """
    ds = system("henon map", [HENON_A, HENON_B])
    guess = [0.5, 0.2]

    found = np.ravel(ds.find_periodic_orbit(guess, 2))
    np.testing.assert_allclose(
        found, henon_fixed_point(), atol=1e-12,
        err_msg="expected the fixed point, which solves F^2(u) = u",
    )

    with pytest.raises(RuntimeError, match="period 1, which divides"):
        ds.find_periodic_orbit(guess, 2, prime_period=True)


def test_singular_monodromy_is_reported():
    """
    At k = 0 the standard map has a whole line of fixed points, so 1 is a
    Floquet multiplier, M - I is singular and the Newton step is undefined.

    The failure must be reported rather than papered over: without the check,
    the linear solve would return an arbitrary point on that line and it would
    look like a successful result.
    """
    ds = system("standard map", 0.0)

    with pytest.raises(RuntimeError, match="singular"):
        ds.find_periodic_orbit([0.3, 0.001], 1, periods=[1.0, 1.0])


# --------------------------------------------------------------------------
# Error contracts and backwards compatibility
# --------------------------------------------------------------------------

def test_failure_to_converge_is_reported():
    """A guess far from any orbit must raise, not return the last iterate."""
    ds = system("henon map", [HENON_A, HENON_B])

    with pytest.raises(RuntimeError, match="did not converge"):
        ds.find_periodic_orbit([50.0, 50.0], 1, max_iter=5)


@pytest.mark.parametrize(
    "kwargs, guess",
    [
        ({}, [0.1, 0.2, 0.3]),
        ({"periods": [1.0]}, [0.1, 0.2]),
        ({"periods": [1.0, 0.0]}, [0.1, 0.2]),
    ],
    ids=["guess of the wrong length", "periods of the wrong length",
         "non-positive period"],
)
def test_invalid_newton_arguments_are_rejected(kwargs, guess):
    """Shape and sign errors must be caught in the wrapper, before the solver."""
    ds = system("henon map", [HENON_A, HENON_B])

    with pytest.raises(ValueError):
        ds.find_periodic_orbit(guess, 1, **kwargs)


def test_grid_search_still_rejects_systems_that_are_not_two_dimensional():
    """
    The grid refinement is still 2D only, and asking for it on a larger system
    must say so, pointing at the solver that does support it.
    """
    ds = system("4d symplectic map", [0.5, 0.3, 0.1])

    with pytest.raises(ValueError, match="only implemented for 2D"):
        ds.find_periodic_orbit(search_grid((0.0, 1.0), (0.0, 1.0)), 1)


def test_symmetry_line_search_is_unaffected():
    """
    A 1D array selects Newton only when no symmetry line is given, so the
    symmetry-line search must keep receiving its 1D array of sampled
    coordinates as before.
    """
    from numba import njit

    @njit
    def symmetry_line(x, parameters):
        return 0.0 * x

    ds = system("standard map", STANDARD_K)
    found = ds.find_periodic_orbit(
        np.linspace(0.0, 1.0, 400), 1, tolerance=1e-2,
        symmetry_line=symmetry_line, axis=0,
    )

    assert np.ravel(found).shape == (2,), (
        f"the symmetry-line search should still return a point, got {found}"
    )
