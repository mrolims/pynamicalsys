"""
The Billiard class and its molecular dynamics collision kernels.

The map is checked against what does not depend on it: an independent dense
scan of the trajectory, exact results for the circle and for the equilateral
triangle orbit, identities that hold at every point of phase space (time
reversal, the preserved measure), and results published for the oval billiard.
"""

import numpy as np
import pytest
from joblib import Parallel, delayed
from numba import njit
from scipy.integrate import quad
from scipy.optimize import brentq

from pynamicalsys import Billiard
from pynamicalsys.discrete_time.billiards import (
    md_collision,
    oval_arc_length,
    oval_arc_position,
    oval_arc_speed,
    reflect,
)

TWO_PI = 2.0 * np.pi

# Below, at, and above the critical value eps_c = 1 / (1 + p^2) for each p.
OVAL_PARAMETERS = [
    (eps, p)
    for p in (1, 2, 3, 4, 5)
    for eps in (0.01, 0.05, 0.08, 0.1, 0.12, 0.13, 0.38, 0.54, 0.9)
]


def oval_radius(theta, eps, p):
    return 1.0 + eps * np.cos(p * theta)


def oval_outgoing(theta, alpha, eps, p):
    """Position and direction of a particle leaving the boundary, from first principles."""
    radius = oval_radius(theta, eps, p)
    dradius = -eps * p * np.sin(p * theta)
    phi = np.arctan2(
        dradius * np.sin(theta) + radius * np.cos(theta),
        dradius * np.cos(theta) - radius * np.sin(theta),
    )
    mu = phi + alpha

    return radius * np.cos(theta), radius * np.sin(theta), np.cos(mu), np.sin(mu)


def scanned_collision(theta, alpha, eps, p):
    """Find the first collision by scanning the whole trajectory on a dense grid."""
    x0, y0, dx, dy = oval_outgoing(theta, alpha, eps, p)

    def gap(tau):
        x, y = x0 + tau * dx, y0 + tau * dy
        return np.hypot(x, y) - oval_radius(np.arctan2(y, x), eps, p)

    taus = np.concatenate(
        (
            np.geomspace(1e-9, 1e-2, 2000, endpoint=False),
            np.linspace(1e-2, 2.0 * (1.0 + eps), 200_000),
        )
    )
    outside = np.flatnonzero(gap(taus) > 0.0)
    first = outside[0]
    tau = brentq(gap, taus[first - 1], taus[first], xtol=1e-14)

    return np.arctan2(y0 + tau * dy, x0 + tau * dx) % TWO_PI


def random_states(rng, size, margin=0.02):
    return np.column_stack(
        (rng.uniform(0.0, TWO_PI, size), rng.uniform(margin, np.pi - margin, size))
    )


def angle_difference(a, b):
    return (a - b + np.pi) % TWO_PI - np.pi


@pytest.fixture
def rng():
    return np.random.default_rng(20261102)


# ! --- Collision core ---------------------------------------------------- !


@njit
def unit_disk(x, y, t, parameters):
    return x * x + y * y < 1.0


@njit
def growing_disk(x, y, t, parameters):
    return np.sqrt(x * x + y * y) < 1.0 + parameters[0] * t


def test_collision_with_any_speed_and_start_time():
    t1, x1, y1 = md_collision(
        -1.0, 0.0, 2.5, 0.0, 7.0, unit_disk, np.zeros(1), 1e-3
    )
    assert x1 == pytest.approx(1.0, abs=1e-12)
    assert y1 == pytest.approx(0.0, abs=1e-12)
    assert t1 == pytest.approx(7.0 + 2.0 / 2.5, abs=1e-12)


def test_the_boundary_sees_the_current_time():
    # r = t for the particle and r = 1 + 0.1 t for the wall.
    t1, x1, _ = md_collision(
        0.0, 0.0, 1.0, 0.0, 0.0, growing_disk, np.array([0.1]), 1e-3
    )
    assert t1 == pytest.approx(1.0 / 0.9, abs=1e-12)
    assert x1 == pytest.approx(1.0 / 0.9, abs=1e-12)


def test_near_grazing_start_is_resolved_below_the_step():
    # A chord of length 2 sin(alpha) that is much shorter than the step. Round-off
    # in x^2 + y^2 limits the collision time to about 1e-16 / sin(alpha).
    alpha = 1e-6
    t1, _, _ = md_collision(
        1.0, 0.0, -np.sin(alpha), np.cos(alpha), 0.0, unit_disk, np.zeros(1), 1e-2
    )
    assert t1 == pytest.approx(2.0 * np.sin(alpha), abs=1e-9)


def test_reflection_at_a_static_wall_is_specular():
    assert reflect(0.6, -0.8, 1.0, 0.0, 0.0, 0.0) == pytest.approx((0.6, 0.8))


def test_reflection_at_a_moving_wall():
    # A head-on hit on a wall moving at w gives v' = -v + 2 w.
    assert reflect(1.0, 0.0, 0.0, 1.0, -0.5, 0.0) == pytest.approx((-2.0, 0.0))
    # A wall moving along its own tangent does not change the velocity.
    assert reflect(0.6, 0.8, 1.0, 0.0, 0.3, 0.0) == pytest.approx((0.6, -0.8))


# ! --- Oval billiard ----------------------------------------------------- !


@pytest.mark.parametrize("eps, p", OVAL_PARAMETERS)
def test_collisions_match_a_dense_scan(rng, eps, p):
    billiard = Billiard("oval", parameters=[eps, p], md_steps=5000)
    for theta, alpha in random_states(rng, 3):
        theta1, alpha1 = billiard.trajectory([theta, alpha], 1)[0]
        expected = scanned_collision(theta, alpha, eps, p)
        assert angle_difference(theta1, expected) == pytest.approx(0.0, abs=1e-9)
        assert 0.0 < alpha1 < np.pi


@pytest.mark.parametrize("eps, p", [(0.08, 3), (0.4, 3), (0.9, 2)])
def test_collisions_along_an_orbit_match_a_dense_scan(eps, p):
    billiard = Billiard("oval", parameters=[eps, p], md_steps=5000)
    orbit = billiard.trajectory([1.3, 0.9], 12)
    state = np.array([1.3, 0.9])
    for next_state in orbit:
        expected = scanned_collision(state[0], state[1], eps, p)
        assert angle_difference(next_state[0], expected) == pytest.approx(0.0, abs=1e-9)
        state = next_state


@pytest.mark.parametrize("alpha", [1e-5, 1e-3, np.pi - 1e-3, np.pi - 1e-5])
def test_near_grazing_collisions_match_the_circle(alpha):
    billiard = Billiard("oval", parameters=[0.0, 3])
    theta1, alpha1 = billiard.trajectory([1.0, alpha], 1)[0]
    assert theta1 == pytest.approx((1.0 + 2.0 * alpha) % TWO_PI, abs=1e-10)
    assert alpha1 == pytest.approx(alpha, abs=1e-10)


def test_circle_preserves_alpha_and_advances_theta_by_twice_alpha(rng):
    billiard = Billiard("oval", parameters=[0.0, 3])
    for theta, alpha in random_states(rng, 20):
        theta1, alpha1 = billiard.trajectory([theta, alpha], 1)[0]
        assert angle_difference(theta1, theta + 2.0 * alpha) == pytest.approx(
            0.0, abs=1e-12
        )
        assert alpha1 == pytest.approx(alpha, abs=1e-12)


@pytest.mark.parametrize(
    "eps, start",
    [
        (0.08, 0.0),
        (0.1, 2.0 * np.pi / 3.0),
        (0.3, 0.0),
        (0.08, np.pi / 3.0),
        (0.5, np.pi / 3.0),
        (0.9, np.pi),
    ],
)
def test_equilateral_triangle_is_a_period_three_orbit(eps, start):
    # For p = 3, the boundary repeats every 2 pi / 3 and is symmetric about each
    # such axis, so these triangles are exact orbits. Each collision starts from
    # the exact point: these orbits are so unstable (a multiplier near 1e9 over
    # the three collisions at eps = 0.9) that round-off grows along a trajectory.
    billiard = Billiard("oval", parameters=[eps, 3], md_steps=5000)
    for k in range(3):
        theta = start + k * TWO_PI / 3.0
        theta1, alpha1 = billiard.trajectory([theta, np.pi / 3.0], 1)[0]
        assert angle_difference(theta1, theta + TWO_PI / 3.0) == pytest.approx(
            0.0, abs=1e-10
        )
        assert alpha1 == pytest.approx(np.pi / 3.0, abs=1e-10)


def test_inward_bulge_blocks_the_triangle_above_one_third():
    # With vertices at the largest radius, 1 + eps, the middle of a side is at
    # radius (1 + eps) / 2 and the bulge reaches 1 - eps, so for eps > 1 / 3 the
    # particle hits the bulge before the next vertex.
    eps = 0.5
    billiard = Billiard("oval", parameters=[eps, 3], md_steps=5000)
    theta1, _ = billiard.trajectory([0.0, np.pi / 3.0], 1)[0]
    assert theta1 == pytest.approx(scanned_collision(0.0, np.pi / 3.0, eps, 3), abs=1e-9)
    assert theta1 < np.pi / 3.0


def test_a_large_step_skips_the_first_collision_da_costa_fig_9():
    # da Costa et al. (2022), Fig. 9: 50 steps cross the inward bulge of the
    # boundary and reach the far wall, which is not the true collision.
    state = [0.55, 0.1442]
    parameters = [0.4, 3]
    coarse = Billiard("oval", parameters=parameters, md_steps=50)
    fine = Billiard("oval", parameters=parameters)
    x_coarse, _ = coarse.to_cartesian(coarse.trajectory(state, 1)[0])
    x_fine, _ = fine.to_cartesian(fine.trajectory(state, 1)[0])
    assert x_coarse == pytest.approx(-0.86, abs=0.01)
    assert x_fine == pytest.approx(0.417, abs=0.01)


@pytest.mark.parametrize("eps, p", [(0.05, 3), (0.08, 3), (0.4, 3), (0.3, 2)])
def test_backward_map_inverts_forward_map(rng, eps, p):
    billiard = Billiard("oval", parameters=[eps, p], md_steps=5000)
    backward = billiard._DiscreteDynamicalSystem__backwards_mapping
    parameters = billiard.get_parameters()
    for state in random_states(rng, 20, margin=0.1):
        forward = billiard.trajectory(state, 1)[0]
        back = backward(forward, parameters)
        assert angle_difference(back[0], state[0]) == pytest.approx(0.0, abs=1e-9)
        assert back[1] == pytest.approx(state[1], abs=1e-9)


@pytest.mark.parametrize("eps, p", [(0.05, 3), (0.08, 3), (0.2, 3), (0.1, 2)])
def test_analytic_jacobian_matches_finite_differences(rng, eps, p):
    billiard = Billiard("oval", parameters=[eps, p], md_steps=5000)
    mapping = billiard._DiscreteDynamicalSystem__mapping
    jacobian = billiard._DiscreteDynamicalSystem__jacobian
    parameters = billiard.get_parameters()
    h = 1e-6
    for state in random_states(rng, 10, margin=0.3):
        numerical = np.empty((2, 2))
        for j in range(2):
            step = np.zeros(2)
            step[j] = h
            plus = mapping(state + step, parameters)
            minus = mapping(state - step, parameters)
            numerical[0, j] = angle_difference(plus[0], minus[0]) / (2.0 * h)
            numerical[1, j] = (plus[1] - minus[1]) / (2.0 * h)
        np.testing.assert_allclose(
            jacobian(state, parameters), numerical, rtol=1e-4, atol=1e-4
        )


@pytest.mark.parametrize("eps, p", [(0.08, 3), (0.2, 3), (0.5, 4)])
def test_jacobian_determinant_follows_the_preserved_measure(rng, eps, p):
    # The map preserves sin(alpha) (ds/dtheta) dtheta dalpha.
    billiard = Billiard("oval", parameters=[eps, p])
    jacobian = billiard._DiscreteDynamicalSystem__jacobian
    parameters = billiard.get_parameters()

    def density(theta, alpha):
        dradius = -eps * p * np.sin(p * theta)
        return np.sin(alpha) * np.hypot(oval_radius(theta, eps, p), dradius)

    for state in random_states(rng, 20, margin=0.1):
        theta1, alpha1 = billiard.trajectory(state, 1)[0]
        assert np.linalg.det(jacobian(state, parameters)) == pytest.approx(
            density(*state) / density(theta1, alpha1), rel=1e-8
        )


def hansen_survival_slope(center, lower, first, last):
    billiard = Billiard("oval", parameters=[0.08, 3])
    thetas = np.linspace(lower, lower + 0.2, 42)[1:-1]
    alphas = np.linspace(0.0, np.pi, 42)[1:-1]
    max_time = 2000
    times = [
        billiard.escape_analysis(
            [theta, alpha], max_time, exits=[center, 0.0], hole_size=[0.2, np.inf]
        )[1]
        for theta in thetas
        for alpha in alphas
    ]
    n, survival = billiard.survival_probability(times, max_time)
    window = (n >= first) & (n <= last) & (survival > 0.0)

    return np.polyfit(n[window], np.log(survival[window]), 1)[0]


def test_survival_decay_of_the_mixed_hole_matches_hansen():
    # Hansen et al. (2018), Fig. 4(a): -0.0188(7) for the hole h1 in (3.04, 3.24).
    assert hansen_survival_slope(3.14, 3.04, 20, 100) == pytest.approx(-0.0188, rel=0.2)


def test_survival_decays_faster_through_the_chaotic_hole_as_in_hansen():
    # Hansen et al. (2018), Fig. 4(b): -0.0437(3) for the hole h2 in (3.95, 4.15).
    # The fit window is not stated and the slope depends on it, so this checks
    # the early decay loosely, and that this hole empties faster than h1.
    chaotic = hansen_survival_slope(4.05, 3.95, 1, 60)
    mixed = hansen_survival_slope(3.14, 3.04, 1, 60)
    assert chaotic == pytest.approx(-0.0437, rel=0.25)
    assert chaotic < 1.5 * mixed


# ! --- Birkhoff coordinates ---------------------------------------------- !

ARC_PARAMETERS = [
    (eps, p)
    for p in (1, 2, 3, 4, 5, 12)
    for eps in (0.0, 0.01, 0.08, 0.1, 0.5, 0.9, 0.99, 0.999, 0.999999)
]


def oval_speed(theta, eps, p):
    return np.hypot(oval_radius(theta, eps, p), eps * p * np.sin(p * theta))


def integrated_arc_length(theta, eps, p):
    # Adaptive quadrature between the extrema of the radius.
    wedge = np.pi / p
    breaks = np.arange(0.0, theta, wedge)
    edges = np.append(breaks, theta)
    return sum(
        quad(
            oval_speed,
            a,
            b,
            args=(eps, p),
            limit=500,
            epsabs=1e-13,
            epsrel=1e-13,
        )[0]
        for a, b in zip(edges[:-1], edges[1:])
    )


@pytest.mark.parametrize("eps, p", ARC_PARAMETERS)
def test_perimeter_matches_adaptive_quadrature(eps, p):
    billiard = Billiard("oval", parameters=[eps, p])
    assert billiard.perimeter() == pytest.approx(
        integrated_arc_length(TWO_PI, eps, p), abs=1e-11
    )


@pytest.mark.parametrize("eps, p", ARC_PARAMETERS)
def test_arc_length_matches_adaptive_quadrature(rng, eps, p):
    parameters = np.array([eps, p], dtype=np.float64)
    for theta in rng.uniform(0.0, TWO_PI, 5):
        assert oval_arc_length(theta, 0.0, parameters) == pytest.approx(
            integrated_arc_length(theta, eps, p), abs=1e-11
        )


@pytest.mark.parametrize("eps, p", ARC_PARAMETERS)
def test_arc_position_inverts_arc_length(rng, eps, p):
    parameters = np.array([eps, p], dtype=np.float64)
    thetas = np.concatenate(
        (
            rng.uniform(0.0, TWO_PI, 8),
            np.arange(2 * p) * np.pi / p,  # Ends of the wedges.
            np.arange(2 * p) * np.pi / p + 1e-9,
        )
    )
    for theta in thetas:
        s = oval_arc_length(theta, 0.0, parameters)
        position = oval_arc_position(s, 0.0, parameters)
        # The error as a length along the boundary: where ds/dtheta is small, as at
        # an inward bulge with eps close to one, round-off in s is large in theta.
        error = angle_difference(position, theta) * oval_speed(theta, eps, p)
        assert error == pytest.approx(0.0, abs=1e-12)


def test_arc_length_increases_and_covers_the_perimeter():
    parameters = np.array([0.9, 3.0])
    thetas = np.linspace(0.0, TWO_PI, 2001)[:-1]
    arcs = np.array([oval_arc_length(theta, 0.0, parameters) for theta in thetas])
    assert np.all(np.diff(arcs) > 0.0)
    assert arcs[0] == 0.0
    assert arcs[-1] < Billiard("oval", parameters=parameters).perimeter()


def test_arc_speed_is_the_derivative_of_the_arc_length():
    parameters = np.array([0.4, 3.0])
    h = 1e-5
    for theta in (0.3, 1.0, 2.5, 5.0):
        derivative = (
            oval_arc_length(theta + h, 0.0, parameters)
            - oval_arc_length(theta - h, 0.0, parameters)
        ) / (2.0 * h)
        assert derivative == pytest.approx(
            oval_arc_speed(theta, 0.0, parameters), rel=1e-8
        )


def test_circle_in_birkhoff_coordinates():
    billiard = Billiard("oval", parameters=[0.0, 3], coordinates="birkhoff")
    assert billiard.perimeter() == pytest.approx(TWO_PI)
    s1, cos_alpha1 = billiard.trajectory([1.0, 0.5], 1)[0]
    assert s1 == pytest.approx(1.0 + 2.0 * np.arccos(0.5), abs=1e-12)
    assert cos_alpha1 == pytest.approx(0.5, abs=1e-12)


@pytest.mark.parametrize("eps, p", [(0.05, 3), (0.08, 3), (0.2, 3), (0.5, 4), (0.9, 2)])
def test_both_coordinates_give_the_same_orbit(eps, p):
    angles = Billiard("oval", parameters=[eps, p], md_steps=5000)
    birkhoff = Billiard(
        "oval", parameters=[eps, p], coordinates="birkhoff", md_steps=5000
    )
    parameters = angles.get_parameters()
    theta, alpha = 1.3, 0.9
    # Compare over few collisions, before round-off grows along the chaotic orbit.
    orbit = angles.trajectory([theta, alpha], 8)
    converted = birkhoff.trajectory(
        [oval_arc_length(theta, 0.0, parameters), np.cos(alpha)], 8
    )
    np.testing.assert_allclose(
        birkhoff.to_cartesian(converted), angles.to_cartesian(orbit), atol=1e-6
    )
    np.testing.assert_allclose(converted[:, 1], np.cos(orbit[:, 1]), atol=1e-6)


def birkhoff_states(rng, parameters, size, margin=0.1):
    perimeter = Billiard("oval", parameters=parameters).perimeter()
    return np.column_stack(
        (
            rng.uniform(0.0, perimeter, size),
            rng.uniform(-1.0 + margin, 1.0 - margin, size),
        )
    )


@pytest.mark.parametrize("eps, p", [(0.05, 3), (0.08, 3), (0.2, 3), (0.1, 2)])
def test_birkhoff_jacobian_matches_finite_differences(rng, eps, p):
    billiard = Billiard(
        "oval", parameters=[eps, p], coordinates="birkhoff", md_steps=5000
    )
    mapping = billiard._DiscreteDynamicalSystem__mapping
    jacobian = billiard._DiscreteDynamicalSystem__jacobian
    parameters = billiard.get_parameters()
    perimeter = billiard.perimeter()
    h = 1e-6
    for state in birkhoff_states(rng, [eps, p], 10, margin=0.2):
        numerical = np.empty((2, 2))
        for j in range(2):
            step = np.zeros(2)
            step[j] = h
            plus = mapping(state + step, parameters)
            minus = mapping(state - step, parameters)
            wrapped = (plus[0] - minus[0] + perimeter / 2.0) % perimeter
            numerical[0, j] = (wrapped - perimeter / 2.0) / (2.0 * h)
            numerical[1, j] = (plus[1] - minus[1]) / (2.0 * h)
        np.testing.assert_allclose(
            jacobian(state, parameters), numerical, rtol=1e-4, atol=1e-4
        )


@pytest.mark.parametrize(
    "eps, p", [(0.05, 3), (0.08, 3), (0.2, 3), (0.5, 4), (0.9, 3), (0.3, 1)]
)
def test_birkhoff_map_preserves_area(rng, eps, p):
    billiard = Billiard("oval", parameters=[eps, p], coordinates="birkhoff")
    jacobian = billiard._DiscreteDynamicalSystem__jacobian
    parameters = billiard.get_parameters()
    for state in birkhoff_states(rng, [eps, p], 30):
        assert np.linalg.det(jacobian(state, parameters)) == pytest.approx(1.0, abs=1e-8)


@pytest.mark.parametrize("eps, p", [(0.05, 3), (0.08, 3), (0.4, 3), (0.3, 2)])
def test_birkhoff_backward_map_inverts_forward_map(rng, eps, p):
    billiard = Billiard(
        "oval", parameters=[eps, p], coordinates="birkhoff", md_steps=5000
    )
    backward = billiard._DiscreteDynamicalSystem__backwards_mapping
    parameters = billiard.get_parameters()
    perimeter = billiard.perimeter()
    for state in birkhoff_states(rng, [eps, p], 20):
        back = backward(billiard.trajectory(state, 1)[0], parameters)
        difference = (back[0] - state[0] + perimeter / 2.0) % perimeter - perimeter / 2.0
        assert difference == pytest.approx(0.0, abs=1e-9)
        assert back[1] == pytest.approx(state[1], abs=1e-9)


@pytest.mark.parametrize("cos_alpha", [1.0, -1.0, 1.5, np.nan])
def test_cos_alpha_outside_its_range(cos_alpha):
    billiard = Billiard("oval", parameters=[0.08, 3], coordinates="birkhoff")
    with pytest.raises(ValueError, match="alpha"):
        billiard.trajectory([1.0, cos_alpha], 1)


def test_birkhoff_info():
    billiard = Billiard("oval", coordinates="birkhoff")
    assert billiard.info["state"] == ["s", "cos(alpha)"]
    assert billiard.info["period"] is None
    billiard.set_parameters([0.2, 3])
    assert billiard.info["period"] == pytest.approx(billiard.perimeter())


def test_a_hole_in_arc_length_is_the_same_hole_in_angle():
    parameters = [0.08, 3]
    angles = Billiard("oval", parameters=parameters)
    birkhoff = Billiard("oval", parameters=parameters, coordinates="birkhoff")
    values = np.asarray(parameters, dtype=np.float64)
    lower = oval_arc_length(1.0, 0.0, values)
    upper = oval_arc_length(1.2, 0.0, values)
    for theta, alpha in [(2.0, 1.0), (3.5, 0.6), (5.0, 2.0)]:
        in_angle = angles.escape_analysis(
            [theta, alpha], 2000, exits=[1.1, 0.0], hole_size=[0.2, np.inf]
        )
        # The hole in angle is not centered in arc length, but has the same ends.
        in_arc = birkhoff.escape_analysis(
            [oval_arc_length(theta, 0.0, values), np.cos(alpha)],
            2000,
            exits=[0.5 * (lower + upper), 0.0],
            hole_size=[upper - lower, np.inf],
        )
        assert in_arc == in_angle


def test_birkhoff_hole_at_zero_wraps_with_the_perimeter():
    billiard = Billiard("oval", parameters=[0.08, 3], coordinates="birkhoff")
    perimeter = billiard.perimeter()
    state = [2.0, 0.5]
    orbit = billiard.trajectory(state, 500)
    width = 0.2
    index, time = billiard.escape_analysis(
        state, 500, exits=[0.0, 0.0], hole_size=[width, np.inf]
    )
    inside = (orbit[:, 0] <= width / 2.0) | (orbit[:, 0] >= perimeter - width / 2.0)
    assert (index, time) == (0, np.flatnonzero(inside)[0] + 1)


def test_birkhoff_lyapunov_exponents_sum_to_zero():
    billiard = Billiard("oval", parameters=[0.1, 3], coordinates="birkhoff")
    exponents = billiard.lyapunov([2.0, 0.8], 5000)
    assert exponents[0] > 0.0
    assert np.sum(exponents) == pytest.approx(0.0, abs=1e-6)


def test_parallel_workers_in_birkhoff_coordinates():
    billiard = Billiard("oval", parameters=[0.08, 3], coordinates="birkhoff")
    states = [[2.0, c] for c in (-0.5, 0.0, 0.5)]
    kwargs = dict(max_time=200, exits=[1.0, 0.0], hole_size=[0.3, np.inf])
    in_parallel = Parallel(n_jobs=2)(
        delayed(billiard.escape_analysis)(state, **kwargs) for state in states
    )
    assert in_parallel == [billiard.escape_analysis(s, **kwargs) for s in states]


# ! --- Interface --------------------------------------------------------- !


def test_available_names():
    assert "oval" in Billiard.available_billiards()
    assert Billiard.available_solvers("oval") == ["md"]


def test_info():
    info = Billiard("oval", md_steps=100).info
    assert info["parameters"] == ["eps", "p"]
    assert info["state"] == ["theta", "alpha"]
    assert info["solver"] == "md"
    assert info["md_steps"] == 100


@pytest.mark.parametrize(
    "kwargs, error",
    [
        (dict(name="triangle"), ValueError),
        (dict(name="oval", solver="newton"), ValueError),
        (dict(name="oval", coordinates="polar"), ValueError),
        (dict(name="oval", md_steps=0), ValueError),
        (dict(name="oval", md_steps=10.5), TypeError),
        (dict(name="oval", md_steps=True), TypeError),
    ],
)
def test_invalid_construction(kwargs, error):
    with pytest.raises(error):
        Billiard(**kwargs)


def test_available_solvers_of_an_unknown_billiard():
    with pytest.raises(ValueError, match="triangle"):
        Billiard.available_solvers("triangle")


@pytest.mark.parametrize(
    "parameters", [[1.0, 3.0], [-0.1, 3.0], [0.1, 0.0], [0.1, 2.5], [0.1, -3.0]]
)
def test_invalid_parameters(parameters):
    billiard = Billiard("oval")
    with pytest.raises(ValueError):
        billiard.trajectory([1.0, 1.0], 1, parameters=parameters)
    with pytest.raises(ValueError):
        billiard.boundary(parameters=parameters)


@pytest.mark.parametrize("alpha", [0.0, np.pi, -0.5, 4.0, np.nan])
def test_alpha_outside_its_range(alpha):
    billiard = Billiard("oval", parameters=[0.08, 3])
    with pytest.raises(ValueError, match="alpha"):
        billiard.trajectory([1.0, alpha], 1)


def test_boundary_lies_on_the_radius():
    billiard = Billiard("oval", parameters=[0.2, 3])
    points = billiard.boundary(200)
    assert points.shape == (200, 2)
    theta = np.arctan2(points[:, 1], points[:, 0])
    np.testing.assert_allclose(
        np.hypot(points[:, 0], points[:, 1]), oval_radius(theta, 0.2, 3), atol=1e-12
    )
    np.testing.assert_allclose(points[0], points[-1], atol=1e-12)


def test_to_cartesian_gives_points_on_the_boundary(rng):
    billiard = Billiard("oval", parameters=[0.2, 3])
    orbit = billiard.trajectory([1.0, 1.0], 50)
    points = billiard.to_cartesian(orbit)
    assert points.shape == (50, 2)
    np.testing.assert_allclose(
        np.hypot(points[:, 0], points[:, 1]),
        oval_radius(orbit[:, 0], 0.2, 3),
        atol=1e-12,
    )
    np.testing.assert_array_equal(billiard.to_cartesian(orbit[0]), points[0])


def test_to_cartesian_rejects_other_shapes():
    billiard = Billiard("oval", parameters=[0.2, 3])
    with pytest.raises(ValueError, match="states"):
        billiard.to_cartesian(np.zeros((2, 3)))


def test_a_method_inherited_from_the_discrete_class():
    # Inside the chaotic sea, against the regular equilateral triangle region.
    billiard = Billiard("oval", parameters=[0.1, 3])
    assert billiard.lyapunov([2.0, 0.5], 5000)[0] > 0.0


# ! --- Escape through holes ---------------------------------------------- !


def test_a_strip_stops_at_the_first_collision_inside_it():
    billiard = Billiard("oval", parameters=[0.08, 3])
    state = [2.0, 1.0]
    orbit = billiard.trajectory(state, 200)
    center, width = 3.0, 0.2
    inside = np.abs(angle_difference(orbit[:, 0], center)) <= width / 2.0
    assert billiard.escape_analysis(
        state, 200, exits=[center, 0.0], hole_size=[width, np.inf]
    ) == (0, np.flatnonzero(inside)[0] + 1)


def test_a_hole_at_zero_catches_collisions_on_both_sides():
    billiard = Billiard("oval", parameters=[0.08, 3])
    state = [2.0, 1.0]
    orbit = billiard.trajectory(state, 500)
    width = 0.2
    index, time = billiard.escape_analysis(
        state, 500, exits=[0.0, 0.0], hole_size=[width, np.inf]
    )
    inside = np.abs(angle_difference(orbit[:, 0], 0.0)) <= width / 2.0
    assert (index, time) == (0, np.flatnonzero(inside)[0] + 1)
    below = orbit[inside, 0] > np.pi
    above = ~below
    assert below.any() and above.any()


def test_holes_of_different_widths():
    billiard = Billiard("oval", parameters=[0.08, 3])
    state = [2.0, 1.0]
    orbit = billiard.trajectory(state, 500)
    centers = np.array([[1.0, 0.0], [4.0, 0.0]])
    widths = np.array([0.05, 0.3])
    index, time = billiard.escape_analysis(
        state,
        500,
        exits=centers,
        hole_size=np.column_stack((widths, np.full(2, np.inf))),
    )
    distance = np.abs(angle_difference(orbit[:, 0, None], centers[:, 0]))
    inside = distance <= widths / 2.0
    first = np.flatnonzero(inside.any(axis=1))[0]
    assert time == first + 1
    assert inside[first, index]


def test_escape_can_turn_wrapping_off():
    billiard = Billiard("oval", parameters=[0.08, 3])
    state = [2.0, 1.0]
    wrapped = billiard.escape_analysis(
        state, 2000, exits=[0.0, 0.0], hole_size=[0.2, np.inf]
    )
    unwrapped = billiard.escape_analysis(
        state,
        2000,
        exits=[0.0, 0.0],
        hole_size=[0.2, np.inf],
        periods=[np.inf, np.inf],
    )
    assert wrapped[1] <= unwrapped[1]


def test_parallel_workers():
    billiard = Billiard("oval", parameters=[0.08, 3])
    states = [[2.0, a] for a in (0.5, 1.0, 1.5)]
    kwargs = dict(max_time=200, exits=[1.0, 0.0], hole_size=[0.3, np.inf])
    in_parallel = Parallel(n_jobs=2)(
        delayed(billiard.escape_analysis)(state, **kwargs) for state in states
    )
    assert in_parallel == [billiard.escape_analysis(s, **kwargs) for s in states]
