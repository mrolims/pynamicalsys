"""
The Billiard class and its molecular dynamics collision kernels.

The map is checked against what does not depend on it: an independent dense
scan of the trajectory, exact results for the circle and for the equilateral
triangle orbit, identities that hold at every point of phase space (time
reversal, the preserved measure), and results published for the oval billiard.
"""

import pickle
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from joblib import Parallel, delayed
from joblib.externals import cloudpickle
from numba import njit
from scipy.integrate import quad
from scipy.optimize import brentq

from pynamicalsys import Billiard
from pynamicalsys.core.billiards import (
    _fixed_pieces_table,
    _pieces_meeting_point,
    _pieces_problem,
)
from pynamicalsys.discrete_time.billiards import (
    inside_pieces,
    md_collision,
    mushroom_inside,
    new_pieces,
    oval_arc_length,
    oval_arc_position,
    oval_arc_speed,
    oval_boundary_point,
    oval_circle_distance,
    oval_polygon_distance,
    oval_tangent,
    piece_point,
    reflect,
    set_arc,
    set_line,
    sinai_inside,
    stadium_inside,
)

TWO_PI = 2.0 * np.pi


def oval_solver(eps, p):
    # The fastest valid solver for the oval billiard.
    if p == 1:
        return "exact"
    if eps <= 1.0 / (1.0 + p * p):
        return "circle"
    return "polygon" if eps < 1.0 else "md"


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
    billiard = Billiard("oval", parameters=[eps, p], md_steps=5000, solver=oval_solver(eps, p))
    for theta, alpha in random_states(rng, 3):
        theta1, alpha1 = billiard.trajectory([theta, alpha], 1)[0]
        expected = scanned_collision(theta, alpha, eps, p)
        assert angle_difference(theta1, expected) == pytest.approx(0.0, abs=1e-9)
        assert 0.0 < alpha1 < np.pi


@pytest.mark.parametrize("eps, p", [(0.08, 3), (0.4, 3), (0.9, 2)])
def test_collisions_along_an_orbit_match_a_dense_scan(eps, p):
    billiard = Billiard("oval", parameters=[eps, p], md_steps=5000, solver=oval_solver(eps, p))
    orbit = billiard.trajectory([1.3, 0.9], 12)
    state = np.array([1.3, 0.9])
    for next_state in orbit:
        expected = scanned_collision(state[0], state[1], eps, p)
        assert angle_difference(next_state[0], expected) == pytest.approx(0.0, abs=1e-9)
        state = next_state


@pytest.mark.parametrize("alpha", [1e-5, 1e-3, np.pi - 1e-3, np.pi - 1e-5])
def test_near_grazing_collisions_match_the_circle(alpha):
    billiard = Billiard("oval", parameters=[0.0, 3], solver="circle")
    theta1, alpha1 = billiard.trajectory([1.0, alpha], 1)[0]
    assert theta1 == pytest.approx((1.0 + 2.0 * alpha) % TWO_PI, abs=1e-10)
    assert alpha1 == pytest.approx(alpha, abs=1e-10)


def test_circle_preserves_alpha_and_advances_theta_by_twice_alpha(rng):
    billiard = Billiard("oval", parameters=[0.0, 3], solver="circle")
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
    billiard = Billiard("oval", parameters=[eps, 3], md_steps=5000, solver=oval_solver(eps, 3))
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
    billiard = Billiard("oval", parameters=[eps, 3], md_steps=5000, solver=oval_solver(eps, 3))
    theta1, _ = billiard.trajectory([0.0, np.pi / 3.0], 1)[0]
    assert theta1 == pytest.approx(scanned_collision(0.0, np.pi / 3.0, eps, 3), abs=1e-9)
    assert theta1 < np.pi / 3.0


def test_a_large_step_skips_the_first_collision_da_costa_fig_9():
    # da Costa et al. (2022), Fig. 9: 50 steps cross the inward bulge of the
    # boundary and reach the far wall, which is not the true collision.
    state = [0.55, 0.1442]
    parameters = [0.4, 3]
    coarse = Billiard("oval", parameters=parameters, solver="md", md_steps=50)
    fine = Billiard("oval", parameters=parameters, solver="md")
    x_coarse, _ = coarse.to_cartesian(coarse.trajectory(state, 1)[0])
    x_fine, _ = fine.to_cartesian(fine.trajectory(state, 1)[0])
    assert x_coarse == pytest.approx(-0.86, abs=0.01)
    assert x_fine == pytest.approx(0.417, abs=0.01)


@pytest.mark.parametrize("eps, p", [(0.05, 3), (0.08, 3), (0.4, 3), (0.3, 2)])
def test_backward_map_inverts_forward_map(rng, eps, p):
    billiard = Billiard("oval", parameters=[eps, p], md_steps=5000, solver=oval_solver(eps, p))
    backward = billiard._DiscreteDynamicalSystem__backwards_mapping
    parameters = billiard.get_parameters()
    for state in random_states(rng, 20, margin=0.1):
        forward = billiard.trajectory(state, 1)[0]
        back = backward(forward, parameters)
        assert angle_difference(back[0], state[0]) == pytest.approx(0.0, abs=1e-9)
        assert back[1] == pytest.approx(state[1], abs=1e-9)


@pytest.mark.parametrize("eps, p", [(0.05, 3), (0.08, 3), (0.2, 3), (0.1, 2)])
def test_analytic_jacobian_matches_finite_differences(rng, eps, p):
    billiard = Billiard("oval", parameters=[eps, p], md_steps=5000, solver=oval_solver(eps, p))
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
    billiard = Billiard("oval", parameters=[eps, p], solver=oval_solver(eps, p))
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
    billiard = Billiard("oval", parameters=[0.08, 3], solver="circle")
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
    for eps in (0.0, 0.01, 0.08, 0.1, 0.5, 0.9, 0.99, 0.999, 0.999999, 1.0)
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
    billiard = Billiard("oval", parameters=[eps, p], solver=oval_solver(eps, p))
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
    assert arcs[-1] < Billiard("oval", parameters=parameters, solver=oval_solver(*parameters)).perimeter()


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
    billiard = Billiard("oval", parameters=[0.0, 3], coordinates="birkhoff", solver="circle")
    assert billiard.perimeter() == pytest.approx(TWO_PI)
    s1, cos_alpha1 = billiard.trajectory([1.0, 0.5], 1)[0]
    assert s1 == pytest.approx(1.0 + 2.0 * np.arccos(0.5), abs=1e-12)
    assert cos_alpha1 == pytest.approx(0.5, abs=1e-12)


@pytest.mark.parametrize("eps, p", [(0.05, 3), (0.08, 3), (0.2, 3), (0.5, 4), (0.9, 2)])
def test_both_coordinates_give_the_same_orbit(eps, p):
    angles = Billiard("oval", parameters=[eps, p], md_steps=5000, solver=oval_solver(eps, p))
    birkhoff = Billiard(
        "oval", parameters=[eps, p], coordinates="birkhoff", md_steps=5000,
        solver=oval_solver(eps, p),
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
    perimeter = Billiard("oval", parameters=parameters, solver=oval_solver(*parameters)).perimeter()
    return np.column_stack(
        (
            rng.uniform(0.0, perimeter, size),
            rng.uniform(-1.0 + margin, 1.0 - margin, size),
        )
    )


@pytest.mark.parametrize("eps, p", [(0.05, 3), (0.08, 3), (0.2, 3), (0.1, 2)])
def test_birkhoff_jacobian_matches_finite_differences(rng, eps, p):
    billiard = Billiard(
        "oval", parameters=[eps, p], coordinates="birkhoff", md_steps=5000,
        solver=oval_solver(eps, p),
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
    billiard = Billiard("oval", parameters=[eps, p], coordinates="birkhoff", solver=oval_solver(eps, p))
    jacobian = billiard._DiscreteDynamicalSystem__jacobian
    parameters = billiard.get_parameters()
    for state in birkhoff_states(rng, [eps, p], 30):
        assert np.linalg.det(jacobian(state, parameters)) == pytest.approx(1.0, abs=1e-8)


@pytest.mark.parametrize("eps, p", [(0.05, 3), (0.08, 3), (0.4, 3), (0.3, 2)])
def test_birkhoff_backward_map_inverts_forward_map(rng, eps, p):
    billiard = Billiard(
        "oval", parameters=[eps, p], coordinates="birkhoff", md_steps=5000,
        solver=oval_solver(eps, p),
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
    billiard = Billiard("oval", parameters=[0.08, 3], coordinates="birkhoff", solver="circle")
    with pytest.raises(ValueError, match="alpha"):
        billiard.trajectory([1.0, cos_alpha], 1)


def test_birkhoff_info():
    billiard = Billiard("oval", coordinates="birkhoff", solver="md")
    assert billiard.info["state"] == ["s", "cos(alpha)"]
    assert billiard.info["period"] is None
    billiard.set_parameters([0.2, 3])
    assert billiard.info["period"] == pytest.approx(billiard.perimeter())


def test_a_hole_in_arc_length_is_the_same_hole_in_angle():
    parameters = [0.08, 3]
    angles = Billiard("oval", parameters=parameters, solver=oval_solver(*parameters))
    birkhoff = Billiard("oval", parameters=parameters, coordinates="birkhoff", solver=oval_solver(*parameters))
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
    billiard = Billiard("oval", parameters=[0.08, 3], coordinates="birkhoff", solver="circle")
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
    billiard = Billiard("oval", parameters=[0.1, 3], coordinates="birkhoff", solver="circle")
    exponents = billiard.lyapunov([2.0, 0.8], 5000)
    assert exponents[0] > 0.0
    assert np.sum(exponents) == pytest.approx(0.0, abs=1e-6)


def test_parallel_workers_in_birkhoff_coordinates():
    billiard = Billiard("oval", parameters=[0.08, 3], coordinates="birkhoff", solver="circle")
    states = [[2.0, c] for c in (-0.5, 0.0, 0.5)]
    kwargs = dict(max_time=200, exits=[1.0, 0.0], hole_size=[0.3, np.inf])
    in_parallel = Parallel(n_jobs=2)(
        delayed(billiard.escape_analysis)(state, **kwargs) for state in states
    )
    assert in_parallel == [billiard.escape_analysis(s, **kwargs) for s in states]


# ! --- Interface --------------------------------------------------------- !


def test_available_names():
    assert "oval" in Billiard.available_billiards()
    assert Billiard.available_solvers("oval") == [
        "md",
        "exact",
        "circle",
        "polygon",
    ]


def test_info():
    info = Billiard("oval", md_steps=100, solver="md").info
    assert info["parameters"] == ["eps", "p"]
    assert info["state"] == ["theta", "alpha"]
    assert info["solver"] == "md"
    assert "active_solver" not in info
    assert info["md_steps"] == 100


@pytest.mark.parametrize(
    "kwargs, error",
    [
        (dict(name="triangle"), ValueError),
        (dict(name="oval", solver="newton"), ValueError),
        (dict(name="oval", solver="auto"), ValueError),
        (dict(name="oval", coordinates="polar"), ValueError),
        (dict(name="oval", md_steps=0), ValueError),
        (dict(name="oval", md_steps=10.5), TypeError),
        (dict(name="oval", md_steps=True), TypeError),
    ],
)
def test_invalid_construction(kwargs, error):
    with pytest.raises(error):
        Billiard(**{"solver": "md", **kwargs})


def test_available_solvers_of_an_unknown_billiard():
    with pytest.raises(ValueError, match="triangle"):
        Billiard.available_solvers("triangle")


@pytest.mark.parametrize(
    "parameters", [[1.5, 3.0], [-0.1, 3.0], [0.1, 0.0], [0.1, 2.5], [0.1, -3.0]]
)
def test_invalid_parameters(parameters):
    billiard = Billiard("oval", solver="md")
    with pytest.raises(ValueError):
        billiard.trajectory([1.0, 1.0], 1, parameters=parameters)
    with pytest.raises(ValueError):
        billiard.boundary(parameters=parameters)


@pytest.mark.parametrize("alpha", [0.0, np.pi, -0.5, 4.0, np.nan])
def test_alpha_outside_its_range(alpha):
    billiard = Billiard("oval", parameters=[0.08, 3], solver="circle")
    with pytest.raises(ValueError, match="alpha"):
        billiard.trajectory([1.0, alpha], 1)


def test_boundary_lies_on_the_radius():
    billiard = Billiard("oval", parameters=[0.2, 3], solver="polygon")
    points = billiard.boundary(200)
    assert points.shape == (200, 2)
    theta = np.arctan2(points[:, 1], points[:, 0])
    np.testing.assert_allclose(
        np.hypot(points[:, 0], points[:, 1]), oval_radius(theta, 0.2, 3), atol=1e-12
    )
    np.testing.assert_allclose(points[0], points[-1], atol=1e-12)


def test_to_cartesian_gives_points_on_the_boundary(rng):
    billiard = Billiard("oval", parameters=[0.2, 3], solver="polygon")
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
    billiard = Billiard("oval", parameters=[0.2, 3], solver="polygon")
    with pytest.raises(ValueError, match="states"):
        billiard.to_cartesian(np.zeros((2, 3)))


def test_a_method_inherited_from_the_discrete_class():
    # Inside the chaotic sea, against the regular equilateral triangle region.
    billiard = Billiard("oval", parameters=[0.1, 3], solver="circle")
    assert billiard.lyapunov([2.0, 0.5], 5000)[0] > 0.0


# ! --- Escape through holes ---------------------------------------------- !


def test_a_strip_stops_at_the_first_collision_inside_it():
    billiard = Billiard("oval", parameters=[0.08, 3], solver="circle")
    state = [2.0, 1.0]
    orbit = billiard.trajectory(state, 200)
    center, width = 3.0, 0.2
    inside = np.abs(angle_difference(orbit[:, 0], center)) <= width / 2.0
    assert billiard.escape_analysis(
        state, 200, exits=[center, 0.0], hole_size=[width, np.inf]
    ) == (0, np.flatnonzero(inside)[0] + 1)


def test_a_hole_at_zero_catches_collisions_on_both_sides():
    billiard = Billiard("oval", parameters=[0.08, 3], solver="circle")
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
    billiard = Billiard("oval", parameters=[0.08, 3], solver="circle")
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
    billiard = Billiard("oval", parameters=[0.08, 3], solver="circle")
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
    billiard = Billiard("oval", parameters=[0.08, 3], solver="circle")
    states = [[2.0, a] for a in (0.5, 1.0, 1.5)]
    kwargs = dict(max_time=200, exits=[1.0, 0.0], hole_size=[0.3, np.inf])
    in_parallel = Parallel(n_jobs=2)(
        delayed(billiard.escape_analysis)(state, **kwargs) for state in states
    )
    assert in_parallel == [billiard.escape_analysis(s, **kwargs) for s in states]


# ! --- Stadium, Sinai, and mushroom -------------------------------------- !

# Independent descriptions of the boundaries, as (kind, ...) pieces: a line from one
# point to another, or a circle with the angular range it occupies.
def stadium_shape(r, l):
    half = l / 2.0
    return [
        ("arc", half, 0.0, r, -np.pi / 2.0, np.pi / 2.0),
        ("line", half, r, -half, r),
        ("arc", -half, 0.0, r, np.pi / 2.0, 3.0 * np.pi / 2.0),
        ("line", -half, -r, half, -r),
    ]


def sinai_shape(r, x_d, y_d):
    return [
        ("line", 0.0, 0.0, 1.0, 0.0),
        ("line", 1.0, 0.0, 1.0, 1.0),
        ("line", 1.0, 1.0, 0.0, 1.0),
        ("line", 0.0, 1.0, 0.0, 0.0),
        ("arc", x_d, y_d, r, 0.0, TWO_PI),
    ]


def mushroom_shape(R, w, h):
    half = w / 2.0
    return [
        ("line", -half, -h, half, -h),
        ("line", half, -h, half, 0.0),
        ("line", half, 0.0, R, 0.0),
        ("arc", 0.0, 0.0, R, 0.0, np.pi),
        ("line", -R, 0.0, -half, 0.0),
        ("line", -half, 0.0, -half, -h),
    ]


SOLVERS = ["md", "exact"]

SHAPES = {"stadium": stadium_shape, "sinai": sinai_shape, "mushroom": mushroom_shape}

NEW_BILLIARDS = {
    "stadium": [1.0, 2.0],
    "sinai": [0.2, 0.5, 0.5],
    "mushroom": [1.0, 0.5, 1.0],
}

PIECEWISE_PARAMETERS = [
    ("stadium", [1.0, 2.0]),
    ("stadium", [0.5, 0.3]),
    ("sinai", [0.2, 0.5, 0.5]),
    ("sinai", [0.1, 0.3, 0.7]),
    ("sinai", [0.4, 0.45, 0.55]),
    ("mushroom", [1.0, 0.5, 1.0]),
    ("mushroom", [1.0, 1.5, 0.3]),
    ("mushroom", [0.5, 0.2, 2.0]),
]


def exact_hit(shape, x0, y0, dx, dy):
    """Exact first hit of a ray from the boundary, ignoring the piece it starts on."""
    best = np.inf
    for piece in shape:
        if piece[0] == "line":
            _, ax, ay, bx, by = piece
            ex, ey = bx - ax, by - ay
            denominator = dx * ey - dy * ex
            if abs(denominator) < 1e-14:
                continue
            tau = ((ax - x0) * ey - (ay - y0) * ex) / denominator
            along = ((ax - x0) * dy - (ay - y0) * dx) / denominator
            if tau > 1e-9 and -1e-12 <= along <= 1.0 + 1e-12:
                best = min(best, tau)
        else:
            _, cx, cy, r, first, last = piece
            fx, fy = x0 - cx, y0 - cy
            b = fx * dx + fy * dy
            disc = b * b - (fx * fx + fy * fy - r * r)
            if disc < 0.0:
                continue
            for tau in (-b - np.sqrt(disc), -b + np.sqrt(disc)):
                if tau <= 1e-9:
                    continue
                angle = np.arctan2(fy + tau * dy, fx + tau * dx)
                if (angle - first) % TWO_PI <= (last - first) + 1e-12:
                    best = min(best, tau)
                    break
    return x0 + best * dx, y0 + best * dy


def random_arc_states(rng, billiard, size, margin=0.02):
    return np.column_stack(
        (
            rng.uniform(0.0, billiard.perimeter(), size),
            rng.uniform(margin, np.pi - margin, size),
        )
    )


def outgoing_from_state(billiard, tangent_step, state):
    """Position and direction of the particle, with the tangent from the boundary points."""
    s, alpha = state
    x0, y0 = billiard.to_cartesian(state)
    xa, ya = billiard.to_cartesian([s - tangent_step, alpha])
    xb, yb = billiard.to_cartesian([s + tangent_step, alpha])
    phi = np.arctan2(yb - ya, xb - xa)
    return x0, y0, np.cos(phi + alpha), np.sin(phi + alpha)


def near_a_corner(billiard, shape_points, x, y, distance=1e-4):
    return np.min(np.hypot(shape_points[:, 0] - x, shape_points[:, 1] - y)) < distance


def piece_ends(shape):
    points = []
    for piece in shape:
        if piece[0] == "line":
            points += [(piece[1], piece[2]), (piece[3], piece[4])]
        else:
            _, cx, cy, r, first, last = piece
            points += [
                (cx + r * np.cos(first), cy + r * np.sin(first)),
                (cx + r * np.cos(last), cy + r * np.sin(last)),
            ]
    return np.array(points)


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("name, parameters", PIECEWISE_PARAMETERS)
def test_lines_and_arcs_match_exact_collisions(rng, name, parameters, solver):
    billiard = Billiard(name, parameters=parameters, md_steps=5000, solver=solver)
    shape = SHAPES[name](*parameters)
    corners = piece_ends(shape)
    checked = 0
    for state in random_arc_states(rng, billiard, 60, margin=0.05):
        x0, y0, dx, dy = outgoing_from_state(billiard, 1e-7, state)
        if near_a_corner(billiard, corners, x0, y0, 1e-3):
            continue
        hit = exact_hit(shape, x0, y0, dx, dy)
        if near_a_corner(billiard, corners, *hit, 1e-3):
            continue
        found = billiard.to_cartesian(billiard.trajectory(state, 1)[0])
        np.testing.assert_allclose(found, hit, atol=1e-6)
        checked += 1
    assert checked > 30


@pytest.mark.parametrize(
    "name, parameters, perimeter",
    [
        ("oval", [1.0, 1.0], 8.0),
        ("stadium", [1.0, 2.0], TWO_PI + 4.0),
        ("stadium", [0.5, 0.0], np.pi),
        ("sinai", [0.2, 0.5, 0.5], 4.0 + 0.4 * np.pi),
        ("mushroom", [1.0, 0.5, 1.0], np.pi + 2.0 + 2.0 + 0.5 - 0.5),
    ],
)
def test_perimeters(name, parameters, perimeter):
    billiard = Billiard(name, parameters=parameters, solver="exact")
    assert billiard.perimeter() == pytest.approx(perimeter)


@pytest.mark.parametrize("solver", SOLVERS)
def test_stadium_bounce_between_the_straight_walls(solver):
    r, l = 1.0, 2.0
    billiard = Billiard("stadium", parameters=[r, l], solver=solver)
    top_middle = np.pi * r + l / 2.0
    bottom_middle = 2.0 * np.pi * r + l + l / 2.0
    s1, alpha1 = billiard.trajectory([top_middle, np.pi / 2.0], 1)[0]
    assert s1 == pytest.approx(bottom_middle, abs=1e-12)
    assert alpha1 == pytest.approx(np.pi / 2.0, abs=1e-12)
    s2, _ = billiard.trajectory([s1, alpha1], 1)[0]
    assert s2 == pytest.approx(top_middle, abs=1e-12)


@pytest.mark.parametrize("solver", SOLVERS)
def test_stadium_without_straight_segments_is_a_circle(rng, solver):
    r = 0.7
    billiard = Billiard("stadium", parameters=[r, 0.0], solver=solver)
    for s, alpha in random_arc_states(rng, billiard, 20, margin=0.05):
        s1, alpha1 = billiard.trajectory([s, alpha], 1)[0]
        difference = (s1 - s - 2.0 * r * alpha + np.pi * r) % (2.0 * np.pi * r) - np.pi * r
        assert difference == pytest.approx(0.0, abs=1e-11)
        assert alpha1 == pytest.approx(alpha, abs=1e-11)


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("x_d, y_d", [(0.5, 0.5), (0.3, 0.6)])
def test_sinai_head_on_hit_on_the_disk(x_d, y_d, solver):
    r = 0.1
    billiard = Billiard("sinai", parameters=[r, x_d, y_d], solver=solver)
    # From the bottom wall straight up, hitting the bottom of the disk.
    s1, alpha1 = billiard.trajectory([x_d, np.pi / 2.0], 1)[0]
    assert billiard.to_cartesian([s1, alpha1]) == pytest.approx(
        (x_d, y_d - r), abs=1e-12
    )
    assert s1 == pytest.approx(4.0 + r * np.pi / 2.0, abs=1e-12)
    assert alpha1 == pytest.approx(np.pi / 2.0, abs=1e-12)
    s2, _ = billiard.trajectory([s1, alpha1], 1)[0]
    assert s2 == pytest.approx(x_d, abs=1e-12)


@pytest.mark.parametrize("solver", SOLVERS)
def test_sinai_disk_position_changes_the_collision(solver):
    state = [0.25, np.pi / 2.0]
    blocked = Billiard("sinai", parameters=[0.1, 0.25, 0.5], solver=solver)
    free = Billiard("sinai", parameters=[0.1, 0.75, 0.5], solver=solver)
    assert blocked.to_cartesian(blocked.trajectory(state, 1)[0]) == pytest.approx(
        (0.25, 0.4), abs=1e-12
    )
    assert free.to_cartesian(free.trajectory(state, 1)[0]) == pytest.approx(
        (0.25, 1.0), abs=1e-12
    )


@pytest.mark.parametrize("solver", SOLVERS)
def test_mushroom_bounce_in_the_stem(solver):
    R, w, h = 1.0, 0.5, 1.0
    billiard = Billiard("mushroom", parameters=[R, w, h], solver=solver)
    left_middle = billiard.perimeter() - h / 2.0
    s1, alpha1 = billiard.trajectory([left_middle, np.pi / 2.0], 1)[0]
    assert billiard.to_cartesian([s1, alpha1]) == pytest.approx((w / 2.0, -h / 2.0), abs=1e-12)
    assert alpha1 == pytest.approx(np.pi / 2.0, abs=1e-12)


@pytest.mark.parametrize("solver", SOLVERS)
def test_mushroom_orbit_with_a_large_caustic_stays_in_the_hat(solver):
    # Bunimovich's divided phase space: chords of the hat keep their distance from
    # the center, R |cos(alpha)|, also after a reflection at the flat underside. If it
    # exceeds w / 2, the orbit never reaches the stem.
    R, w, h = 1.0, 0.5, 1.0
    billiard = Billiard("mushroom", parameters=[R, w, h], solver=solver)
    hat_start = w + h + (R - w / 2.0)  # The start of the arc.
    alpha = np.arccos(0.4)
    orbit = billiard.trajectory([hat_start + 0.7, alpha], 5000)
    points = billiard.to_cartesian(orbit)
    assert np.all(points[:, 1] >= -1e-12)
    on_arc = points[:, 1] > 1e-9
    np.testing.assert_allclose(R * np.abs(np.cos(orbit[on_arc, 1])), 0.4, atol=1e-7)


@pytest.mark.parametrize("solver", SOLVERS)
def test_mushroom_orbit_with_a_small_caustic_reaches_the_stem(solver):
    R, w, h = 1.0, 0.5, 1.0
    billiard = Billiard("mushroom", parameters=[R, w, h], solver=solver)
    hat_start = w + h + (R - w / 2.0)
    orbit = billiard.trajectory([hat_start + 0.7, np.arccos(0.1)], 5000)
    assert np.any(billiard.to_cartesian(orbit)[:, 1] < 0.0)


@pytest.mark.parametrize(
    "name, parameters",
    [
        ("stadium", [0.0, 1.0]),
        ("stadium", [1.0, -1.0]),
        ("sinai", [0.0, 0.5, 0.5]),
        ("sinai", [0.6, 0.5, 0.5]),
        ("sinai", [0.2, 0.1, 0.5]),
        ("sinai", [0.2, 0.5, 0.95]),
        ("mushroom", [0.0, 0.5, 1.0]),
        ("mushroom", [1.0, 0.0, 1.0]),
        ("mushroom", [1.0, 2.0, 1.0]),
        ("mushroom", [1.0, 0.5, 0.0]),
    ],
)
def test_invalid_geometry(name, parameters):
    billiard = Billiard(name, solver="exact")
    with pytest.raises(ValueError):
        billiard.trajectory([0.1, 1.0], 1, parameters=parameters)
    with pytest.raises(ValueError):
        billiard.perimeter(parameters)


def named_billiard(name, coordinates="angles", md_steps=500, solver="md"):
    return Billiard(
        name,
        parameters=NEW_BILLIARDS[name],
        coordinates=coordinates,
        md_steps=md_steps,
        solver=solver,
    )


def states_for(rng, billiard, name, size, margin):
    return np.column_stack(
        (
            rng.uniform(0.0, billiard.perimeter(), size),
            rng.uniform(margin, np.pi - margin, size),
        )
    )


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("name", list(NEW_BILLIARDS))
def test_new_billiards_stay_in_range_and_invert(rng, name, solver):
    billiard = named_billiard(name, md_steps=5000, solver=solver)
    backward = billiard._DiscreteDynamicalSystem__backwards_mapping
    parameters = billiard.get_parameters()
    period = billiard.info["period"] or billiard.perimeter()
    for state in states_for(rng, billiard, name, 25, 0.1):
        forward = billiard.trajectory(state, 1)[0]
        assert 0.0 < forward[1] < np.pi
        back = backward(forward, parameters)
        difference = (back[0] - state[0] + period / 2.0) % period - period / 2.0
        assert difference == pytest.approx(0.0, abs=1e-7)
        assert back[1] == pytest.approx(state[1], abs=1e-7)


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("name", list(NEW_BILLIARDS))
@pytest.mark.parametrize("coordinates", ["angles", "birkhoff"])
def test_new_billiards_analytic_jacobian(rng, name, coordinates, solver):
    billiard = named_billiard(name, coordinates, md_steps=5000, solver=solver)
    mapping = billiard._DiscreteDynamicalSystem__mapping
    jacobian = billiard._DiscreteDynamicalSystem__jacobian
    parameters = billiard.get_parameters()
    angles = named_billiard(name, solver=solver)
    period = billiard.perimeter()
    h = 1e-6
    compared = 0
    for state in states_for(rng, angles, name, 40, 0.3):
        if coordinates == "birkhoff":
            state = np.array([state[0], np.cos(state[1])])
        numerical = np.empty((2, 2))
        for j in range(2):
            step = np.zeros(2)
            step[j] = h
            plus = mapping(state + step, parameters)
            minus = mapping(state - step, parameters)
            numerical[0, j] = ((plus[0] - minus[0] + period / 2.0) % period - period / 2.0) / (2.0 * h)
            numerical[1, j] = (plus[1] - minus[1]) / (2.0 * h)
        analytic = jacobian(state, parameters)
        # Pieces and corners make the map discontinuous, so a difference across a
        # discontinuity is not a derivative: compare only where the map is smooth here.
        if np.max(np.abs(numerical)) > 1e3:
            continue
        np.testing.assert_allclose(analytic, numerical, rtol=1e-3, atol=1e-3)
        compared += 1
    assert compared >= 20


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("name", list(NEW_BILLIARDS))
def test_new_billiards_preserve_area_in_birkhoff_coordinates(rng, name, solver):
    billiard = named_billiard(name, "birkhoff", solver=solver)
    jacobian = billiard._DiscreteDynamicalSystem__jacobian
    parameters = billiard.get_parameters()
    perimeter = billiard.perimeter()
    states = np.column_stack(
        (rng.uniform(0.0, perimeter, 30), rng.uniform(-0.9, 0.9, 30))
    )
    for state in states:
        assert np.linalg.det(jacobian(state, parameters)) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("name", ["stadium", "sinai", "mushroom"])
def test_both_coordinates_give_the_same_orbit_on_arc_length(name, solver):
    angles = named_billiard(name, md_steps=2000, solver=solver)
    birkhoff = named_billiard(name, "birkhoff", md_steps=2000, solver=solver)
    s = 0.37
    alpha = 1.1
    orbit = angles.trajectory([s, alpha], 6)
    converted = birkhoff.trajectory([s, np.cos(alpha)], 6)
    np.testing.assert_allclose(converted[:, 0], orbit[:, 0], atol=1e-7)
    np.testing.assert_allclose(converted[:, 1], np.cos(orbit[:, 1]), atol=1e-7)


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("name", ["stadium", "sinai"])
def test_chaotic_billiards_have_a_positive_exponent(name, solver):
    state = [1.0, 1.0]
    billiard = named_billiard(name, solver=solver)
    assert billiard.lyapunov(state, 20_000)[0] > 0.1


def test_refining_the_step_keeps_the_collisions(rng):
    for name in NEW_BILLIARDS:
        coarse = named_billiard(name, md_steps=500)
        fine = named_billiard(name, md_steps=20_000)
        for state in states_for(rng, coarse, name, 40, 0.05):
            a = coarse.to_cartesian(coarse.trajectory(state, 1)[0])
            b = fine.to_cartesian(fine.trajectory(state, 1)[0])
            np.testing.assert_allclose(a, b, atol=1e-9)


def test_boundary_of_lines_and_arcs():
    stadium = Billiard("stadium", parameters=[1.0, 2.0], solver="exact").boundary(400)
    assert np.isfinite(stadium).all()
    np.testing.assert_allclose(stadium[0], stadium[-1], atol=1e-12)
    # Corners of the straight segments are included exactly.
    assert np.min(np.hypot(stadium[:, 0] - 1.0, stadium[:, 1] - 1.0)) < 1e-12

    sinai = Billiard("sinai", parameters=[0.2, 0.3, 0.4], solver="exact").boundary(400)
    gap = np.flatnonzero(np.isnan(sinai[:, 0]))
    assert gap.size == 1
    walls, disk = sinai[: gap[0]], sinai[gap[0] + 1 :]
    assert np.min(walls) == pytest.approx(0.0) and np.max(walls) == pytest.approx(1.0)
    np.testing.assert_allclose(np.hypot(disk[:, 0] - 0.3, disk[:, 1] - 0.4), 0.2)


@pytest.mark.parametrize("solver", SOLVERS)
def test_to_cartesian_on_the_boundary_of_every_piece(solver):
    mushroom = Billiard("mushroom", parameters=[1.0, 0.5, 1.0], solver=solver)
    orbit = mushroom.trajectory([3.0, 1.0], 500)
    x, y = mushroom.to_cartesian(orbit).T
    in_hat = y > 1e-9
    in_stem = y < -1e-9
    np.testing.assert_allclose(np.hypot(x[in_hat], y[in_hat]), 1.0, atol=1e-12)
    bottom = np.abs(y[in_stem] + 1.0) < 1e-9
    assert np.all(np.abs(np.abs(x[in_stem][~bottom]) - 0.25) < 1e-9)


@pytest.mark.parametrize("solver", SOLVERS)
def test_stadium_hole_wraps_with_the_perimeter(solver):
    billiard = Billiard("stadium", parameters=[1.0, 2.0], solver=solver)
    perimeter = billiard.perimeter()
    assert billiard.info["period"] == pytest.approx(perimeter)
    state = [3.0, 1.0]
    orbit = billiard.trajectory(state, 500)
    width = 0.3
    index, time = billiard.escape_analysis(
        state, 500, exits=[0.0, 0.0], hole_size=[width, np.inf]
    )
    inside = (orbit[:, 0] <= width / 2.0) | (orbit[:, 0] >= perimeter - width / 2.0)
    assert (index, time) == (0, np.flatnonzero(inside)[0] + 1)
    assert (orbit[inside, 0] > perimeter / 2.0).any()


@pytest.mark.parametrize("solver", SOLVERS)
def test_sinai_hole_does_not_wrap(solver):
    billiard = Billiard("sinai", parameters=[0.2, 0.5, 0.5], solver=solver)
    assert billiard.info["period"] is None
    state = [0.6, 1.0]
    orbit = billiard.trajectory(state, 2000)
    width = 0.2
    _, time = billiard.escape_analysis(
        state, 2000, exits=[0.0, 0.0], hole_size=[width, np.inf]
    )
    inside = orbit[:, 0] <= width / 2.0
    assert time == np.flatnonzero(inside)[0] + 1
    # The end of the disk, just below the perimeter, is not next to the origin.
    near_the_end = orbit[:, 0] >= billiard.perimeter() - width / 2.0
    assert near_the_end[: time - 1].any()


@pytest.mark.parametrize("solver", SOLVERS)
def test_holes_of_different_widths_on_the_walls_and_disk_of_sinai(solver):
    billiard = Billiard("sinai", parameters=[0.2, 0.5, 0.5], solver=solver)
    state = [0.6, 1.0]
    orbit = billiard.trajectory(state, 500)
    centers = np.array([[0.5, 0.0], [4.0 + 0.2 * np.pi, 0.0]])
    widths = np.array([0.05, 0.4])
    index, time = billiard.escape_analysis(
        state,
        500,
        exits=centers,
        hole_size=np.column_stack((widths, np.full(2, np.inf))),
    )
    inside = np.abs(orbit[:, 0, None] - centers[:, 0]) <= widths / 2.0
    first = np.flatnonzero(inside.any(axis=1))[0]
    assert time == first + 1
    assert inside[first, index]


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("name", list(NEW_BILLIARDS))
def test_parallel_workers_for_every_billiard(name, solver):
    billiard = named_billiard(name, solver=solver)
    states = [[0.4, a] for a in (0.5, 1.0, 1.5)]
    kwargs = dict(max_time=100, exits=[1.0, 0.0], hole_size=[0.2, np.inf])
    in_parallel = Parallel(n_jobs=2)(
        delayed(billiard.escape_analysis)(state, **kwargs) for state in states
    )
    assert in_parallel == [billiard.escape_analysis(s, **kwargs) for s in states]


@pytest.mark.parametrize("name, parameters", PIECEWISE_PARAMETERS)
def test_exact_solver_matches_molecular_dynamics_along_an_orbit(name, parameters):
    exact = Billiard(name, parameters=parameters, solver="exact")
    md = Billiard(name, parameters=parameters, solver="md", md_steps=20_000)
    state = np.array([0.37, 1.1])
    corners = piece_ends(SHAPES[name](*parameters))
    for next_state in exact.trajectory(state, 40):
        found = md.trajectory(state, 1)[0]
        point = exact.to_cartesian(next_state)
        # A corner within one fine step can be passed by molecular dynamics.
        if not near_a_corner(md, corners, *point, 1e-3):
            np.testing.assert_allclose(found[0], next_state[0], atol=1e-8)
            np.testing.assert_allclose(found[1], next_state[1], atol=1e-8)
        state = next_state


def test_exact_solver_catches_a_collision_that_molecular_dynamics_steps_over():
    # The trajectory crosses the underside of the hat just outside the stem, and is
    # already inside the stem at the next molecular dynamics step.
    R, w, h = 1.0, 0.5, 1.0
    phi = 0.5
    start = np.array([R * np.cos(phi), R * np.sin(phi)])
    target = np.array([w / 2.0 + 1e-4, 0.0])
    direction = np.arctan2(*(target - start)[::-1])
    alpha = (direction - (phi + np.pi / 2.0)) % TWO_PI
    state = [w + h + (R - w / 2.0) + R * phi, alpha]

    exact = Billiard("mushroom", parameters=[R, w, h], solver="exact")
    md = Billiard("mushroom", parameters=[R, w, h], solver="md")
    assert exact.to_cartesian(exact.trajectory(state, 1)[0]) == pytest.approx(
        tuple(target), abs=1e-12
    )
    x, y = md.to_cartesian(md.trajectory(state, 1)[0])
    assert y < -1e-3


def test_available_solvers():
    for name in ("stadium", "sinai", "mushroom"):
        assert Billiard.available_solvers(name) == ["md", "exact"]


def test_info_reports_the_solver():
    assert Billiard("stadium", solver="exact").info["solver"] == "exact"


def test_exact_solver_with_a_zero_length_straight_segment():
    billiard = Billiard("stadium", parameters=[1.0, 0.0], solver="exact")
    s1, alpha1 = billiard.trajectory([1.0, 0.4], 1)[0]
    assert s1 == pytest.approx(1.8, abs=1e-12)
    assert alpha1 == pytest.approx(0.4, abs=1e-12)


# ! --- The oval billiard at eps = 1 --------------------------------------- !


def exact_cardioid_hit(theta, alpha):
    x0, y0, dx, dy = oval_outgoing(theta, alpha, 1.0, 1)
    # On the line, (x^2 + y^2 - x)^2 = x^2 + y^2 is a quartic in tau with the root 0.
    b = 2.0 * (x0 * dx + y0 * dy)
    c = x0 * x0 + y0 * y0
    area = np.poly1d([1.0, b, c])
    shifted = np.poly1d([1.0, b - dx, c - x0])
    quartic = shifted * shifted - area
    roots = np.roots(np.polydiv(quartic, np.poly1d([1.0, 0.0]))[0])
    real = roots[np.abs(roots.imag) < 1e-9].real
    tau = real[real > 1e-9].min()
    return x0 + tau * dx, y0 + tau * dy


def test_cardioid_matches_the_exact_cubic(rng):
    billiard = Billiard("oval", parameters=[1.0, 1.0], md_steps=5000, solver="exact")
    for theta, alpha in random_states(rng, 60, margin=0.05):
        if abs(theta - np.pi) < 0.1:  # Away from the cusp.
            continue
        found = billiard.to_cartesian(billiard.trajectory([theta, alpha], 1)[0])
        np.testing.assert_allclose(found, exact_cardioid_hit(theta, alpha), atol=1e-9)


def test_cardioid_arc_length_is_the_closed_form(rng):
    parameters = np.array([1.0, 1.0])
    for theta in rng.uniform(0.0, np.pi, 20):
        s = 4.0 * np.sin(theta / 2.0)
        assert oval_arc_length(theta, 0.0, parameters) == pytest.approx(s, abs=1e-12)
        assert oval_arc_length(TWO_PI - theta, 0.0, parameters) == pytest.approx(
            8.0 - s, abs=1e-12
        )


def test_cardioid_is_chaotic_and_preserves_area():
    billiard = Billiard("oval", parameters=[1.0, 1.0], coordinates="birkhoff", solver="exact")
    exponents = billiard.lyapunov([1.0, 0.3], 20_000)
    assert exponents[0] > 0.1
    assert np.sum(exponents) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("p", [1, 2, 3])
def test_collisions_at_eps_one_match_a_dense_scan(rng, p):
    billiard = Billiard("oval", parameters=[1.0, p], md_steps=5000, solver=oval_solver(1.0, p))
    for theta, alpha in random_states(rng, 15, margin=0.05):
        # Away from the points where the boundary meets the origin.
        if np.cos(p * theta) < -0.9:
            continue
        theta1, alpha1 = billiard.trajectory([theta, alpha], 1)[0]
        expected = scanned_collision(theta, alpha, 1.0, p)
        assert angle_difference(theta1, expected) == pytest.approx(0.0, abs=1e-8)
        assert 0.0 < alpha1 < np.pi


def test_lobes_at_eps_one_keep_the_trajectory():
    # With p = 3, each lobe lies within an angle of pi / 3 on each side of its axis.
    billiard = Billiard("oval", parameters=[1.0, 3], solver="md")
    theta = billiard.trajectory([0.3, 1.2], 2000)[:, 0]
    assert np.all(np.abs(angle_difference(theta, 0.0)) <= np.pi / 3.0 + 1e-9)


# ! --- Oval solvers ------------------------------------------------------- !


def critical_eps(p):
    return 1.0 / (1.0 + p * p)


CIRCLE_PARAMETERS = [
    (f * critical_eps(p), p) for p in (2, 3, 4, 5, 8) for f in (0.0, 0.3, 0.8, 1.0)
]
POLYGON_PARAMETERS = [
    (eps, p)
    for p in (2, 3, 4, 5, 8)
    for eps in (1.1 * critical_eps(p), 2.0 * critical_eps(p), 0.38, 0.6, 0.9, 0.99, 0.999)
    if eps > critical_eps(p)
]
EXACT_PARAMETERS = [(eps, 1) for eps in (0.0, 0.1, 0.5, 0.6, 0.9, 0.99, 1.0)]
SOLVER_PARAMETERS = (
    [("circle", eps, p) for eps, p in CIRCLE_PARAMETERS]
    + [("polygon", eps, p) for eps, p in POLYGON_PARAMETERS]
    + [("exact", eps, p) for eps, p in EXACT_PARAMETERS]
)


def oval_ray(theta, alpha, eps, p):
    parameters = np.array([eps, float(p)])
    x0, y0 = oval_boundary_point(theta, 0.0, parameters)
    tx, ty = oval_tangent(theta, 0.0, parameters)
    return (
        x0,
        y0,
        np.cos(alpha) * tx - np.sin(alpha) * ty,
        np.sin(alpha) * tx + np.cos(alpha) * ty,
    )


def first_root_along(x0, y0, dx, dy, eps, p, around=None):
    """The first crossing of the boundary, on a grid refined around a given distance."""

    def gap(tau):
        x, y = x0 + tau * dx, y0 + tau * dy
        return np.hypot(x, y) - oval_radius(np.arctan2(y, x), eps, p)

    grids = [
        np.geomspace(1e-9, 1e-2, 2000, endpoint=False),
        np.linspace(1e-2, 2.0 * (1.0 + eps), 100_000),
    ]
    if around is not None:
        grids.append(np.linspace(max(around - 0.05, 1e-9), around + 0.05, 200_000))
    taus = np.unique(np.concatenate(grids))
    first = np.flatnonzero(gap(taus) > 0.0)[0]

    return brentq(gap, taus[first - 1], taus[first], xtol=1e-14)


@pytest.mark.parametrize("solver, eps, p", SOLVER_PARAMETERS)
def test_oval_solvers_match_a_dense_scan_in_the_full_angle(rng, solver, eps, p):
    billiard = Billiard("oval", parameters=[eps, p], solver=solver)
    states = random_states(rng, 12, margin=0.002)
    for theta, alpha in states:
        if eps == 1.0 and abs(theta - np.pi) < 0.05:  # Away from the cusp.
            continue
        theta1, alpha1 = billiard.trajectory([theta, alpha], 1)[0]
        x0, y0, dx, dy = oval_ray(theta, alpha, eps, p)
        # The grid is refined where the solver lands, because the necks of the boundary
        # at eps close to one are narrower than the coarser grid.
        x1, y1 = billiard.to_cartesian([theta1, alpha1])
        distance = first_root_along(x0, y0, dx, dy, eps, p, np.hypot(x1 - x0, y1 - y0))
        expected = np.arctan2(y0 + distance * dy, x0 + distance * dx)
        # As a length along the boundary, where a small radius magnifies the angle.
        error = angle_difference(theta1, expected) * oval_speed(expected, eps, p)
        assert error == pytest.approx(0.0, abs=1e-8)
        assert 0.0 < alpha1 < np.pi


@pytest.mark.parametrize(
    "solver, eps, p",
    [("circle", 0.05, 3), ("circle", 0.1, 3), ("polygon", 0.2, 3), ("polygon", 0.54, 3)]
    + [("polygon", 0.4, 2), ("polygon", 0.3, 5), ("exact", 0.3, 1), ("exact", 0.8, 1)],
)
def test_oval_solvers_match_molecular_dynamics_along_an_orbit(solver, eps, p):
    fast = Billiard("oval", parameters=[eps, p], solver=solver)
    md = Billiard("oval", parameters=[eps, p], solver="md", md_steps=20_000)
    state = np.array([1.3, 0.9])
    for next_state in fast.trajectory(state, 200):
        theta1, alpha1 = md.trajectory(state, 1)[0]
        assert angle_difference(theta1, next_state[0]) == pytest.approx(0.0, abs=1e-8)
        assert alpha1 == pytest.approx(next_state[1], abs=1e-8)
        state = next_state


@pytest.mark.parametrize("eps, p", POLYGON_PARAMETERS)
def test_polygon_converges_without_the_fallback(rng, eps, p):
    parameters = np.array([eps, float(p)])
    for theta, alpha in random_states(rng, 200, margin=0.002):
        _, _, dx, dy = oval_ray(theta, alpha, eps, p)
        assert not np.isnan(oval_polygon_distance(theta, dx, dy, parameters))


@pytest.mark.parametrize("eps, p", CIRCLE_PARAMETERS)
def test_circle_converges_without_the_fallback(rng, eps, p):
    parameters = np.array([eps, float(p)])
    for theta, alpha in random_states(rng, 200, margin=0.002):
        x0, y0, dx, dy = oval_ray(theta, alpha, eps, p)
        assert not np.isnan(oval_circle_distance(x0, y0, dx, dy, parameters))


@pytest.mark.parametrize("eps, p", [(0.13, 3), (0.4, 3), (0.9, 3), (0.8, 2), (0.5, 5)])
def test_polygon_when_a_trajectory_barely_hits_or_misses_the_inward_bulge(rng, eps, p):
    # The hard case of the seed: a first collision that depends on a very small change.
    parameters = np.array([eps, float(p)])
    wedge = np.pi / p
    hits = misses = 0
    while hits + misses < 25:
        bulge = rng.uniform(0.8 * wedge, wedge)
        xb, yb = oval_boundary_point(bulge, 0.0, parameters)
        tx, ty = oval_tangent(bulge, 0.0, parameters)
        theta = rng.uniform(0.0, TWO_PI)
        x0, y0 = oval_boundary_point(theta, 0.0, parameters)
        offset = rng.choice([-1.0, 1.0]) * 10.0 ** rng.uniform(-8.0, -3.0)
        px, py = xb - ty * offset, yb + tx * offset
        angle = np.arctan2(py - y0, px - x0)
        dx, dy = np.cos(angle), np.sin(angle)
        start_x, start_y = oval_tangent(theta, 0.0, parameters)
        if start_x * dy - start_y * dx < 0.01 or abs(tx * dy - ty * dx) > 0.3:
            continue
        around = np.hypot(px - x0, py - y0)
        expected = first_root_along(x0, y0, dx, dy, eps, p, around)
        if abs(expected - around) < 0.05:
            hits += 1
        else:
            misses += 1
        assert oval_polygon_distance(theta, dx, dy, parameters) == pytest.approx(
            expected, abs=1e-8
        )
    assert hits > 0


def test_polygon_finds_the_first_collision_of_da_costa_fig_9():
    billiard = Billiard("oval", parameters=[0.4, 3], solver="polygon")
    x, y = billiard.to_cartesian(billiard.trajectory([0.55, 0.1442], 1)[0])
    assert x == pytest.approx(0.417, abs=0.01)
    assert y == pytest.approx(0.5, abs=0.01)


@pytest.mark.parametrize("start", [0.3, 1.5, 2.8, 4.4, 5.9])
def test_polygon_recovers_the_angle_across_many_wedges(start):
    # Through the middle of the billiard, the trajectory crosses several wedges.
    eps, p = 0.6, 5
    billiard = Billiard("oval", parameters=[eps, p], solver="polygon")
    theta1, _ = billiard.trajectory([start, np.pi / 2.0], 1)[0]
    x0, y0, dx, dy = oval_ray(start, np.pi / 2.0, eps, p)
    distance = first_root_along(x0, y0, dx, dy, eps, p)
    assert distance > 0.5
    assert angle_difference(
        theta1, np.arctan2(y0 + distance * dy, x0 + distance * dx)
    ) == pytest.approx(0.0, abs=1e-10)


@pytest.mark.parametrize(
    "solver, parameters, message",
    [
        ("circle", [0.2, 3], "circle"),
        ("circle", [0.6, 1], "circle"),
        ("polygon", [0.05, 3], "polygon"),
        ("polygon", [1.0, 3], "polygon"),
        ("polygon", [0.8, 1], "polygon"),
        ("exact", [0.1, 3], "exact"),
    ],
)
def test_a_solver_outside_its_range_raises(solver, parameters, message):
    billiard = Billiard("oval", parameters=parameters, solver=solver)
    with pytest.raises(ValueError, match=message):
        billiard.trajectory([1.0, 1.0], 1)


@pytest.mark.parametrize(
    "solver, eps, p", [("circle", 0.08, 3), ("polygon", 0.4, 3), ("exact", 0.5, 1)]
)
def test_oval_solvers_in_birkhoff_coordinates(rng, solver, eps, p):
    billiard = Billiard(
        "oval", parameters=[eps, p], coordinates="birkhoff", solver=solver
    )
    jacobian = billiard._DiscreteDynamicalSystem__jacobian
    backward = billiard._DiscreteDynamicalSystem__backwards_mapping
    parameters = billiard.get_parameters()
    perimeter = billiard.perimeter()
    for state in birkhoff_states(rng, [eps, p], 20):
        assert np.linalg.det(jacobian(state, parameters)) == pytest.approx(1.0, abs=1e-7)
        back = backward(billiard.trajectory(state, 1)[0], parameters)
        difference = (back[0] - state[0] + perimeter / 2.0) % perimeter - perimeter / 2.0
        assert difference == pytest.approx(0.0, abs=1e-8)
        assert back[1] == pytest.approx(state[1], abs=1e-8)


@pytest.mark.parametrize(
    "solver, eps, p", [("circle", 0.08, 3), ("polygon", 0.4, 3), ("exact", 0.5, 1)]
)
def test_oval_solvers_give_the_same_escape_as_molecular_dynamics(solver, eps, p):
    fast = Billiard("oval", parameters=[eps, p], solver=solver)
    md = Billiard("oval", parameters=[eps, p], solver="md", md_steps=5000)
    # Few collisions, before round-off grows along the chaotic orbit.
    kwargs = dict(max_time=15, exits=[1.0, 0.0], hole_size=[1.0, np.inf])
    for alpha in (0.4, 1.0, 1.6, 2.5):
        state = [2.0, alpha]
        assert fast.escape_analysis(state, **kwargs) == md.escape_analysis(state, **kwargs)


def test_parallel_workers_with_a_fast_solver():
    billiard = Billiard("oval", parameters=[0.3, 3], solver="polygon")
    states = [[2.0, a] for a in (0.5, 1.0, 1.5)]
    kwargs = dict(max_time=100, exits=[1.0, 0.0], hole_size=[0.2, np.inf])
    in_parallel = Parallel(n_jobs=2)(
        delayed(billiard.escape_analysis)(state, **kwargs) for state in states
    )
    assert in_parallel == [billiard.escape_analysis(s, **kwargs) for s in states]


# ! --- Billiards defined by users ----------------------------------------- !


@njit
def oval_radius_user(theta, parameters):
    return 1.0 + parameters[0] * np.cos(parameters[1] * theta)


@njit
def oval_dradius_user(theta, parameters):
    return -parameters[0] * parameters[1] * np.sin(parameters[1] * theta)


@njit
def oval_ddradius_user(theta, parameters):
    return -parameters[0] * parameters[1] ** 2 * np.cos(parameters[1] * theta)


@njit
def oval_max_radius_user(parameters):
    return 1.0 + parameters[0]


@njit
def two_mode_radius(theta, parameters):
    return 1.0 + 0.1 * np.cos(2.0 * theta) + 0.05 * np.sin(5.0 * theta)


@njit
def two_mode_dradius(theta, parameters):
    return -0.2 * np.sin(2.0 * theta) + 0.25 * np.cos(5.0 * theta)


@njit
def two_mode_ddradius(theta, parameters):
    return -0.4 * np.cos(2.0 * theta) - 1.25 * np.sin(5.0 * theta)


def user_oval(parameters, **kwargs):
    return Billiard.from_polar(
        oval_radius_user,
        oval_dradius_user,
        ddradius=oval_ddradius_user,
        max_radius=oval_max_radius_user,
        parameters=parameters,
        **kwargs,
    )


def stadium_as_pieces(r, l):
    half = l / 2.0
    return [
        ("arc", half, 0.0, r, -np.pi / 2.0, np.pi),
        ("line", half, r, -half, r),
        ("arc", -half, 0.0, r, np.pi / 2.0, np.pi),
        ("line", -half, -r, half, -r),
    ]


def sinai_as_pieces(r, x_d, y_d):
    return [
        ("line", 0.0, 0.0, 1.0, 0.0),
        ("line", 1.0, 0.0, 1.0, 1.0),
        ("line", 1.0, 1.0, 0.0, 1.0),
        ("line", 0.0, 1.0, 0.0, 0.0),
        ("arc", x_d, y_d, r, 0.0, -TWO_PI),
    ]


def mushroom_as_pieces(R, w, h):
    half = w / 2.0
    return [
        ("line", -half, -h, half, -h),
        ("line", half, -h, half, 0.0),
        ("line", half, 0.0, R, 0.0),
        ("arc", 0.0, 0.0, R, 0.0, np.pi),
        ("line", -R, 0.0, -half, 0.0),
        ("line", -half, 0.0, -half, -h),
    ]


AS_PIECES = {
    "stadium": stadium_as_pieces,
    "sinai": sinai_as_pieces,
    "mushroom": mushroom_as_pieces,
}


@njit
def stadium_table(parameters):
    r, l = parameters
    half = l / 2.0
    table = new_pieces(4)
    set_arc(table, 0, half, 0.0, r, -np.pi / 2.0, np.pi)
    set_line(table, 1, half, r, -half, r)
    set_arc(table, 2, -half, 0.0, r, np.pi / 2.0, np.pi)
    set_line(table, 3, -half, -r, half, -r)
    return table


@pytest.mark.parametrize("eps, p", [(0.05, 3), (0.08, 3), (0.1, 3), (0.1, 2)])
@pytest.mark.parametrize("solver", ["md", "circle"])
def test_from_polar_reproduces_the_oval(rng, eps, p, solver):
    user = user_oval([eps, p], solver=solver, md_steps=5000)
    builtin = Billiard("oval", parameters=[eps, p], solver="md", md_steps=5000)
    for state in random_states(rng, 10, margin=0.01):
        np.testing.assert_allclose(
            user.trajectory(state, 1), builtin.trajectory(state, 1), atol=1e-10
        )
        np.testing.assert_allclose(
            user._DiscreteDynamicalSystem__jacobian(state, user.get_parameters()),
            builtin._DiscreteDynamicalSystem__jacobian(state, builtin.get_parameters()),
            atol=1e-7,
        )


@pytest.mark.parametrize("eps, p", [(0.08, 3), (0.4, 3), (0.2, 5)])
def test_from_polar_reproduces_the_oval_with_negative_curvature(rng, eps, p):
    user = user_oval([eps, p], solver="md", md_steps=5000)
    builtin = Billiard("oval", parameters=[eps, p], solver="md", md_steps=5000)
    for state in random_states(rng, 10, margin=0.01):
        np.testing.assert_allclose(
            user.trajectory(state, 1), builtin.trajectory(state, 1), atol=1e-10
        )


@pytest.mark.parametrize("eps, p", [(0.08, 3), (0.4, 3), (0.2, 5), (0.5, 2)])
def test_from_polar_arc_length_matches_the_oval(rng, eps, p):
    user = user_oval([eps, p], solver="md", coordinates="birkhoff")
    builtin = Billiard("oval", parameters=[eps, p], coordinates="birkhoff", solver=oval_solver(eps, p))
    assert user.perimeter() == pytest.approx(builtin.perimeter(), abs=1e-11)

    values = np.array([eps, p], dtype=np.float64)
    for s in rng.uniform(0.0, user.perimeter(), 10):
        state = np.array([s, 0.3])
        np.testing.assert_allclose(
            user.to_cartesian(state), builtin.to_cartesian(state), atol=1e-10
        )
    assert values.size == 2


def test_two_mode_boundary_matches_a_dense_scan(rng):
    billiard = Billiard.from_polar(
        two_mode_radius, two_mode_dradius, ddradius=two_mode_ddradius, md_steps=5000,
        solver="md",
    )
    for theta, alpha in random_states(rng, 20, margin=0.01):
        r0 = two_mode_radius(theta, np.zeros(0))
        dr0 = two_mode_dradius(theta, np.zeros(0))
        phi = np.arctan2(dr0 * np.sin(theta) + r0 * np.cos(theta), dr0 * np.cos(theta) - r0 * np.sin(theta))
        dx, dy = np.cos(phi + alpha), np.sin(phi + alpha)
        x0, y0 = r0 * np.cos(theta), r0 * np.sin(theta)

        def gap(tau):
            x, y = x0 + tau * dx, y0 + tau * dy
            angle = np.arctan2(y, x)
            return np.hypot(x, y) - (
                1.0 + 0.1 * np.cos(2.0 * angle) + 0.05 * np.sin(5.0 * angle)
            )

        taus = np.concatenate(
            (np.geomspace(1e-9, 1e-2, 1000, endpoint=False), np.linspace(1e-2, 3.0, 60_000))
        )
        first = np.flatnonzero(gap(taus) > 0.0)[0]
        distance = brentq(gap, taus[first - 1], taus[first], xtol=1e-14)
        theta1, alpha1 = billiard.trajectory([theta, alpha], 1)[0]
        assert angle_difference(
            theta1, np.arctan2(y0 + distance * dy, x0 + distance * dx)
        ) == pytest.approx(0.0, abs=1e-9)
        assert 0.0 < alpha1 < np.pi


def test_user_polar_invariants(rng):
    billiard = Billiard.from_polar(
        two_mode_radius,
        two_mode_dradius,
        ddradius=two_mode_ddradius,
        coordinates="birkhoff",
        solver="md",
    )
    jacobian = billiard._DiscreteDynamicalSystem__jacobian
    backward = billiard._DiscreteDynamicalSystem__backwards_mapping
    values = np.zeros(0)
    perimeter = billiard.perimeter()
    for s, c in zip(rng.uniform(0.0, perimeter, 20), rng.uniform(-0.9, 0.9, 20)):
        state = np.array([s, c])
        assert np.linalg.det(jacobian(state, values)) == pytest.approx(1.0, abs=1e-7)
        back = backward(billiard.trajectory(state, 1)[0], values)
        difference = (back[0] - s + perimeter / 2.0) % perimeter - perimeter / 2.0
        assert difference == pytest.approx(0.0, abs=1e-8)
        assert back[1] == pytest.approx(c, abs=1e-8)


def test_user_polar_without_the_second_derivative_differentiates_numerically():
    billiard = Billiard.from_polar(two_mode_radius, two_mode_dradius, solver="md")
    assert billiard.lyapunov([1.0, 1.0], 3000)[0] > 0.0
    jacobian = billiard._DiscreteDynamicalSystem__jacobian
    mapping = billiard._DiscreteDynamicalSystem__mapping
    J = jacobian(np.array([1.0, 1.0]), np.zeros(0), mapping)
    assert J.shape == (2, 2)


def test_user_polar_escape():
    billiard = user_oval([0.08, 3], solver="circle")
    state = [2.0, 1.0]
    orbit = billiard.trajectory(state, 500)
    index, time = billiard.escape_analysis(
        state, 500, exits=[0.0, 0.0], hole_size=[0.2, np.inf]
    )
    inside = np.abs(angle_difference(orbit[:, 0], 0.0)) <= 0.1
    assert (index, time) == (0, np.flatnonzero(inside)[0] + 1)
    assert billiard.info["period"] == pytest.approx(TWO_PI)


def test_circle_on_a_user_boundary_that_is_not_convex_raises():
    with pytest.raises(ValueError, match="convex"):
        user_oval([0.3, 3], solver="circle")
    billiard = user_oval([0.08, 3], solver="circle")
    with pytest.raises(ValueError, match="convex"):
        billiard.set_parameters([0.3, 3])


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(solver="circle"), "not available"),
        (dict(solver="polygon"), "not available"),
        (dict(solver="exact"), "not available"),
    ],
)
def test_unavailable_solver_on_a_user_polar_boundary(kwargs, match):
    with pytest.raises(ValueError, match=match):
        Billiard.from_polar(two_mode_radius, two_mode_dradius, **kwargs)


@njit
def negative_radius(theta, parameters):
    return -1.0 + 0.0 * theta


@njit
def zero_derivative(theta, parameters):
    return 0.0 * theta


@njit
def not_periodic(theta, parameters):
    return 1.0 + 0.1 * theta


@njit
def not_periodic_derivative(theta, parameters):
    return 0.1 + 0.0 * theta


@njit
def too_small(parameters):
    return 0.5


@njit
def always_invalid(parameters):
    raise ValueError("these parameters are never valid")


def test_wrong_polar_functions_raise():
    with pytest.raises(ValueError, match="positive"):
        Billiard.from_polar(negative_radius, zero_derivative, solver="md")
    with pytest.raises(ValueError, match="period"):
        Billiard.from_polar(not_periodic, not_periodic_derivative, solver="md")
    with pytest.raises(ValueError, match="dradius"):
        Billiard.from_polar(two_mode_radius, zero_derivative, solver="md")
    with pytest.raises(ValueError, match="ddradius"):
        Billiard.from_polar(two_mode_radius, two_mode_dradius, ddradius=zero_derivative, solver="md")
    with pytest.raises(ValueError, match="max_radius"):
        Billiard.from_polar(two_mode_radius, two_mode_dradius, max_radius=too_small, solver="md")


def test_functions_must_be_compiled():
    def plain(theta, parameters):
        return 1.0

    with pytest.raises(TypeError, match="radius"):
        Billiard.from_polar(plain, two_mode_dradius, solver="md")
    with pytest.raises(TypeError, match="dradius"):
        Billiard.from_polar(two_mode_radius, plain, solver="md")
    with pytest.raises(TypeError, match="pieces"):
        Billiard.from_pieces(plain, solver="exact")
    with pytest.raises(TypeError, match="inside"):
        Billiard.from_pieces(stadium_as_pieces(1.0, 1.0), inside=plain, solver="exact")


def test_user_check_function():
    with pytest.raises(ValueError, match="never valid"):
        Billiard.from_polar(
            oval_radius_user,
            oval_dradius_user,
            check=always_invalid,
            parameters=[0.1, 3],
            solver="md",
        )


@pytest.mark.parametrize("name", list(AS_PIECES))
@pytest.mark.parametrize("solver", ["exact", "md"])
def test_from_pieces_reproduces_the_builtin_billiards(rng, name, solver):
    parameters = NEW_BILLIARDS[name]
    user = Billiard.from_pieces(AS_PIECES[name](*parameters), solver=solver)
    builtin = Billiard(name, parameters=parameters, solver=solver)
    assert user.perimeter() == pytest.approx(builtin.perimeter())
    for state in states_for(rng, builtin, name, 30, 0.05):
        np.testing.assert_allclose(
            user.trajectory(state, 1), builtin.trajectory(state, 1), atol=1e-10
        )


@pytest.mark.parametrize("name", list(AS_PIECES))
def test_inside_from_the_pieces_agrees_with_the_formula(name):
    parameters = np.array(NEW_BILLIARDS[name], dtype=np.float64)
    formula = {
        "stadium": stadium_inside,
        "sinai": sinai_inside,
        "mushroom": mushroom_inside,
    }[name]
    table = _fixed_pieces_table(AS_PIECES[name](*parameters))
    # The grid includes the heights of ends of pieces, such as y = 0 on the mushroom.
    # The shift keeps the points off the vertical walls and the ends of the arcs.
    for x in np.linspace(-3.0, 3.0, 121) + 0.00137:
        for y in np.linspace(-1.5, 1.5, 61):
            # Points on the boundary itself may go either way.
            if name == "mushroom" and y == 0.0 and abs(x) >= 0.25:
                continue
            if name == "stadium" and abs(abs(y) - 1.0) < 1e-12:
                continue
            if name == "sinai" and (abs(y - 1.0) < 1e-12 or abs(y) < 1e-12):
                continue
            if name == "mushroom" and abs(y + 1.0) < 1e-12:
                continue
            assert inside_pieces(table, x, y) == formula(x, y, 0.0, parameters), (x, y)


def test_shape_with_a_hole_matches_exact_intersections(rng):
    # A rectangle with two disks.
    pieces = [
        ("line", 0.0, 0.0, 3.0, 0.0),
        ("line", 3.0, 0.0, 3.0, 1.0),
        ("line", 3.0, 1.0, 0.0, 1.0),
        ("line", 0.0, 1.0, 0.0, 0.0),
        ("arc", 1.0, 0.5, 0.2, 0.0, -TWO_PI),
        ("arc", 2.0, 0.5, 0.3, np.pi / 2.0, -TWO_PI),
    ]
    shape = [
        ("line", 0.0, 0.0, 3.0, 0.0),
        ("line", 3.0, 0.0, 3.0, 1.0),
        ("line", 3.0, 1.0, 0.0, 1.0),
        ("line", 0.0, 1.0, 0.0, 0.0),
        ("arc", 1.0, 0.5, 0.2, 0.0, TWO_PI),
        ("arc", 2.0, 0.5, 0.3, 0.0, TWO_PI),
    ]
    corners = np.array([[0, 0], [3, 0], [3, 1], [0, 1]], dtype=float)
    billiard = Billiard.from_pieces(pieces, md_steps=5000, solver="exact")
    md = Billiard.from_pieces(pieces, solver="md", md_steps=5000)
    assert billiard.info["period"] is None
    for state in random_arc_states(rng, billiard, 40, margin=0.05):
        x0, y0, dx, dy = outgoing_from_state(billiard, 1e-7, state)
        if near_a_corner(billiard, corners, x0, y0, 1e-3):
            continue
        hit = exact_hit(shape, x0, y0, dx, dy)
        found = billiard.to_cartesian(billiard.trajectory(state, 1)[0])
        np.testing.assert_allclose(found, hit, atol=1e-6)
        np.testing.assert_allclose(
            md.to_cartesian(md.trajectory(state, 1)[0]), found, atol=1e-8
        )


def test_pieces_with_parameters_match_the_stadium(rng):
    user = Billiard.from_pieces(stadium_table, parameters=[1.0, 2.0], solver="exact")
    builtin = Billiard("stadium", parameters=[1.0, 2.0], solver="exact")
    for state in states_for(rng, builtin, "stadium", 20, 0.05):
        np.testing.assert_allclose(
            user.trajectory(state, 1), builtin.trajectory(state, 1), atol=1e-12
        )
    assert user.perimeter([0.5, 1.0]) == pytest.approx(np.pi + 2.0)


def test_user_pieces_invariants(rng):
    billiard = Billiard.from_pieces(mushroom_as_pieces(1.0, 0.5, 1.0), coordinates="birkhoff", solver="exact")
    jacobian = billiard._DiscreteDynamicalSystem__jacobian
    backward = billiard._DiscreteDynamicalSystem__backwards_mapping
    values = np.zeros(0)
    perimeter = billiard.perimeter()
    for s, c in zip(rng.uniform(0.0, perimeter, 30), rng.uniform(-0.9, 0.9, 30)):
        state = np.array([s, c])
        assert np.linalg.det(jacobian(state, values)) == pytest.approx(1.0, abs=1e-8)
        back = backward(billiard.trajectory(state, 1)[0], values)
        difference = (back[0] - s + perimeter / 2.0) % perimeter - perimeter / 2.0
        assert difference == pytest.approx(0.0, abs=1e-9)


def test_user_pieces_escape_wraps_only_on_a_single_closed_curve():
    closed = Billiard.from_pieces(stadium_as_pieces(1.0, 2.0), solver="exact")
    perimeter = closed.perimeter()
    assert closed.info["period"] == pytest.approx(perimeter)
    state = [3.0, 1.0]
    orbit = closed.trajectory(state, 500)
    index, time = closed.escape_analysis(
        state, 500, exits=[0.0, 0.0], hole_size=[0.3, np.inf]
    )
    inside = (orbit[:, 0] <= 0.15) | (orbit[:, 0] >= perimeter - 0.15)
    assert (index, time) == (0, np.flatnonzero(inside)[0] + 1)

    with_disk = Billiard.from_pieces(sinai_as_pieces(0.2, 0.5, 0.5), solver="exact")
    assert with_disk.info["period"] is None


def test_user_inside_function_is_used_by_molecular_dynamics(rng):
    fast = Billiard.from_pieces(
        stadium_table, inside=stadium_inside, solver="md", parameters=[1.0, 2.0]
    )
    derived = Billiard.from_pieces(stadium_table, solver="md", parameters=[1.0, 2.0])
    for state in states_for(rng, fast, "stadium", 10, 0.05):
        np.testing.assert_allclose(
            fast.trajectory(state, 1), derived.trajectory(state, 1), atol=1e-10
        )


def test_boundary_and_cartesian_of_user_billiards():
    billiard = Billiard.from_pieces(sinai_as_pieces(0.2, 0.3, 0.4), solver="exact")
    boundary = billiard.boundary(200)
    assert np.isnan(boundary[:, 0]).sum() == 1
    polar = Billiard.from_polar(two_mode_radius, two_mode_dradius, solver="md")
    points = polar.boundary(100)
    assert points.shape == (100, 2)
    np.testing.assert_allclose(points[0], points[-1], atol=1e-12)
    assert polar.to_cartesian([0.0, 1.0]) == pytest.approx((1.1, 0.0))


@pytest.mark.parametrize(
    "pieces, error, match",
    [
        ([], ValueError, "empty"),
        ([("curve", 0.0, 0.0, 1.0, 1.0)], ValueError, "line"),
        ([("line", 0.0, 0.0, 1.0)], ValueError, "line 0"),
        ([("arc", 0.0, 0.0, 1.0, 0.0)], ValueError, "arc 0"),
        ([("line", "a", 0.0, 1.0, 1.0)], TypeError, "numeric"),
        ([("arc", 0.0, 0.0, -1.0, 0.0, TWO_PI)], ValueError, "radius"),
        ([("arc", 0.0, 0.0, 1.0, 0.0, 7.0)], ValueError, "2 pi"),
        (
            [("line", 0.0, 0.0, 0.0, 0.0), ("arc", 0.0, 0.0, 1.0, 0.0, TWO_PI)],
            ValueError,
            "length",
        ),
        (
            [
                ("line", 0.0, 0.0, 1.0, 0.0),
                ("line", 1.0, 0.0, 1.0, 1.0),
                ("line", 1.0, 1.0, 0.0, 1.0),
            ],
            ValueError,
            "connected",
        ),
        (
            [
                ("line", 0.0, 0.0, 0.0, 1.0),
                ("line", 0.0, 1.0, 1.0, 1.0),
                ("line", 1.0, 1.0, 1.0, 0.0),
                ("line", 1.0, 0.0, 0.0, 0.0),
            ],
            ValueError,
            "wrong way",
        ),
        (
            [
                ("line", 0.0, 0.0, 1.0, 0.0),
                ("line", 1.0, 0.0, 1.0, 1.0),
                ("line", 1.0, 1.0, 0.0, 1.0),
                ("line", 0.0, 1.0, 0.0, 0.0),
                ("arc", 0.5, 0.5, 0.2, 0.0, TWO_PI),
            ],
            ValueError,
            "wrong way",
        ),
    ],
)
def test_invalid_pieces(pieces, error, match):
    with pytest.raises(error, match=match):
        Billiard.from_pieces(pieces, solver="exact")


@njit
def clockwise_stadium(parameters):
    table = new_pieces(2)
    set_arc(table, 0, 0.0, 0.0, 1.0, 0.0, -np.pi)
    set_arc(table, 1, 0.0, 0.0, 1.0, np.pi, -np.pi)
    return table


def test_invalid_pieces_from_a_function_raise_with_the_parameters():
    with pytest.raises(ValueError, match="wrong way"):
        Billiard.from_pieces(clockwise_stadium, parameters=[1.0], solver="exact")


SQUARE = [
    ("line", 0.0, 0.0, 1.0, 0.0),
    ("line", 1.0, 0.0, 1.0, 1.0),
    ("line", 1.0, 1.0, 0.0, 1.0),
    ("line", 0.0, 1.0, 0.0, 0.0),
]


def disk(x, y, r):
    return ("arc", x, y, r, 0.0, -TWO_PI)


def two_rooms(r, h=0.4, L=1.0, a=0.1):
    # Two squares joined by an aperture, with half circles at the ends of the wall.
    w = 2.0 * r if r > 0.0 else a
    y = (L - h - 2.0 * r) / 2.0
    if r > 0.0:
        lower = ("arc", L + r, y, r, np.pi, -np.pi)
        upper = ("arc", L + r, L - y, r, 0.0, -np.pi)
    else:
        lower = ("line", L, y, L + w, y)
        upper = ("line", L + w, L - y, L, L - y)
    return [
        ("line", 0.0, 0.0, L, 0.0),
        ("line", L, 0.0, L, y),
        lower,
        ("line", L + w, y, L + w, 0.0),
        ("line", L + w, 0.0, 2.0 * L + w, 0.0),
        ("line", 2.0 * L + w, 0.0, 2.0 * L + w, L),
        ("line", 2.0 * L + w, L, L + w, L),
        ("line", L + w, L, L + w, L - y),
        upper,
        ("line", L, L - y, L, L),
        ("line", L, L, 0.0, L),
        ("line", 0.0, L, 0.0, 0.0),
        disk(L + w + 0.4, L / 2.0, 0.1),
    ]


def rotated(pieces, angle):
    # The same boundary at another angle, which changes the round-off at its joints.
    cos, sin = np.cos(angle), np.sin(angle)

    def turn(x, y):
        return cos * x - sin * y, sin * x + cos * y

    result = []
    for piece in pieces:
        if piece[0] == "line":
            result.append(("line", *turn(*piece[1:3]), *turn(*piece[3:5])))
        else:
            result.append(("arc", *turn(*piece[1:3]), piece[3], piece[4] + angle, piece[5]))
    return result


@pytest.mark.parametrize(
    "pieces, match",
    [
        # A bow tie
        (
            [
                ("line", 0.0, 0.0, 1.0, 1.0),
                ("line", 1.0, 1.0, 1.0, 0.0),
                ("line", 1.0, 0.0, 0.0, 1.0),
                ("line", 0.0, 1.0, 0.0, 0.0),
            ],
            r"pieces 0 and 2 cross or touch at \(0\.5, 0\.5\)",
        ),
        # A disk through a wall
        (SQUARE + [disk(0.5, 1.0, 0.3)], "pieces 2 and 4 cross"),
        # An arc of the boundary through a wall
        (
            [
                ("line", 0.0, 0.0, 2.0, 0.0),
                ("line", 2.0, 0.0, 2.0, 1.0),
                ("arc", 1.5, 1.0, 0.5, 0.0, -np.pi),
                ("line", 1.0, 1.0, 1.0, -0.5),
                ("line", 1.0, -0.5, 0.5, -0.5),
                ("line", 0.5, -0.5, 0.5, 1.0),
                ("line", 0.5, 1.0, 0.0, 1.0),
                ("line", 0.0, 1.0, 0.0, 0.0),
            ],
            "cross",
        ),
        # Two overlapping disks
        (SQUARE + [disk(0.4, 0.5, 0.2), disk(0.6, 0.5, 0.2)], "pieces 4 and 5 cross"),
        # Tangent parts
        (SQUARE + [disk(0.5, 0.2, 0.2)], r"pieces 0 and 4 cross or touch at \(0\.5, 0\)"),
        (SQUARE + [disk(0.3, 0.5, 0.2), disk(0.7, 0.5, 0.2)], "pieces 4 and 5 cross"),
        # A piece that doubles back over its neighbor
        (
            [
                ("line", 0.0, 0.0, 1.0, 0.0),
                ("line", 1.0, 0.0, 0.5, 0.0),
                ("line", 0.5, 0.0, 0.5, 1.0),
                ("line", 0.5, 1.0, 0.0, 0.0),
            ],
            "pieces 0 and 1 cross",
        ),
        # A half circle wider than the wall it sits on, which also doubles back
        (
            [
                ("line", 0.0, 0.0, 1.0, 0.0),
                ("line", 1.0, 0.0, 0.8, 0.0),
                ("arc", 1.0, 0.0, 0.2, np.pi, -np.pi),
                ("line", 1.2, 0.0, 2.0, 0.0),
                ("line", 2.0, 0.0, 2.0, 1.0),
                ("line", 2.0, 1.0, 0.0, 1.0),
                ("line", 0.0, 1.0, 0.0, 0.0),
            ],
            "pieces 0 and 1 cross",
        ),
        # A corner that touches the opposite wall
        (
            [
                ("line", 0.0, 0.0, 1.0, 0.0),
                ("line", 1.0, 0.0, 1.0, 1.0),
                ("line", 1.0, 1.0, 0.5, 0.0),
                ("line", 0.5, 0.0, 0.0, 1.0),
                ("line", 0.0, 1.0, 0.0, 0.0),
            ],
            "cross",
        ),
        # The walls of the two rooms overlap across the aperture
        (two_rooms(0.0, h=-0.1), "cross"),
        (two_rooms(0.1, h=-0.1), "cross"),
        # Holes in the wrong place
        (SQUARE + [disk(3.0, 0.5, 0.2)], "outside the billiard"),
        (SQUARE + [disk(0.5, 0.5, 0.3), disk(0.5, 0.5, 0.1)], "inside the hole"),
    ],
)
def test_invalid_boundaries_are_rejected_and_named(pieces, match):
    with pytest.raises(ValueError, match=match):
        Billiard.from_pieces(pieces, solver="exact")
    assert plt.fignum_exists("Invalid billiard boundary")


@pytest.mark.parametrize("angle", [0.0, 0.3, 1.0, 2.5])
def test_valid_boundaries_are_accepted(angle):
    half_disk = [("arc", 0.0, 0.0, 1.0, 0.0, np.pi), ("line", -1.0, 0.0, 1.0, 0.0)]
    circle_in_halves = [
        ("arc", 0.0, 0.0, 1.0, 0.0, np.pi),
        ("arc", 0.0, 0.0, 1.0, np.pi, np.pi),
    ]
    collinear = [
        ("line", 0.0, 0.0, 0.5, 0.0),
        ("line", 0.5, 0.0, 1.0, 0.0),
        ("line", 1.0, 0.0, 0.0, 1.0),
        ("line", 0.0, 1.0, 0.0, 0.0),
    ]
    # Arcs of two radii with one tangent at their joint, and a line tangent to the arc
    tangent_arcs = [
        ("arc", 0.0, 0.0, 1.0, -np.pi / 2.0, np.pi),
        ("arc", 0.0, 0.5, 0.5, np.pi / 2.0, np.pi),
        ("arc", 0.0, -0.5, 0.5, np.pi / 2.0, -np.pi),
    ]
    boundaries = [half_disk, circle_in_halves, collinear, tangent_arcs]
    boundaries += [SQUARE + [disk(0.3, 0.3, 0.2), disk(0.7, 0.7, 0.2)]]
    boundaries += [two_rooms(r) for r in (0.0, 1e-3, 0.05, 0.2, 0.29)]
    boundaries += [
        AS_PIECES[name](*parameters)
        for name, parameters in PIECEWISE_PARAMETERS
    ]
    for pieces in boundaries:
        table = _fixed_pieces_table(rotated(pieces, angle))
        assert _pieces_problem(table) is None


def test_random_builtin_shapes_are_accepted(rng):
    for _ in range(300):
        angle = rng.uniform(0.0, TWO_PI)
        R = rng.uniform(0.2, 2.0)
        w = rng.uniform(0.01, 1.9 * R)
        r = rng.uniform(0.01, 0.45)
        center = rng.uniform(r + 0.01, 1.0 - r - 0.01, 2)
        for pieces in (
            stadium_as_pieces(rng.uniform(0.01, 3.0), rng.uniform(0.01, 3.0)),
            mushroom_as_pieces(R, w, rng.uniform(0.01, 3.0)),
            sinai_as_pieces(r, *center),
        ):
            assert _pieces_problem(_fixed_pieces_table(rotated(pieces, angle))) is None


def segments_cross(p0, p1, q0, q1):
    def side(a, b, c):
        return np.sign((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

    return side(p0, p1, q0) != side(p0, p1, q1) and side(q0, q1, p0) != side(q0, q1, p1)


def test_crossings_of_random_polygons_match_an_independent_test(rng):
    crossing = 0
    for _ in range(400):
        corners = rng.uniform(0.0, 1.0, (7, 2))
        pieces = [
            ("line", *corners[j], *corners[(j + 1) % 7]) for j in range(7)
        ]
        expected = any(
            segments_cross(corners[i], corners[(i + 1) % 7], corners[j], corners[(j + 1) % 7])
            for i in range(7)
            for j in range(i + 2, 7)
            if (j + 1) % 7 != i
        )
        problem = _pieces_problem(_fixed_pieces_table(pieces))
        found = problem is not None and "cross" in problem[0]
        assert found == expected
        crossing += expected
    assert 0 < crossing < 400


def test_arc_crossings_match_a_dense_scan(rng):
    # Two random arcs and whether they meet, against points sampled along both.
    checked = 0
    for _ in range(300):
        pieces = [
            (
                "arc",
                *rng.uniform(0.0, 1.0, 2),
                rng.uniform(0.1, 0.6),
                rng.uniform(0.0, TWO_PI),
                rng.uniform(0.2, 6.0) * rng.choice([-1.0, 1.0]),
            )
            for _ in range(2)
        ]
        table = _fixed_pieces_table(pieces)
        curves = [
            np.array(
                [piece_point(table, row, u) for u in np.linspace(0.0, table[row, 2], 1500)]
            )
            for row in range(2)
        ]
        separation = np.min(
            np.hypot(*(curves[0][:, None, :] - curves[1][None, :, :]).transpose(2, 0, 1))
        )
        found = _pieces_meeting_point(table, 0, 1, []) is not None
        if separation > 5e-3:
            assert not found
            checked += 1
        elif separation < 1e-4:
            assert found
            checked += 1
    assert checked > 200


def test_two_rooms_stay_valid_up_to_the_largest_radius():
    assert Billiard.from_pieces(two_rooms(0.2999), solver="exact").perimeter() > 0.0
    # At the limit the vertical walls vanish, as a piece without length.
    with pytest.raises(ValueError, match="positive length"):
        Billiard.from_pieces(two_rooms(0.3), solver="exact")


@njit
def square_with_disk(parameters):
    table = new_pieces(5)
    set_line(table, 0, 0.0, 0.0, 1.0, 0.0)
    set_line(table, 1, 1.0, 0.0, 1.0, 1.0)
    set_line(table, 2, 1.0, 1.0, 0.0, 1.0)
    set_line(table, 3, 0.0, 1.0, 0.0, 0.0)
    set_arc(table, 4, 0.5, 0.5, parameters[0], 0.0, -2.0 * np.pi)
    return table


def test_parameters_that_make_the_boundary_cross_raise():
    billiard = Billiard.from_pieces(square_with_disk, parameters=[0.2], solver="exact")
    plt.close("all")
    with pytest.raises(ValueError, match="pieces 0 and 4 cross"):
        billiard.set_parameters([0.6])
    assert plt.fignum_exists("Invalid billiard boundary")


def test_rejected_boundary_is_drawn_with_the_problem_marked():
    with pytest.raises(ValueError, match="pieces 2 and 4 cross") as error:
        Billiard.from_pieces(SQUARE + [disk(0.5, 1.0, 0.3)], solver="exact")
    assert "plt.show" in str(error.value)

    axes = plt.figure("Invalid billiard boundary").axes[0]
    assert "pieces 2 and 4 cross" in axes.get_title()
    red = [line for line in axes.get_lines() if line.get_color() == "tab:red"]
    # The two pieces, and the point where they meet
    assert len(red) == 3
    assert {text.get_text() for text in axes.texts if text.get_text()} == set("01234")


def test_repeated_rejections_reuse_one_figure():
    plt.close("all")
    for radius in (0.55, 0.6, 0.65):
        with pytest.raises(ValueError, match="cross"):
            Billiard.from_pieces(SQUARE + [disk(0.5, 0.5, radius)], solver="exact")
    assert len(plt.get_fignums()) == 1


def test_failure_to_draw_keeps_the_error(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("no display")

    monkeypatch.setattr(plt, "figure", fail)
    with pytest.raises(ValueError, match="pieces 2 and 4 cross") as error:
        Billiard.from_pieces(SQUARE + [disk(0.5, 1.0, 0.3)], solver="exact")
    assert "plt.show" not in str(error.value)


def test_plot_pieces():
    axes = Billiard.plot_pieces(SQUARE + [disk(0.5, 0.5, 0.2)])
    assert len(axes.get_lines()) == 5
    assert {text.get_text() for text in axes.texts if text.get_text()} == set("01234")

    # A boundary that is not valid can be drawn too.
    _, given = plt.subplots()
    assert Billiard.plot_pieces(SQUARE + [disk(0.5, 1.0, 0.3)], ax=given) is given

    assert len(Billiard.plot_pieces(square_with_disk, [0.2]).get_lines()) == 5
    with pytest.raises(ValueError, match="requires parameters"):
        Billiard.plot_pieces(square_with_disk)


@njit
def dented_radius(theta, parameters):
    x = (theta - 1.0) / 0.001
    return 1.0 - 1e-3 * np.exp(-x * x)


@njit
def dented_dradius(theta, parameters):
    x = (theta - 1.0) / 0.001
    return 1e-3 * np.exp(-x * x) * 2.0 * x / 0.001


@njit
def dented_ddradius(theta, parameters):
    x = (theta - 1.0) / 0.001
    return 1e-3 * np.exp(-x * x) * (2.0 - 4.0 * x * x) / 0.001**2


@njit
def dented_max_radius(parameters):
    return 1.0


def test_circle_rejects_a_narrow_dent_and_draws_it():
    plt.close("all")
    with pytest.raises(ValueError, match="convex") as error:
        Billiard.from_polar(
            dented_radius,
            dented_dradius,
            ddradius=dented_ddradius,
            max_radius=dented_max_radius,
            solver="circle",
        )
    assert "solver='md'" in str(error.value)
    assert plt.fignum_exists("Invalid billiard boundary")

    # Molecular dynamics has no such limit.
    Billiard.from_polar(
        dented_radius,
        dented_dradius,
        ddradius=dented_ddradius,
        max_radius=dented_max_radius,
        solver="md",
    )


@pytest.mark.parametrize(
    "build",
    [
        lambda **kwargs: Billiard("stadium", **kwargs),
        lambda **kwargs: Billiard.from_polar(two_mode_radius, two_mode_dradius, **kwargs),
        lambda **kwargs: Billiard.from_pieces(stadium_as_pieces(1.0, 1.0), **kwargs),
    ],
)
def test_the_solver_must_be_chosen(build):
    with pytest.raises(TypeError, match="solver"):
        build()
    with pytest.raises(ValueError, match="not available"):
        build(solver="auto")


def test_circle_without_max_radius_says_what_it_needs():
    with pytest.raises(ValueError, match="requires max_radius"):
        Billiard.from_polar(two_mode_radius, two_mode_dradius, solver="circle")


@pytest.mark.parametrize(
    "billiard",
    [
        lambda: Billiard("oval", parameters=[0.08, 3], solver="circle"),
        lambda: Billiard("oval", parameters=[0.08, 3], solver="md", md_steps=700),
        lambda: Billiard.from_pieces(stadium_as_pieces(1.0, 2.0), solver="md"),
    ],
)
def test_pickling_keeps_the_chosen_solver(billiard):
    original = billiard()
    restored = pickle.loads(pickle.dumps(original))
    assert restored.info["solver"] == original.info["solver"]
    assert restored.info["md_steps"] == original.info["md_steps"]


def test_parallel_runs_use_the_chosen_solver():
    state = [2.0, 1.0]
    kwargs = dict(max_time=100, exits=[1.0, 0.0], hole_size=[0.2, np.inf])
    for solver in ("circle", "md"):
        billiard = Billiard("oval", parameters=[0.08, 3], solver=solver)
        in_parallel = Parallel(n_jobs=2)(
            delayed(billiard.escape_analysis)(state, **kwargs) for _ in range(2)
        )
        assert in_parallel[0] == billiard.escape_analysis(state, **kwargs)


def test_rejected_polar_boundary_is_drawn():
    plt.close("all")
    with pytest.raises(ValueError, match="positive") as error:
        Billiard.from_polar(negative_radius, zero_derivative, solver="md")
    assert "plt.show" in str(error.value)
    assert plt.fignum_exists("Invalid billiard boundary")


def test_unavailable_solver_on_user_pieces():
    with pytest.raises(ValueError, match="not available"):
        Billiard.from_pieces(stadium_as_pieces(1.0, 1.0), solver="polygon")


def test_the_same_definition_is_compiled_once():
    first = Billiard.from_pieces(stadium_as_pieces(1.0, 1.0), solver="exact")
    second = Billiard.from_pieces(stadium_as_pieces(1.0, 1.0), solver="exact")
    assert first._DiscreteDynamicalSystem__mapping is second._DiscreteDynamicalSystem__mapping
    first = Billiard.from_polar(two_mode_radius, two_mode_dradius, solver="md")
    second = Billiard.from_polar(two_mode_radius, two_mode_dradius, solver="md")
    assert first._DiscreteDynamicalSystem__mapping is second._DiscreteDynamicalSystem__mapping


def test_parameters_are_required_by_a_user_billiard_that_has_them():
    billiard = Billiard.from_pieces(stadium_table, number_of_parameters=2, solver="exact")
    with pytest.raises(ValueError):
        billiard.perimeter()
    assert billiard.info["period"] is None
    billiard.set_parameters([1.0, 1.0])
    assert billiard.info["period"] == pytest.approx(TWO_PI + 2.0)


def test_parallel_workers_on_user_billiards():
    for billiard in (
        Billiard.from_polar(two_mode_radius, two_mode_dradius, solver="md"),
        Billiard.from_pieces(stadium_as_pieces(1.0, 2.0), solver="exact"),
        Billiard.from_pieces(stadium_table, parameters=[1.0, 2.0], solver="exact"),
    ):
        states = [[0.4, a] for a in (0.5, 1.0, 1.5)]
        kwargs = dict(max_time=100, exits=[1.0, 0.0], hole_size=[0.2, np.inf])
        in_parallel = Parallel(n_jobs=2)(
            delayed(billiard.escape_analysis)(state, **kwargs) for state in states
        )
        assert in_parallel == [billiard.escape_analysis(s, **kwargs) for s in states]


def test_coarse_arc_length_warns_and_more_panels_fix_it():
    with pytest.warns(UserWarning, match="num_panels"):
        coarse = user_oval([0.9, 3], solver="md", coordinates="birkhoff")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fine = user_oval([0.9, 3], solver="md", coordinates="birkhoff", num_panels=512)
        # The arc length plays no part in these coordinates.
        user_oval([0.9, 3], solver="md")
    builtin = Billiard("oval", parameters=[0.9, 3], solver="polygon").perimeter()
    assert abs(fine.perimeter() - builtin) < 1e-11
    assert abs(coarse.perimeter() - builtin) > 1e-8


@pytest.mark.parametrize(
    "build",
    [
        lambda: Billiard("oval", parameters=[0.08, 3], solver="circle"),
        lambda: Billiard("oval", parameters=[0.3, 3], coordinates="birkhoff", solver="polygon"),
        lambda: Billiard("oval", parameters=[0.3, 3], solver="md", md_steps=321),
        lambda: Billiard("sinai", parameters=[0.2, 0.5, 0.5], solver="exact"),
        lambda: Billiard("stadium", solver="exact"),
        lambda: Billiard.from_pieces(stadium_as_pieces(1.0, 2.0), solver="exact"),
        lambda: Billiard.from_pieces(stadium_table, parameters=[1.0, 2.0], solver="exact"),
        lambda: Billiard.from_polar(two_mode_radius, two_mode_dradius, solver="md"),
        lambda: user_oval([0.08, 3], solver="circle"),
    ],
)
def test_pickling_keeps_the_billiard_and_is_light(build):
    billiard = build()
    data = cloudpickle.dumps(billiard)
    copy = pickle.loads(data)
    assert copy.info == billiard.info
    assert copy._DiscreteDynamicalSystem__mapping is (
        billiard._DiscreteDynamicalSystem__mapping
    )
    # One task for each trajectory must not pay for the compiled maps.
    assert len(data) < 30_000


def test_pickling_a_builtin_billiard_takes_no_compiled_code():
    assert len(pickle.dumps(Billiard("oval", parameters=[0.08, 3], solver="circle"))) < 1000
