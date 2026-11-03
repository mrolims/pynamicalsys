# billiards.py

# Copyright (C) 2025-2026 Matheus Rolim Sales
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

from typing import Any, Callable, Dict, Tuple

import numpy as np
from numba import njit
from numpy.typing import NDArray

# Smallest length resolved when locating a collision.
COLLISION_TOLERANCE = 1e-14

# How far past the end of a line or arc an intersection still counts, in its length
# or angle, so that round-off cannot let a trajectory through the joint of two pieces.
PIECE_END_TOLERANCE = 1e-12

# Limit of the tangent method, and the steps across the billiard of the molecular
# dynamics used instead when the method does not converge.
MAX_TANGENT_ITERATIONS = 200
FALLBACK_MD_STEPS = 20_000

# Gauss-Legendre rule for the arc length of smooth boundaries.
_QUADRATURE_NODES, _QUADRATURE_WEIGHTS = np.polynomial.legendre.leggauss(12)


# ! ---------------------- !
# ! --- Collision core --- !
# ! ---------------------- !


@njit
def md_collision(
    x0: float,
    y0: float,
    vx: float,
    vy: float,
    t0: float,
    inside: Callable,
    parameters: NDArray[np.float64],
    step_length: float,
) -> Tuple[float, float, float]:
    """
    Find the first collision of a free particle by molecular dynamics.

    The particle starts on the boundary and advances in steps of `step_length`
    until it is outside the billiard. The collision is then located by
    bisection on the last step. A collision and a return to the interior within
    one step are not detected, so `step_length` must resolve the smallest
    feature of the boundary along the trajectory.

    Parameters
    ----------
    x0, y0 : float
        Initial position, on the boundary.
    vx, vy : float
        Velocity of the particle.
    t0 : float
        Time at the initial position.
    inside : Callable
        Function `inside(x, y, t, parameters)` that returns whether a point is
        inside the billiard at time `t`.
    parameters : NDArray[np.float64]
        Parameters of the billiard.
    step_length : float
        Distance travelled in one step.

    Returns
    -------
    Tuple[float, float, float]
        Time and position `(t1, x1, y1)` of the collision. If no interior point
        is resolved next to the start, the initial time and position are
        returned.
    """
    speed = np.sqrt(vx * vx + vy * vy)
    dt = step_length / speed
    resolution = COLLISION_TOLERANCE / speed

    t_in = dt
    t_out = dt
    if inside(x0 + vx * dt, y0 + vy * dt, t0 + dt, parameters):
        # Advance until the particle is outside.
        t_out = t_in + dt
        while inside(x0 + vx * t_out, y0 + vy * t_out, t0 + t_out, parameters):
            t_in = t_out
            t_out = t_in + dt
    else:
        # A near-grazing start leaves within the first step: halve the step
        # until there is an interior point to bisect against.
        t_in = dt / 2.0
        while not inside(x0 + vx * t_in, y0 + vy * t_in, t0 + t_in, parameters):
            t_out = t_in
            t_in = t_in / 2.0
            if t_in < resolution:
                return t0, x0, y0

    while t_out - t_in > resolution:
        t_mid = 0.5 * (t_in + t_out)
        if t_mid <= t_in or t_mid >= t_out:
            break
        if inside(x0 + vx * t_mid, y0 + vy * t_mid, t0 + t_mid, parameters):
            t_in = t_mid
        else:
            t_out = t_mid

    t_hit = 0.5 * (t_in + t_out)

    return t0 + t_hit, x0 + vx * t_hit, y0 + vy * t_hit


@njit
def reflect(
    vx: float,
    vy: float,
    tx: float,
    ty: float,
    wall_vx: float,
    wall_vy: float,
) -> Tuple[float, float]:
    """
    Reflect a velocity at a wall that may be moving.

    In the frame of the wall, the tangential component of the velocity is kept
    and the normal component changes sign.

    Parameters
    ----------
    vx, vy : float
        Velocity of the particle before the collision.
    tx, ty : float
        Unit tangent vector of the boundary at the collision.
    wall_vx, wall_vy : float
        Velocity of the wall at the collision.

    Returns
    -------
    Tuple[float, float]
        Velocity of the particle after the collision.
    """
    ux = vx - wall_vx
    uy = vy - wall_vy
    tangential = ux * tx + uy * ty

    return 2.0 * tangential * tx - ux + wall_vx, 2.0 * tangential * ty - uy + wall_vy


# ! ----------------------- !
# ! --- Static map core --- !
# ! ----------------------- !


@njit
def _static_collision(
    q: float,
    cos_alpha: float,
    sin_alpha: float,
    parameters: NDArray[np.float64],
    collide: Callable,
    boundary_point: Callable,
    tangent: Callable,
    project: Callable,
) -> Tuple[float, float, float, float]:
    """Return `(q1, cos_alpha1, sin_alpha1, chord)` for a collision with a static wall."""
    x0, y0 = boundary_point(q, 0.0, parameters)
    tx, ty = tangent(q, 0.0, parameters)
    vx = cos_alpha * tx - sin_alpha * ty
    vy = sin_alpha * tx + cos_alpha * ty

    chord, x1, y1 = collide(q, x0, y0, vx, vy, 0.0, parameters)

    q1 = project(x1, y1, 0.0, parameters)
    tx, ty = tangent(q1, 0.0, parameters)
    vx, vy = reflect(vx, vy, tx, ty, 0.0, 0.0)

    return q1, tx * vx + ty * vy, tx * vy - ty * vx, chord


def make_md_solver(inside: Callable, diameter: Callable, md_steps: int) -> Callable:
    """
    Build the molecular dynamics solver of a billiard.

    Every solver is a function `collide(q0, x0, y0, vx, vy, t0, parameters)` that takes
    the position `q0` and the point `(x0, y0)` of a particle leaving the boundary at
    time `t0` with velocity `(vx, vy)`, and returns the time and point `(t1, x1, y1)` of
    its next collision.

    Parameters
    ----------
    inside : Callable
        Function `inside(x, y, t, parameters)`.
    diameter : Callable
        Function `diameter(parameters)` that returns the size of the billiard.
    md_steps : int
        Number of molecular dynamics steps across the diameter of the billiard.

    Returns
    -------
    Callable
        The solver.
    """

    @njit
    def collide(
        q0: float,
        x0: float,
        y0: float,
        vx: float,
        vy: float,
        t0: float,
        parameters: NDArray[np.float64],
    ) -> Tuple[float, float, float]:
        return md_collision(
            x0, y0, vx, vy, t0, inside, parameters, diameter(parameters) / md_steps
        )

    return collide


def make_angle_maps(
    check: Callable,
    collide: Callable,
    boundary_point: Callable,
    tangent: Callable,
    curvature: Callable,
    arc_speed: Callable,
    project: Callable,
) -> Tuple[Callable, Callable, Callable]:
    """
    Build the map of a static billiard in the coordinates `(q, alpha)`.

    Parameters
    ----------
    check : Callable
        Function `check(parameters)` that raises if the parameters are invalid.
    collide : Callable
        Solver that finds the next collision, as in `make_md_solver`.
    boundary_point, tangent : Callable
        Functions of `(q, t, parameters)` that return the point and the unit
        tangent of the boundary, with the billiard on the left of the tangent.
    curvature, arc_speed : Callable
        Functions of `(q, t, parameters)` that return the signed curvature and
        `ds/dq`, the arc length per unit of `q`.
    project : Callable
        Function `project(x, y, t, parameters)` that returns `q`.

    Returns
    -------
    Tuple[Callable, Callable, Callable]
        Tuple `(mapping, jacobian, backwards_mapping)`.
    """

    @njit
    def mapping(
        u: NDArray[np.float64], parameters: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        check(parameters)
        q, alpha = u
        if not 0.0 < alpha < np.pi:
            raise ValueError("alpha must lie in (0, pi)")
        q1, cos_alpha1, sin_alpha1, _ = _static_collision(
            q,
            np.cos(alpha),
            np.sin(alpha),
            parameters,
            collide,
            boundary_point,
            tangent,
            project,
        )

        return np.array([q1, np.arctan2(sin_alpha1, cos_alpha1)])

    @njit
    def jacobian(
        u: NDArray[np.float64], parameters: NDArray[np.float64], *args: Any
    ) -> NDArray[np.float64]:
        check(parameters)
        q, alpha = u
        if not 0.0 < alpha < np.pi:
            raise ValueError("alpha must lie in (0, pi)")
        sin_alpha = np.sin(alpha)
        q1, _, sin_alpha1, chord = _static_collision(
            q,
            np.cos(alpha),
            sin_alpha,
            parameters,
            collide,
            boundary_point,
            tangent,
            project,
        )

        kappa = curvature(q, 0.0, parameters)
        kappa1 = curvature(q1, 0.0, parameters)
        speed = arc_speed(q, 0.0, parameters)
        speed1 = arc_speed(q1, 0.0, parameters)

        # Derivatives with respect to arc length, from
        # sin(alpha1) ds1 = (chord kappa - sin(alpha)) ds + chord dalpha
        # dalpha1 = kappa1 ds1 - kappa ds - dalpha
        ds1_ds = (chord * kappa - sin_alpha) / sin_alpha1
        ds1_dalpha = chord / sin_alpha1

        J = np.zeros((2, 2))
        J[0, 0] = ds1_ds * speed / speed1
        J[0, 1] = ds1_dalpha / speed1
        J[1, 0] = (kappa1 * ds1_ds - kappa) * speed
        J[1, 1] = kappa1 * ds1_dalpha - 1.0

        return J

    @njit
    def backwards_mapping(
        u: NDArray[np.float64], parameters: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        # Time reversal: reverse the velocity, advance, and reverse it again.
        reversed_state = mapping(np.array([u[0], np.pi - u[1]]), parameters)

        return np.array([reversed_state[0], np.pi - reversed_state[1]])

    return mapping, jacobian, backwards_mapping


def make_birkhoff_maps(
    check: Callable,
    collide: Callable,
    boundary_point: Callable,
    tangent: Callable,
    curvature: Callable,
    project: Callable,
    arc_length: Callable,
    arc_position: Callable,
) -> Tuple[Callable, Callable, Callable]:
    """
    Build the map of a static billiard in the Birkhoff coordinates `(s, p)`.

    Here `s` is the arc length along the boundary and `p = cos(alpha)`. These
    coordinates are canonical, so the map preserves area.

    Parameters
    ----------
    check, collide, boundary_point, tangent, curvature, project : Callable
        As in `make_angle_maps`.
    arc_length : Callable
        Function `arc_length(q, t, parameters)` that returns `s`.
    arc_position : Callable
        Function `arc_position(s, t, parameters)` that returns `q`.

    Returns
    -------
    Tuple[Callable, Callable, Callable]
        Tuple `(mapping, jacobian, backwards_mapping)`.
    """

    @njit
    def mapping(
        u: NDArray[np.float64], parameters: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        check(parameters)
        s, cos_alpha = u
        if not -1.0 < cos_alpha < 1.0:
            raise ValueError("cos(alpha) must lie in (-1, 1)")
        q1, cos_alpha1, _, _ = _static_collision(
            arc_position(s, 0.0, parameters),
            cos_alpha,
            np.sqrt(1.0 - cos_alpha * cos_alpha),
            parameters,
            collide,
            boundary_point,
            tangent,
            project,
        )

        return np.array([arc_length(q1, 0.0, parameters), cos_alpha1])

    @njit
    def jacobian(
        u: NDArray[np.float64], parameters: NDArray[np.float64], *args: Any
    ) -> NDArray[np.float64]:
        check(parameters)
        s, cos_alpha = u
        if not -1.0 < cos_alpha < 1.0:
            raise ValueError("cos(alpha) must lie in (-1, 1)")
        q = arc_position(s, 0.0, parameters)
        sin_alpha = np.sqrt(1.0 - cos_alpha * cos_alpha)
        q1, _, sin_alpha1, chord = _static_collision(
            q,
            cos_alpha,
            sin_alpha,
            parameters,
            collide,
            boundary_point,
            tangent,
            project,
        )

        kappa = curvature(q, 0.0, parameters)
        kappa1 = curvature(q1, 0.0, parameters)
        ds1_ds = (chord * kappa - sin_alpha) / sin_alpha1

        J = np.zeros((2, 2))
        J[0, 0] = ds1_ds
        J[0, 1] = -chord / (sin_alpha * sin_alpha1)
        J[1, 0] = sin_alpha1 * (kappa - kappa1 * ds1_ds)
        J[1, 1] = (kappa1 * chord - sin_alpha1) / sin_alpha

        return J

    @njit
    def backwards_mapping(
        u: NDArray[np.float64], parameters: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        # Time reversal: reverse the velocity, advance, and reverse it again.
        reversed_state = mapping(np.array([u[0], -u[1]]), parameters)

        return np.array([reversed_state[0], -reversed_state[1]])

    return mapping, jacobian, backwards_mapping


# ! --------------------- !
# ! --- Oval billiard --- !
# ! --------------------- !


@njit
def oval_check(parameters: NDArray[np.float64]) -> None:
    eps, p = parameters
    if not 0.0 <= eps <= 1.0:
        raise ValueError("eps must lie in [0, 1]")
    if p < 1.0 or p != np.floor(p):
        raise ValueError("p must be a positive integer")


@njit
def oval_diameter(parameters: NDArray[np.float64]) -> float:
    return 2.0 * (1.0 + parameters[0])


@njit
def oval_inside(
    x: float, y: float, t: float, parameters: NDArray[np.float64]
) -> bool:
    eps, p = parameters
    radius = 1.0 + eps * np.cos(p * np.arctan2(y, x))

    return x * x + y * y < radius * radius


@njit
def oval_boundary_point(
    theta: float, t: float, parameters: NDArray[np.float64]
) -> Tuple[float, float]:
    eps, p = parameters
    radius = 1.0 + eps * np.cos(p * theta)

    return radius * np.cos(theta), radius * np.sin(theta)


@njit
def oval_tangent(
    theta: float, t: float, parameters: NDArray[np.float64]
) -> Tuple[float, float]:
    eps, p = parameters
    radius = 1.0 + eps * np.cos(p * theta)
    dradius = -eps * p * np.sin(p * theta)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    norm = np.sqrt(radius * radius + dradius * dradius)

    return (
        (dradius * cos_theta - radius * sin_theta) / norm,
        (dradius * sin_theta + radius * cos_theta) / norm,
    )


@njit
def oval_curvature(
    theta: float, t: float, parameters: NDArray[np.float64]
) -> float:
    eps, p = parameters
    radius = 1.0 + eps * np.cos(p * theta)
    dradius = -eps * p * np.sin(p * theta)
    ddradius = -eps * p * p * np.cos(p * theta)
    norm2 = radius * radius + dradius * dradius

    return (radius * radius + 2.0 * dradius * dradius - radius * ddradius) / norm2**1.5


@njit
def oval_arc_speed(
    theta: float, t: float, parameters: NDArray[np.float64]
) -> float:
    eps, p = parameters
    radius = 1.0 + eps * np.cos(p * theta)
    dradius = -eps * p * np.sin(p * theta)

    return np.sqrt(radius * radius + dradius * dradius)


@njit
def oval_project(
    x: float, y: float, t: float, parameters: NDArray[np.float64]
) -> float:
    return np.arctan2(y, x) % (2.0 * np.pi)


@njit
def _oval_panel_edges(parameters: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Return the quadrature panels of `[0, pi / p]`, graded towards both ends.

    The boundary repeats every `2 pi / p` and is symmetric about `pi / p`, so this
    wedge determines the arc length. The integrand `ds/dtheta` has complex
    singularities whose real parts are `0` and `pi / p` only. With
    `D = sqrt(1 + eps^2 (p^2 - 1))`, the nearest one is at a distance
    `log((D + 1) / (eps (p + 1))) / p` from `pi / p`, which tends to zero as `eps`
    tends to one, so the smallest panels are made no wider than this distance. At
    `eps = 1`, the integrand is smooth up to `pi / p` from within the wedge, and the
    nearest one is at `log((D + 1) / (eps (p - 1))) / p` from `0`.
    """
    eps, p = parameters
    wedge = np.pi / p
    halvings = 1
    if eps > 0.0 and (eps < 1.0 or p > 1.0):
        root = np.sqrt(1.0 + eps * eps * (p * p - 1.0))
        if eps < 1.0:
            distance = np.log((root + 1.0) / (eps * (p + 1.0))) / p
        else:
            distance = np.log((root + 1.0) / (eps * (p - 1.0))) / p
        halvings = max(1, int(np.ceil(np.log2(0.5 * wedge / distance))) + 1)

    edges = np.empty(2 * halvings + 1)
    for j in range(halvings):
        width = 0.5 * wedge * 0.5**j
        edges[halvings - j] = width
        edges[halvings + j] = wedge - width
    edges[0] = 0.0
    edges[-1] = wedge

    return edges


@njit
def _oval_panel_integral(
    lower: float, upper: float, parameters: NDArray[np.float64]
) -> float:
    half_width = 0.5 * (upper - lower)
    center = lower + half_width
    total = 0.0
    for node, weight in zip(_QUADRATURE_NODES, _QUADRATURE_WEIGHTS):
        total += weight * oval_arc_speed(center + half_width * node, 0.0, parameters)

    return half_width * total


@njit
def oval_perimeter(parameters: NDArray[np.float64]) -> float:
    edges = _oval_panel_edges(parameters)
    half_wedge = 0.0
    for j in range(edges.size - 1):
        half_wedge += _oval_panel_integral(edges[j], edges[j + 1], parameters)

    return 2.0 * parameters[1] * half_wedge


@njit
def oval_arc_length(
    theta: float, t: float, parameters: NDArray[np.float64]
) -> float:
    p = parameters[1]
    wedge = np.pi / p
    theta = theta % (2.0 * np.pi)
    wedges = np.floor(theta / (2.0 * wedge))
    x = theta - 2.0 * wedge * wedges
    mirrored = x > wedge
    if mirrored:
        x = 2.0 * wedge - x

    edges = _oval_panel_edges(parameters)
    half_wedge = 0.0
    partial = 0.0
    for j in range(edges.size - 1):
        lower = edges[j]
        upper = edges[j + 1]
        half_wedge += _oval_panel_integral(lower, upper, parameters)
        if lower < x < upper:
            partial = half_wedge - _oval_panel_integral(x, upper, parameters)
        elif x >= upper:
            partial = half_wedge

    if mirrored:
        return 2.0 * half_wedge * (wedges + 1.0) - partial

    return 2.0 * half_wedge * wedges + partial


@njit
def oval_arc_position(
    s: float, t: float, parameters: NDArray[np.float64]
) -> float:
    p = parameters[1]
    wedge = np.pi / p
    edges = _oval_panel_edges(parameters)
    panels = np.empty(edges.size - 1)
    for j in range(panels.size):
        panels[j] = _oval_panel_integral(edges[j], edges[j + 1], parameters)
    half_wedge = np.sum(panels)

    s = s % (2.0 * p * half_wedge)
    wedges = np.floor(s / (2.0 * half_wedge))
    remainder = s - 2.0 * half_wedge * wedges
    mirrored = remainder > half_wedge
    if mirrored:
        remainder = 2.0 * half_wedge - remainder

    j = 0
    while j < panels.size - 1 and remainder > panels[j]:
        remainder -= panels[j]
        j += 1

    # Newton's method inside the panel, kept within it.
    lower = edges[j]
    upper = edges[j + 1]
    x = lower + (upper - lower) * min(remainder / panels[j], 1.0)
    for _ in range(50):
        error = _oval_panel_integral(lower, x, parameters) - remainder
        speed = oval_arc_speed(x, 0.0, parameters)
        if speed == 0.0:  # At the origin, with eps = 1.
            break
        x_new = x - error / speed
        x_new = min(max(x_new, lower), upper)
        converged = abs(x_new - x) < 1e-15
        x = x_new
        if converged:
            break

    if mirrored:
        x = 2.0 * wedge - x

    return (2.0 * wedge * wedges + x) % (2.0 * np.pi)


# ! ------------------------ !
# ! --- Tangent iteration --- !
# ! ------------------------ !


@njit
def _tangent_iteration(
    x0: float,
    y0: float,
    dx: float,
    dy: float,
    seed: float,
    largest: float,
    parameters: NDArray[np.float64],
    boundary_point: Callable,
    tangent: Callable,
) -> Tuple[float, bool]:
    """
    Find where a trajectory meets the boundary by the tangent method.

    From a point on the trajectory, take the boundary point at the same polar angle
    and move to where the tangent line there meets the trajectory (Sales et al., 2024,
    Eq. 19; da Costa et al., 2022, Eq. 13). From a seed on the correct side of the
    collision, the distance changes monotonically.

    Parameters
    ----------
    x0, y0 : float
        Start of the trajectory.
    dx, dy : float
        Unit direction of the trajectory.
    seed : float
        Initial distance along the trajectory.
    largest : float
        Largest distance allowed.
    parameters : NDArray[np.float64]
        Parameters of the billiard.
    boundary_point, tangent : Callable
        Functions of `(q, t, parameters)`, with `q` the polar angle.

    Returns
    -------
    Tuple[float, bool]
        The distance, and whether the iteration converged monotonically.
    """
    distance = seed
    direction = 0.0

    for _ in range(MAX_TANGENT_ITERATIONS):
        angle = np.arctan2(y0 + distance * dy, x0 + distance * dx)
        xa, ya = boundary_point(angle, 0.0, parameters)
        tx, ty = tangent(angle, 0.0, parameters)
        denominator = tx * dy - ty * dx
        if denominator == 0.0:
            return distance, False

        new_distance = (ty * (x0 - xa) - tx * (y0 - ya)) / denominator
        change = new_distance - distance
        # Near grazing, round-off in the distance grows as the angle of incidence falls.
        if abs(change) <= COLLISION_TOLERANCE + 4e-16 / abs(denominator):
            return new_distance, new_distance > 0.0

        if new_distance <= 0.0 or new_distance > largest:
            return new_distance, False
        if change * direction < 0.0:
            return new_distance, False

        direction = change
        distance = new_distance

    return distance, False


@njit
def _distance_to_circle(
    x0: float, y0: float, dx: float, dy: float, radius: float
) -> float:
    """Return the distance along a trajectory to a circle around the origin that contains it."""
    b = x0 * dx + y0 * dy

    return -b + np.sqrt(b * b - (x0 * x0 + y0 * y0 - radius * radius))


# ! ------------------- !
# ! --- Oval solvers --- !
# ! ------------------- !


@njit
def oval_critical_eps(p: float) -> float:
    """Return the value of `eps` above which the oval has negative curvature."""
    return 1.0 / (1.0 + p * p)


@njit
def _oval_fine_collision(
    x0: float,
    y0: float,
    vx: float,
    vy: float,
    t0: float,
    parameters: NDArray[np.float64],
) -> Tuple[float, float, float]:
    """Molecular dynamics with a fine step, for the rare collisions a seed does not reach."""
    return md_collision(
        x0,
        y0,
        vx,
        vy,
        t0,
        oval_inside,
        parameters,
        oval_diameter(parameters) / FALLBACK_MD_STEPS,
    )


@njit
def oval_exact_distance(
    x0: float, y0: float, dx: float, dy: float, eps: float
) -> float:
    """
    Return the distance to the next collision on the oval with `p = 1`, or NaN.

    The boundary is the curve `(x^2 + y^2 - eps x)^2 = x^2 + y^2`. On the trajectory,
    this is a quartic in the distance with a root at zero, the start, which leaves a
    cubic. The collision is its smallest positive real root.
    """
    b = 2.0 * (x0 * dx + y0 * dy)
    k = x0 * x0 + y0 * y0 - eps * x0
    h = b - eps * dx
    a2 = 2.0 * h
    a1 = h * h + 2.0 * k - 1.0
    a0 = 2.0 * h * k - b

    depressed_p = a1 - a2 * a2 / 3.0
    depressed_q = 2.0 * a2 * a2 * a2 / 27.0 - a2 * a1 / 3.0 + a0
    half_q = 0.5 * depressed_q
    third_p = depressed_p / 3.0
    discriminant = half_q * half_q + third_p * third_p * third_p

    roots = np.empty(3)
    count = 0
    if discriminant > 0.0:
        root = np.sqrt(discriminant)
        # The sum with the larger magnitude, to avoid cancellation.
        u = np.cbrt(-half_q - root if half_q > 0.0 else -half_q + root)
        v = -third_p / u if u != 0.0 else 0.0
        roots[0] = u + v - a2 / 3.0
        count = 1
    elif third_p == 0.0:
        roots[0] = -a2 / 3.0
        count = 1
    else:
        modulus = 2.0 * np.sqrt(-third_p)
        cosine = min(max(3.0 * depressed_q / (depressed_p * modulus), -1.0), 1.0)
        angle = np.arccos(cosine)
        for j in range(3):
            roots[j] = modulus * np.cos((angle + 2.0 * np.pi * j) / 3.0) - a2 / 3.0
        count = 3

    nearest = np.inf
    for j in range(count):
        root = roots[j]
        # Newton's method recovers the digits that the closed form loses near a
        # double root, which is where the trajectory almost touches the boundary.
        for _ in range(3):
            slope = (3.0 * root + 2.0 * a2) * root + a1
            if slope == 0.0:
                break
            root -= (((root + a2) * root + a1) * root + a0) / slope
        if COLLISION_TOLERANCE < root < nearest:
            nearest = root

    return nearest if nearest < np.inf else np.nan


@njit
def oval_exact_solver(
    q0: float,
    x0: float,
    y0: float,
    vx: float,
    vy: float,
    t0: float,
    parameters: NDArray[np.float64],
) -> Tuple[float, float, float]:
    eps, p = parameters
    if p != 1.0:
        raise ValueError("solver='exact' on the oval billiard requires p = 1")
    speed = np.sqrt(vx * vx + vy * vy)
    distance = oval_exact_distance(x0, y0, vx / speed, vy / speed, eps)
    if np.isnan(distance):
        return _oval_fine_collision(x0, y0, vx, vy, t0, parameters)
    time = distance / speed

    return t0 + time, x0 + vx * time, y0 + vy * time


@njit
def oval_circle_distance(
    x0: float,
    y0: float,
    dx: float,
    dy: float,
    parameters: NDArray[np.float64],
) -> float:
    """Return the distance to the next collision by Algorithm 1 of da Costa et al. (2022), or NaN."""
    seed = _distance_to_circle(x0, y0, dx, dy, 1.0 + parameters[0])
    distance, converged = _tangent_iteration(
        x0, y0, dx, dy, seed, np.inf, parameters, oval_boundary_point, oval_tangent
    )

    return distance if converged else np.nan


@njit
def oval_circle_solver(
    q0: float,
    x0: float,
    y0: float,
    vx: float,
    vy: float,
    t0: float,
    parameters: NDArray[np.float64],
) -> Tuple[float, float, float]:
    eps, p = parameters
    if eps > oval_critical_eps(p) * (1.0 + 1e-12):
        raise ValueError("solver='circle' requires eps <= 1 / (1 + p^2)")
    speed = np.sqrt(vx * vx + vy * vy)
    distance = oval_circle_distance(x0, y0, vx / speed, vy / speed, parameters)
    if np.isnan(distance):
        return _oval_fine_collision(x0, y0, vx, vy, t0, parameters)
    time = distance / speed

    return t0 + time, x0 + vx * time, y0 + vy * time


@njit
def _distance_to_segment(
    x0: float,
    y0: float,
    dx: float,
    dy: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> float:
    """Return the distance along a trajectory to a segment, or infinity if it misses."""
    ex = bx - ax
    ey = by - ay
    denominator = dx * ey - dy * ex
    if denominator == 0.0:
        return np.inf
    fx = ax - x0
    fy = ay - y0
    distance = (fx * ey - fy * ex) / denominator
    along = (fx * dy - fy * dx) / denominator
    if distance <= COLLISION_TOLERANCE or not -1e-12 <= along <= 1.0 + 1e-12:
        return np.inf

    return distance


@njit
def _lines_meet(
    ax: float,
    ay: float,
    atx: float,
    aty: float,
    bx: float,
    by: float,
    btx: float,
    bty: float,
) -> Tuple[float, float]:
    """Return the point where the line through `a` meets the line through `b`."""
    distance = ((bx - ax) * bty - (by - ay) * btx) / (atx * bty - aty * btx)

    return ax + distance * atx, ay + distance * aty


@njit
def oval_polygon_distance(
    q0: float,
    dx: float,
    dy: float,
    parameters: NDArray[np.float64],
) -> float:
    """
    Return the distance to the next collision by Algorithm 2 of da Costa et al. (2022), or NaN.

    The trajectory is followed in the wedge `[0, pi / p]`, which determines the whole
    boundary, reflecting it at the edges of the wedge. Reflections preserve distances,
    so the distance found is also the one along the true, straight trajectory. The seed
    comes from a polygon with five sides: `L1` and `L2`, tangent to the boundary at
    `theta = pi / p` and at the inflection point, the vertical `L3`, and the edges of the
    wedge `L4` and `L5`. The paper derives the polygon for `p = 3`. It also works for
    larger `p`, as a comparison with dense scans shows, but not for `p = 1`.

    NaN is returned if the method does not converge. The listing in the paper repeats
    forever if the seed from `L1` and `L2` does not converge, so this follows its text,
    which moves on to the other sides.
    """
    eps, p = parameters
    wedge = np.pi / p

    # The inflection point, where the numerator of the curvature vanishes.
    a = eps * eps * (1.0 - p * p)
    b = eps * (2.0 + p * p)
    c = 1.0 + 2.0 * eps * eps * p * p
    if a == 0.0:
        cosine = -c / b
    else:
        cosine = (-b + np.sqrt(b * b - 4.0 * a * c)) / (2.0 * a)
    inflection = np.arccos(cosine) / p

    x1, y1 = oval_boundary_point(wedge, 0.0, parameters)
    t1x, t1y = oval_tangent(wedge, 0.0, parameters)
    xi, yi = oval_boundary_point(inflection, 0.0, parameters)
    tix, tiy = oval_tangent(inflection, 0.0, parameters)
    x2, y2 = _lines_meet(x1, y1, t1x, t1y, xi, yi, tix, tiy)
    x4 = 1.0 + eps
    y3 = yi + (x4 - xi) * tiy / tix if tix != 0.0 else np.inf

    # Fold the start and the direction into the wedge.
    theta = q0 % (2.0 * np.pi)
    wedges = np.floor(theta / (2.0 * wedge))
    theta -= 2.0 * wedge * wedges
    mu = np.arctan2(dy, dx) - 2.0 * wedge * wedges
    if theta > wedge:
        theta = 2.0 * wedge - theta
        mu = 2.0 * wedge - mu
    x0, y0 = oval_boundary_point(theta, 0.0, parameters)

    # On the inward bulge, the start lies beyond L1 and L2.
    use_tangent_sides = not inflection < theta < wedge
    travelled = 0.0

    for _ in range(4 * int(p) + 16):
        dx_r = np.cos(mu)
        dy_r = np.sin(mu)

        if use_tangent_sides:
            to_l1 = _distance_to_segment(x0, y0, dx_r, dy_r, x1, y1, x2, y2)
            to_l2 = _distance_to_segment(x0, y0, dx_r, dy_r, x2, y2, x4, y3)
            crosses_both = to_l1 < np.inf and to_l2 < np.inf
            nearest = min(to_l1, to_l2)
            if nearest < np.inf:
                # After crossing both sides, the trajectory may have missed the bulge.
                largest = max(to_l1, to_l2) if crosses_both else np.inf
                distance, converged = _tangent_iteration(
                    x0,
                    y0,
                    dx_r,
                    dy_r,
                    nearest,
                    largest,
                    parameters,
                    oval_boundary_point,
                    oval_tangent,
                )
                if converged:
                    angle = np.arctan2(y0 + distance * dy_r, x0 + distance * dx_r)
                    if -1e-9 <= angle <= wedge + 1e-9:
                        return travelled + distance
                if not crosses_both:
                    return np.nan

        to_l3 = _distance_to_segment(x0, y0, dx_r, dy_r, x4, y3, x4, 0.0)
        to_l4 = _distance_to_segment(x0, y0, dx_r, dy_r, x4, 0.0, 0.0, 0.0)
        to_l5 = _distance_to_segment(x0, y0, dx_r, dy_r, 0.0, 0.0, x1, y1)
        nearest = min(to_l3, to_l4, to_l5)
        if nearest == np.inf:
            return np.nan

        if to_l3 == nearest:
            distance, converged = _tangent_iteration(
                x0,
                y0,
                dx_r,
                dy_r,
                to_l3,
                np.inf,
                parameters,
                oval_boundary_point,
                oval_tangent,
            )
            return travelled + distance if converged else np.nan

        travelled += nearest
        x0 += nearest * dx_r
        y0 += nearest * dy_r
        mu = -mu if to_l4 == nearest else 2.0 * wedge - mu
        use_tangent_sides = True

    return np.nan


@njit
def oval_polygon_solver(
    q0: float,
    x0: float,
    y0: float,
    vx: float,
    vy: float,
    t0: float,
    parameters: NDArray[np.float64],
) -> Tuple[float, float, float]:
    eps, p = parameters
    if p < 2.0 or not oval_critical_eps(p) < eps < 1.0:
        raise ValueError("solver='polygon' requires p >= 2 and 1 / (1 + p^2) < eps < 1")
    speed = np.sqrt(vx * vx + vy * vy)
    distance = oval_polygon_distance(q0, vx / speed, vy / speed, parameters)
    if np.isnan(distance):
        return _oval_fine_collision(x0, y0, vx, vy, t0, parameters)
    time = distance / speed

    return t0 + time, x0 + vx * time, y0 + vy * time


def oval_auto_choice(parameters: NDArray[np.float64]) -> str:
    """Return the name of the solver that `solver='auto'` uses on the oval billiard."""
    eps, p = parameters
    if p == 1.0:
        return "exact"
    if eps <= 1.0 / (1.0 + p * p):
        return "circle"
    if eps < 1.0:
        return "polygon"

    return "md"


def make_oval_auto_solver(md_steps: int) -> Callable:
    """Build the solver that picks the fastest valid method for the parameters of each call."""
    md_solver = make_md_solver(oval_inside, oval_diameter, md_steps)

    @njit
    def collide(
        q0: float,
        x0: float,
        y0: float,
        vx: float,
        vy: float,
        t0: float,
        parameters: NDArray[np.float64],
    ) -> Tuple[float, float, float]:
        eps, p = parameters
        if p == 1.0:
            return oval_exact_solver(q0, x0, y0, vx, vy, t0, parameters)
        if eps <= oval_critical_eps(p):
            return oval_circle_solver(q0, x0, y0, vx, vy, t0, parameters)
        if eps < 1.0:
            return oval_polygon_solver(q0, x0, y0, vx, vy, t0, parameters)

        return md_solver(q0, x0, y0, vx, vy, t0, parameters)

    return collide


# ! ----------------------------- !
# ! --- Lines and circular arcs --- !
# ! ----------------------------- !

# A boundary made of lines and circular arcs is a table with one row per piece:
#
#   line: [0, start, length, x0, y0, dx, dy, 0]
#   arc:  [1, start, length, center_x, center_y, radius, first_angle, orientation]
#
# where start is the arc length at the beginning of the piece, (dx, dy) is the unit
# direction of a line, and orientation is +1 or -1 for an arc run counterclockwise or
# clockwise. The billiard is on the left of each piece.
_PIECE_COLUMNS = 8


@njit
def new_pieces(count: int) -> NDArray[np.float64]:
    """Return an empty table for a boundary with `count` pieces."""
    return np.zeros((count, _PIECE_COLUMNS))


@njit
def _piece_start(table: NDArray[np.float64], row: int) -> float:
    return table[row - 1, 1] + table[row - 1, 2] if row > 0 else 0.0


@njit
def set_line(
    table: NDArray[np.float64],
    row: int,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> None:
    """
    Make a row of the table the line from `(x0, y0)` to `(x1, y1)`.

    Rows must be set in order, as each starts where the arc length of the previous one
    ends. The billiard must be on the left of the line.
    """
    length = np.hypot(x1 - x0, y1 - y0)
    table[row, 0] = 0.0
    table[row, 1] = _piece_start(table, row)
    table[row, 2] = length
    table[row, 3] = x0
    table[row, 4] = y0
    table[row, 5] = (x1 - x0) / length if length > 0.0 else 0.0
    table[row, 6] = (y1 - y0) / length if length > 0.0 else 0.0
    table[row, 7] = 0.0


@njit
def set_arc(
    table: NDArray[np.float64],
    row: int,
    center_x: float,
    center_y: float,
    radius: float,
    first_angle: float,
    span: float,
) -> None:
    """
    Make a row of the table an arc of a circle.

    The arc starts at `first_angle` and sweeps the angle `span`, counterclockwise if it
    is positive and clockwise if it is negative. Rows must be set in order, as each starts
    where the arc length of the previous one ends. The billiard must be on the left
    of the arc, so an arc that bounds a disk inside the billiard runs clockwise.
    """
    table[row, 0] = 1.0
    table[row, 1] = _piece_start(table, row)
    table[row, 2] = radius * abs(span)
    table[row, 3] = center_x
    table[row, 4] = center_y
    table[row, 5] = radius
    table[row, 6] = first_angle
    table[row, 7] = 1.0 if span >= 0.0 else -1.0


@njit
def piece_point(
    table: NDArray[np.float64], row: int, u: float
) -> Tuple[float, float]:
    """Return the point at the arc length `u` from the beginning of a piece."""
    if table[row, 0] == 0.0:
        return table[row, 3] + table[row, 5] * u, table[row, 4] + table[row, 6] * u

    radius = table[row, 5]
    angle = table[row, 6] + table[row, 7] * u / radius

    return (
        table[row, 3] + radius * np.cos(angle),
        table[row, 4] + radius * np.sin(angle),
    )


@njit
def _locate_piece(table: NDArray[np.float64], s: float) -> Tuple[int, float]:
    last = table.shape[0] - 1
    s = s % (table[last, 1] + table[last, 2])
    row = 0
    while row < last and s >= table[row + 1, 1]:
        row += 1

    return row, s - table[row, 1]


@njit
def _nearest_on_piece(
    table: NDArray[np.float64], row: int, x: float, y: float
) -> Tuple[float, float]:
    """Return the arc length from the beginning of a piece to its point nearest to `(x, y)`, and the squared distance."""
    length = table[row, 2]
    if table[row, 0] == 0.0:
        u = (x - table[row, 3]) * table[row, 5] + (y - table[row, 4]) * table[row, 6]
        u = min(max(u, 0.0), length)
    else:
        radius = table[row, 5]
        angle = np.arctan2(y - table[row, 4], x - table[row, 3])
        swept = (table[row, 7] * (angle - table[row, 6])) % (2.0 * np.pi)
        span = length / radius
        if swept > span:
            # Outside the arc, so the nearest point is one of its ends.
            swept = span if swept - span < 2.0 * np.pi - swept else 0.0
        u = radius * swept

    px, py = piece_point(table, row, u)

    return u, (x - px) ** 2 + (y - py) ** 2


@njit
def _exact_piecewise_collision(
    table: NDArray[np.float64], q0: float, x0: float, y0: float, dx: float, dy: float
) -> float:
    """
    Return the distance from a point on the boundary to the next collision.

    The distance is the smallest positive one over the intersections of the line with
    the pieces, for a unit direction `(dx, dy)`. On the piece that the particle leaves,
    the intersection at the start is removed exactly: a line has no other, and on a
    circle the other one is at `-2 d.(q - c)`.
    """
    leaving, _ = _locate_piece(table, q0)
    nearest = np.inf

    for row in range(table.shape[0]):
        length = table[row, 2]
        if length == 0.0:
            continue

        if table[row, 0] == 0.0:
            if row == leaving:
                continue
            ex = table[row, 5]
            ey = table[row, 6]
            denominator = dx * ey - dy * ex
            if denominator == 0.0:
                continue
            fx = table[row, 3] - x0
            fy = table[row, 4] - y0
            distance = (fx * ey - fy * ex) / denominator
            along = (fx * dy - fy * dx) / denominator
            if (
                COLLISION_TOLERANCE < distance < nearest
                and -PIECE_END_TOLERANCE <= along <= length + PIECE_END_TOLERANCE
            ):
                nearest = distance
            continue

        fx = x0 - table[row, 3]
        fy = y0 - table[row, 4]
        radius = table[row, 5]
        b = fx * dx + fy * dy
        if row == leaving:
            first = -2.0 * b
            second = np.inf
        else:
            discriminant = b * b - (fx * fx + fy * fy - radius * radius)
            if discriminant < 0.0:
                continue
            root = np.sqrt(discriminant)
            first = -b - root
            second = -b + root

        span = length / radius
        for distance in (first, second):
            if not COLLISION_TOLERANCE < distance < nearest or distance == np.inf:
                continue
            angle = np.arctan2(fy + distance * dy, fx + distance * dx)
            swept = (table[row, 7] * (angle - table[row, 6])) % (2.0 * np.pi)
            # Within the arc, or just past one of its ends.
            if (
                swept <= span + PIECE_END_TOLERANCE
                or swept >= 2.0 * np.pi - PIECE_END_TOLERANCE
            ):
                nearest = distance
                break

    if nearest == np.inf:
        raise RuntimeError("no collision found: the boundary is not closed")

    return nearest


@njit
def inside_pieces(table: NDArray[np.float64], x: float, y: float) -> bool:
    """
    Return whether a point is inside a boundary made of lines and arcs.

    The point is inside if the horizontal ray to its right crosses the boundary an odd
    number of times, which also handles disks and other holes.
    """
    crossings = 0
    for row in range(table.shape[0]):
        length = table[row, 2]
        if length == 0.0:
            continue

        if table[row, 0] == 0.0:
            x0 = table[row, 3]
            y0 = table[row, 4]
            y1 = y0 + table[row, 6] * length
            if (y0 > y) != (y1 > y):
                if x0 + (y - y0) * table[row, 5] / table[row, 6] > x:
                    crossings += 1
            continue

        center_x = table[row, 3]
        radius = table[row, 5]
        height = y - table[row, 4]
        if abs(height) >= radius:
            continue
        half_chord = np.sqrt(radius * radius - height * height)
        first_angle = table[row, 6]
        orientation = table[row, 7]
        span = length / radius
        for side in (1.0, -1.0):
            if center_x + side * half_chord <= x:
                continue
            angle = np.arctan2(height, side * half_chord)
            swept = (orientation * (angle - first_angle)) % (2.0 * np.pi)
            if swept > span:
                continue
            # An end of the arc at this height counts if the arc continues above it,
            # as the end of a line does.
            if swept == 0.0 and span < 2.0 * np.pi:
                if orientation * np.cos(angle) <= 0.0:
                    continue
            elif swept == span and span < 2.0 * np.pi:
                if orientation * np.cos(angle) >= 0.0:
                    continue
            crossings += 1

    return crossings % 2 == 1


@njit
def pieces_size(table: NDArray[np.float64]) -> float:
    """Return the diagonal of a box around a boundary made of lines and arcs."""
    x_low = y_low = np.inf
    x_high = y_high = -np.inf
    for row in range(table.shape[0]):
        if table[row, 0] == 0.0:
            x0 = table[row, 3]
            y0 = table[row, 4]
            x1 = x0 + table[row, 5] * table[row, 2]
            y1 = y0 + table[row, 6] * table[row, 2]
            x_low = min(x_low, x0, x1)
            x_high = max(x_high, x0, x1)
            y_low = min(y_low, y0, y1)
            y_high = max(y_high, y0, y1)
        else:
            radius = table[row, 5]
            x_low = min(x_low, table[row, 3] - radius)
            x_high = max(x_high, table[row, 3] + radius)
            y_low = min(y_low, table[row, 4] - radius)
            y_high = max(y_high, table[row, 4] + radius)

    return np.hypot(x_high - x_low, y_high - y_low)


def make_pieces_inside_and_size(pieces: Callable) -> Dict[str, Callable]:
    """Build `inside` and `diameter` for a boundary given by its table of pieces."""

    @njit
    def inside(
        x: float, y: float, t: float, parameters: NDArray[np.float64]
    ) -> bool:
        return inside_pieces(pieces(parameters), x, y)

    @njit
    def diameter(parameters: NDArray[np.float64]) -> float:
        return pieces_size(pieces(parameters))

    return {"inside": inside, "diameter": diameter}


def make_fixed_pieces(table: NDArray[np.float64]) -> Callable:
    """Build the function `pieces(parameters)` of a boundary with no parameters."""

    @njit
    def pieces(parameters: NDArray[np.float64]) -> NDArray[np.float64]:
        return table

    return pieces


@njit
def no_check(parameters: NDArray[np.float64]) -> None:
    pass


def make_piecewise_geometry(pieces: Callable) -> Dict[str, Callable]:
    """
    Build the geometry of a boundary made of lines and circular arcs.

    The position on the boundary is the arc length, so these functions have the same
    signatures as those of the other billiards, with `q = s`.

    Parameters
    ----------
    pieces : Callable
        Function `pieces(parameters)` that returns the table of pieces.

    Returns
    -------
    Dict[str, Callable]
        The geometry functions, by name.
    """

    @njit
    def exact_solver(
        q0: float,
        x0: float,
        y0: float,
        vx: float,
        vy: float,
        t0: float,
        parameters: NDArray[np.float64],
    ) -> Tuple[float, float, float]:
        speed = np.sqrt(vx * vx + vy * vy)
        distance = _exact_piecewise_collision(
            pieces(parameters), q0, x0, y0, vx / speed, vy / speed
        )
        time = distance / speed

        return t0 + time, x0 + vx * time, y0 + vy * time

    @njit
    def boundary_point(
        s: float, t: float, parameters: NDArray[np.float64]
    ) -> Tuple[float, float]:
        table = pieces(parameters)
        row, u = _locate_piece(table, s)

        return piece_point(table, row, u)

    @njit
    def tangent(
        s: float, t: float, parameters: NDArray[np.float64]
    ) -> Tuple[float, float]:
        table = pieces(parameters)
        row, u = _locate_piece(table, s)
        if table[row, 0] == 0.0:
            return table[row, 5], table[row, 6]

        orientation = table[row, 7]
        angle = table[row, 6] + orientation * u / table[row, 5]

        return -orientation * np.sin(angle), orientation * np.cos(angle)

    @njit
    def curvature(s: float, t: float, parameters: NDArray[np.float64]) -> float:
        table = pieces(parameters)
        row, _ = _locate_piece(table, s)
        if table[row, 0] == 0.0:
            return 0.0

        return table[row, 7] / table[row, 5]

    @njit
    def arc_speed(s: float, t: float, parameters: NDArray[np.float64]) -> float:
        return 1.0

    @njit
    def project(
        x: float, y: float, t: float, parameters: NDArray[np.float64]
    ) -> float:
        table = pieces(parameters)
        nearest = 0
        nearest_u, nearest_distance = _nearest_on_piece(table, 0, x, y)
        for row in range(1, table.shape[0]):
            u, distance = _nearest_on_piece(table, row, x, y)
            if distance < nearest_distance:
                nearest = row
                nearest_u = u
                nearest_distance = distance

        last = table.shape[0] - 1

        return (table[nearest, 1] + nearest_u) % (table[last, 1] + table[last, 2])

    @njit
    def perimeter(parameters: NDArray[np.float64]) -> float:
        table = pieces(parameters)
        last = table.shape[0] - 1

        return table[last, 1] + table[last, 2]

    @njit
    def arc_length(s: float, t: float, parameters: NDArray[np.float64]) -> float:
        return s

    @njit
    def arc_position(s: float, t: float, parameters: NDArray[np.float64]) -> float:
        return s

    return {
        "pieces": pieces,
        "exact_solver": exact_solver,
        "boundary_point": boundary_point,
        "tangent": tangent,
        "curvature": curvature,
        "arc_speed": arc_speed,
        "project": project,
        "perimeter": perimeter,
        "arc_length": arc_length,
        "arc_position": arc_position,
    }


# ! ------------------------ !
# ! --- Stadium billiard --- !
# ! ------------------------ !


@njit
def stadium_check(parameters: NDArray[np.float64]) -> None:
    r, l = parameters
    if not r > 0.0:
        raise ValueError("r must be positive")
    if not l >= 0.0:
        raise ValueError("l must not be negative")


@njit
def stadium_diameter(parameters: NDArray[np.float64]) -> float:
    r, l = parameters

    return 2.0 * r + l


@njit
def stadium_inside(
    x: float, y: float, t: float, parameters: NDArray[np.float64]
) -> bool:
    r, l = parameters
    excess = abs(x) - 0.5 * l
    if excess <= 0.0:
        return abs(y) < r

    return excess * excess + y * y < r * r


@njit
def stadium_pieces(parameters: NDArray[np.float64]) -> NDArray[np.float64]:
    # s = 0 at the start of the right semicircle, at (l / 2, -r).
    r, l = parameters
    half = 0.5 * l
    table = new_pieces(4)
    set_arc(table, 0, half, 0.0, r, -0.5 * np.pi, np.pi)
    set_line(table, 1, half, r, -half, r)
    set_arc(table, 2, -half, 0.0, r, 0.5 * np.pi, np.pi)
    set_line(table, 3, -half, -r, half, -r)

    return table


# ! ---------------------- !
# ! --- Sinai billiard --- !
# ! ---------------------- !


@njit
def sinai_check(parameters: NDArray[np.float64]) -> None:
    r, x_d, y_d = parameters
    if not r > 0.0:
        raise ValueError("r must be positive")
    if not (r < x_d < 1.0 - r and r < y_d < 1.0 - r):
        raise ValueError("the disk must lie inside the unit square")


@njit
def sinai_diameter(parameters: NDArray[np.float64]) -> float:
    return np.sqrt(2.0)


@njit
def sinai_inside(
    x: float, y: float, t: float, parameters: NDArray[np.float64]
) -> bool:
    r, x_d, y_d = parameters

    return (
        0.0 < x < 1.0
        and 0.0 < y < 1.0
        and (x - x_d) ** 2 + (y - y_d) ** 2 > r * r
    )


@njit
def sinai_pieces(parameters: NDArray[np.float64]) -> NDArray[np.float64]:
    # s = 0 at the origin, runs counterclockwise along the walls, then clockwise
    # around the disk from its rightmost point, so the billiard stays on the left.
    r, x_d, y_d = parameters
    table = new_pieces(5)
    set_line(table, 0, 0.0, 0.0, 1.0, 0.0)
    set_line(table, 1, 1.0, 0.0, 1.0, 1.0)
    set_line(table, 2, 1.0, 1.0, 0.0, 1.0)
    set_line(table, 3, 0.0, 1.0, 0.0, 0.0)
    set_arc(table, 4, x_d, y_d, r, 0.0, -2.0 * np.pi)

    return table


# ! ------------------------- !
# ! --- Mushroom billiard --- !
# ! ------------------------- !


@njit
def mushroom_check(parameters: NDArray[np.float64]) -> None:
    R, w, h = parameters
    if not R > 0.0:
        raise ValueError("R must be positive")
    if not 0.0 < w < 2.0 * R:
        raise ValueError("w must lie in (0, 2 R)")
    if not h > 0.0:
        raise ValueError("h must be positive")


@njit
def mushroom_diameter(parameters: NDArray[np.float64]) -> float:
    R, w, h = parameters

    return max(2.0 * R, R + np.hypot(0.5 * w, h))


@njit
def mushroom_inside(
    x: float, y: float, t: float, parameters: NDArray[np.float64]
) -> bool:
    R, w, h = parameters
    if y > 0.0:
        return x * x + y * y < R * R
    if abs(x) >= 0.5 * w:
        return False

    return y > -h


@njit
def mushroom_pieces(parameters: NDArray[np.float64]) -> NDArray[np.float64]:
    # s = 0 at the bottom left of the stem, running counterclockwise.
    R, w, h = parameters
    half = 0.5 * w
    table = new_pieces(6)
    set_line(table, 0, -half, -h, half, -h)
    set_line(table, 1, half, -h, half, 0.0)
    set_line(table, 2, half, 0.0, R, 0.0)
    set_arc(table, 3, 0.0, 0.0, R, 0.0, np.pi)
    set_line(table, 4, -R, 0.0, -half, 0.0)
    set_line(table, 5, -half, 0.0, -half, -h)

    return table


# ! ------------------------- !
# ! --- User polar boundary --- !
# ! ------------------------- !


def make_polar_geometry(
    radius: Callable,
    dradius: Callable,
    ddradius: Callable | None,
    max_radius: Callable | None,
    num_panels: int,
) -> Dict[str, Callable]:
    """
    Build the geometry of the boundary `R(theta)` given by a user.

    Parameters
    ----------
    radius, dradius : Callable
        Functions `radius(theta, parameters)` and `dradius(theta, parameters)`.
    ddradius : Callable | None
        Function `ddradius(theta, parameters)`, needed for the curvature.
    max_radius : Callable | None
        Function `max_radius(parameters)`, needed for the circle solver.
    num_panels : int
        Number of Gauss-Legendre panels in the arc length.

    Returns
    -------
    Dict[str, Callable]
        The geometry functions, by name.
    """
    panel = 2.0 * np.pi / num_panels

    @njit
    def inside(
        x: float, y: float, t: float, parameters: NDArray[np.float64]
    ) -> bool:
        return np.sqrt(x * x + y * y) < radius(np.arctan2(y, x), parameters)

    @njit
    def boundary_point(
        theta: float, t: float, parameters: NDArray[np.float64]
    ) -> Tuple[float, float]:
        r = radius(theta, parameters)

        return r * np.cos(theta), r * np.sin(theta)

    @njit
    def tangent(
        theta: float, t: float, parameters: NDArray[np.float64]
    ) -> Tuple[float, float]:
        r = radius(theta, parameters)
        dr = dradius(theta, parameters)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        norm = np.sqrt(r * r + dr * dr)

        return (dr * cos_theta - r * sin_theta) / norm, (dr * sin_theta + r * cos_theta) / norm

    @njit
    def arc_speed(
        theta: float, t: float, parameters: NDArray[np.float64]
    ) -> float:
        r = radius(theta, parameters)
        dr = dradius(theta, parameters)

        return np.sqrt(r * r + dr * dr)

    @njit
    def project(
        x: float, y: float, t: float, parameters: NDArray[np.float64]
    ) -> float:
        return np.arctan2(y, x) % (2.0 * np.pi)

    @njit
    def integral(
        lower: float, upper: float, parameters: NDArray[np.float64]
    ) -> float:
        half_width = 0.5 * (upper - lower)
        center = lower + half_width
        total = 0.0
        for node, weight in zip(_QUADRATURE_NODES, _QUADRATURE_WEIGHTS):
            total += weight * arc_speed(center + half_width * node, 0.0, parameters)

        return half_width * total

    @njit
    def perimeter(parameters: NDArray[np.float64]) -> float:
        total = 0.0
        for j in range(num_panels):
            total += integral(j * panel, (j + 1) * panel, parameters)

        return total

    @njit
    def arc_length(
        theta: float, t: float, parameters: NDArray[np.float64]
    ) -> float:
        theta = theta % (2.0 * np.pi)
        whole = min(int(theta / panel), num_panels - 1)
        total = 0.0
        for j in range(whole):
            total += integral(j * panel, (j + 1) * panel, parameters)

        return total + integral(whole * panel, theta, parameters)

    @njit
    def arc_position(
        s: float, t: float, parameters: NDArray[np.float64]
    ) -> float:
        s = s % perimeter(parameters)
        j = 0
        size = integral(0.0, panel, parameters)
        while j < num_panels - 1 and s > size:
            s -= size
            j += 1
            size = integral(j * panel, (j + 1) * panel, parameters)

        # Newton's method inside the panel, kept within it.
        lower = j * panel
        upper = (j + 1) * panel
        x = lower + panel * min(s / size, 1.0)
        for _ in range(50):
            speed = arc_speed(x, 0.0, parameters)
            if speed == 0.0:
                break
            x_new = x - (integral(lower, x, parameters) - s) / speed
            x_new = min(max(x_new, lower), upper)
            converged = abs(x_new - x) < 1e-15
            x = x_new
            if converged:
                break

        return x % (2.0 * np.pi)

    geometry = {
        "inside": inside,
        "boundary_point": boundary_point,
        "tangent": tangent,
        "arc_speed": arc_speed,
        "project": project,
        "perimeter": perimeter,
        "arc_length": arc_length,
        "arc_position": arc_position,
    }

    if ddradius is None:

        @njit
        def curvature(
            theta: float, t: float, parameters: NDArray[np.float64]
        ) -> float:
            raise ValueError("the curvature requires ddradius")

    else:

        @njit
        def curvature(
            theta: float, t: float, parameters: NDArray[np.float64]
        ) -> float:
            r = radius(theta, parameters)
            dr = dradius(theta, parameters)
            ddr = ddradius(theta, parameters)
            norm2 = r * r + dr * dr

            return (r * r + 2.0 * dr * dr - r * ddr) / norm2**1.5

    geometry["curvature"] = curvature

    if max_radius is None:

        @njit
        def diameter(parameters: NDArray[np.float64]) -> float:
            largest = 0.0
            for j in range(128):
                largest = max(largest, radius(j * np.pi / 64.0, parameters))

            return 2.0 * largest

    else:

        @njit
        def diameter(parameters: NDArray[np.float64]) -> float:
            return 2.0 * max_radius(parameters)

        @njit
        def circle_solver(
            q0: float,
            x0: float,
            y0: float,
            vx: float,
            vy: float,
            t0: float,
            parameters: NDArray[np.float64],
        ) -> Tuple[float, float, float]:
            speed = np.sqrt(vx * vx + vy * vy)
            dx = vx / speed
            dy = vy / speed
            distance, converged = _tangent_iteration(
                x0,
                y0,
                dx,
                dy,
                _distance_to_circle(x0, y0, dx, dy, max_radius(parameters)),
                np.inf,
                parameters,
                boundary_point,
                tangent,
            )
            if not converged:
                raise ValueError(
                    "solver='circle' did not converge: the boundary must be convex"
                )
            time = distance / speed

            return t0 + time, x0 + vx * time, y0 + vy * time

        geometry["circle_solver"] = circle_solver

    geometry["diameter"] = diameter

    return geometry
