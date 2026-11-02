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
    inside: Callable,
    boundary_point: Callable,
    tangent: Callable,
    project: Callable,
    step_length: float,
) -> Tuple[float, float, float, float]:
    """Return `(q1, cos_alpha1, sin_alpha1, chord)` for a collision with a static wall."""
    x0, y0 = boundary_point(q, 0.0, parameters)
    tx, ty = tangent(q, 0.0, parameters)
    vx = cos_alpha * tx - sin_alpha * ty
    vy = sin_alpha * tx + cos_alpha * ty

    chord, x1, y1 = md_collision(x0, y0, vx, vy, 0.0, inside, parameters, step_length)

    q1 = project(x1, y1, 0.0, parameters)
    tx, ty = tangent(q1, 0.0, parameters)
    vx, vy = reflect(vx, vy, tx, ty, 0.0, 0.0)

    return q1, tx * vx + ty * vy, tx * vy - ty * vx, chord


def make_angle_maps(
    check: Callable,
    diameter: Callable,
    inside: Callable,
    boundary_point: Callable,
    tangent: Callable,
    curvature: Callable,
    arc_speed: Callable,
    project: Callable,
    md_steps: int,
) -> Tuple[Callable, Callable, Callable]:
    """
    Build the map of a static billiard in the coordinates `(q, alpha)`.

    Parameters
    ----------
    check : Callable
        Function `check(parameters)` that raises if the parameters are invalid.
    diameter : Callable
        Function `diameter(parameters)` that returns the size of the billiard.
    inside : Callable
        Function `inside(x, y, t, parameters)`.
    boundary_point, tangent : Callable
        Functions of `(q, t, parameters)` that return the point and the unit
        tangent of the boundary, with the billiard on the left of the tangent.
    curvature, arc_speed : Callable
        Functions of `(q, t, parameters)` that return the signed curvature and
        `ds/dq`, the arc length per unit of `q`.
    project : Callable
        Function `project(x, y, t, parameters)` that returns `q`.
    md_steps : int
        Number of molecular dynamics steps across the diameter of the billiard.

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
            inside,
            boundary_point,
            tangent,
            project,
            diameter(parameters) / md_steps,
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
            inside,
            boundary_point,
            tangent,
            project,
            diameter(parameters) / md_steps,
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
    diameter: Callable,
    inside: Callable,
    boundary_point: Callable,
    tangent: Callable,
    curvature: Callable,
    project: Callable,
    arc_length: Callable,
    arc_position: Callable,
    md_steps: int,
) -> Tuple[Callable, Callable, Callable]:
    """
    Build the map of a static billiard in the Birkhoff coordinates `(s, p)`.

    Here `s` is the arc length along the boundary and `p = cos(alpha)`. These
    coordinates are canonical, so the map preserves area.

    Parameters
    ----------
    check, diameter, inside, boundary_point, tangent, curvature, project : Callable
        As in `make_angle_maps`.
    arc_length : Callable
        Function `arc_length(q, t, parameters)` that returns `s`.
    arc_position : Callable
        Function `arc_position(s, t, parameters)` that returns `q`.
    md_steps : int
        Number of molecular dynamics steps across the diameter of the billiard.

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
            inside,
            boundary_point,
            tangent,
            project,
            diameter(parameters) / md_steps,
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
            inside,
            boundary_point,
            tangent,
            project,
            diameter(parameters) / md_steps,
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
def _set_line(
    table: NDArray[np.float64],
    row: int,
    start: float,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> float:
    length = np.hypot(x1 - x0, y1 - y0)
    table[row, 0] = 0.0
    table[row, 1] = start
    table[row, 2] = length
    table[row, 3] = x0
    table[row, 4] = y0
    table[row, 5] = (x1 - x0) / length
    table[row, 6] = (y1 - y0) / length
    table[row, 7] = 0.0

    return start + length


@njit
def _set_arc(
    table: NDArray[np.float64],
    row: int,
    start: float,
    center_x: float,
    center_y: float,
    radius: float,
    first_angle: float,
    span: float,
    orientation: float,
) -> float:
    length = radius * span
    table[row, 0] = 1.0
    table[row, 1] = start
    table[row, 2] = length
    table[row, 3] = center_x
    table[row, 4] = center_y
    table[row, 5] = radius
    table[row, 6] = first_angle
    table[row, 7] = orientation

    return start + length


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
    table = np.empty((4, _PIECE_COLUMNS))
    s = _set_arc(table, 0, 0.0, half, 0.0, r, -0.5 * np.pi, np.pi, 1.0)
    # A stadium with l = 0 is a circle, and its straight pieces have no direction.
    if l > 0.0:
        s = _set_line(table, 1, s, half, r, -half, r)
    else:
        table[1] = [0.0, s, 0.0, 0.0, r, -1.0, 0.0, 0.0]
    s = _set_arc(table, 2, s, -half, 0.0, r, 0.5 * np.pi, np.pi, 1.0)
    if l > 0.0:
        _set_line(table, 3, s, -half, -r, half, -r)
    else:
        table[3] = [0.0, s, 0.0, 0.0, -r, 1.0, 0.0, 0.0]

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
    table = np.empty((5, _PIECE_COLUMNS))
    s = _set_line(table, 0, 0.0, 0.0, 0.0, 1.0, 0.0)
    s = _set_line(table, 1, s, 1.0, 0.0, 1.0, 1.0)
    s = _set_line(table, 2, s, 1.0, 1.0, 0.0, 1.0)
    s = _set_line(table, 3, s, 0.0, 1.0, 0.0, 0.0)
    _set_arc(table, 4, s, x_d, y_d, r, 0.0, 2.0 * np.pi, -1.0)

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
    table = np.empty((6, _PIECE_COLUMNS))
    s = _set_line(table, 0, 0.0, -half, -h, half, -h)
    s = _set_line(table, 1, s, half, -h, half, 0.0)
    s = _set_line(table, 2, s, half, 0.0, R, 0.0)
    s = _set_arc(table, 3, s, 0.0, 0.0, R, 0.0, np.pi, 1.0)
    s = _set_line(table, 4, s, -R, 0.0, -half, 0.0)
    _set_line(table, 5, s, -half, 0.0, -half, -h)

    return table
