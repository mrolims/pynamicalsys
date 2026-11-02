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

from typing import Any, Callable, Tuple

import numpy as np
from numba import njit
from numpy.typing import NDArray

# Smallest length resolved when locating a collision.
COLLISION_TOLERANCE = 1e-14


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
    alpha: float,
    parameters: NDArray[np.float64],
    inside: Callable,
    boundary_point: Callable,
    tangent: Callable,
    project: Callable,
    step_length: float,
) -> Tuple[float, float, float]:
    """Return `(q1, alpha1, chord)` for a collision at `(q, alpha)` with a static wall."""
    if not 0.0 < alpha < np.pi:
        raise ValueError("alpha must lie in (0, pi)")

    x0, y0 = boundary_point(q, 0.0, parameters)
    tx, ty = tangent(q, 0.0, parameters)
    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)
    vx = cos_alpha * tx - sin_alpha * ty
    vy = sin_alpha * tx + cos_alpha * ty

    chord, x1, y1 = md_collision(x0, y0, vx, vy, 0.0, inside, parameters, step_length)

    q1 = project(x1, y1, 0.0, parameters)
    tx, ty = tangent(q1, 0.0, parameters)
    vx, vy = reflect(vx, vy, tx, ty, 0.0, 0.0)
    alpha1 = np.arctan2(tx * vy - ty * vx, tx * vx + ty * vy)

    return q1, alpha1, chord


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
        Function `diameter(parameters)` that returns the longest chord allowed.
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
        q1, alpha1, _ = _static_collision(
            q,
            alpha,
            parameters,
            inside,
            boundary_point,
            tangent,
            project,
            diameter(parameters) / md_steps,
        )

        return np.array([q1, alpha1])

    @njit
    def jacobian(
        u: NDArray[np.float64], parameters: NDArray[np.float64], *args: Any
    ) -> NDArray[np.float64]:
        check(parameters)
        q, alpha = u
        q1, alpha1, chord = _static_collision(
            q,
            alpha,
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
        sin_alpha = np.sin(alpha)
        sin_alpha1 = np.sin(alpha1)

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


# ! --------------------- !
# ! --- Oval billiard --- !
# ! --------------------- !


@njit
def oval_check(parameters: NDArray[np.float64]) -> None:
    eps, p = parameters
    if not 0.0 <= eps < 1.0:
        raise ValueError("eps must lie in [0, 1)")
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
