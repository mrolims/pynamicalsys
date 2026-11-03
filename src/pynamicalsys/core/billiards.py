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

import warnings
from functools import lru_cache
from numbers import Integral, Real
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np
from numba.core.registry import CPUDispatcher
from numpy.typing import NDArray

from pynamicalsys.common.types import int_t, numeric_like_t, numeric_t
from pynamicalsys.common.validators import validate_parameters, validate_positive
from pynamicalsys.core.discrete_dynamical_systems import DiscreteDynamicalSystem
from pynamicalsys.discrete_time.billiards import (
    inside_pieces,
    make_angle_maps,
    make_birkhoff_maps,
    make_fixed_pieces,
    make_md_solver,
    make_pieces_inside_and_size,
    make_piecewise_geometry,
    make_polar_geometry,
    mushroom_check,
    mushroom_diameter,
    mushroom_inside,
    mushroom_pieces,
    new_pieces,
    no_check,
    oval_arc_length,
    oval_arc_position,
    oval_arc_speed,
    oval_boundary_point,
    oval_check,
    oval_circle_solver,
    oval_curvature,
    oval_diameter,
    oval_exact_solver,
    oval_inside,
    oval_perimeter,
    oval_polygon_solver,
    oval_project,
    oval_tangent,
    piece_point,
    set_arc,
    set_line,
    sinai_check,
    sinai_diameter,
    sinai_inside,
    sinai_pieces,
    stadium_check,
    stadium_diameter,
    stadium_inside,
    stadium_pieces,
)

_BILLIARDS: Dict[str, Dict[str, Any]] = {
    "oval": {
        "name": "oval",
        "description": "Oval billiard, R(theta) = 1 + eps cos(p theta)",
        "parameters": ["eps", "p"],
        "position": "theta",
        "position_range": (0.0, 2.0 * np.pi),
        "period": 2.0 * np.pi,
        "solvers": ["md", "exact", "circle", "polygon"],
        "solver_functions": {
            "exact": oval_exact_solver,
            "circle": oval_circle_solver,
            "polygon": oval_polygon_solver,
        },
        "check": oval_check,
        "diameter": oval_diameter,
        "inside": oval_inside,
        "boundary_point": oval_boundary_point,
        "tangent": oval_tangent,
        "curvature": oval_curvature,
        "arc_speed": oval_arc_speed,
        "project": oval_project,
        "arc_length": oval_arc_length,
        "arc_position": oval_arc_position,
        "perimeter": oval_perimeter,
    },
    "stadium": {
        "name": "stadium",
        "description": "Stadium billiard: semicircles of radius r joined by "
        "straight segments of length l",
        "parameters": ["r", "l"],
        "position": "s",
        "closed": True,
        "solvers": ["md", "exact"],
        "check": stadium_check,
        "diameter": stadium_diameter,
        "inside": stadium_inside,
        **make_piecewise_geometry(stadium_pieces),
    },
    "sinai": {
        "name": "sinai",
        "description": "Sinai billiard: a disk of radius r at (x_d, y_d) in the "
        "unit square with reflecting walls",
        "parameters": ["r", "x_d", "y_d"],
        "position": "s",
        "closed": False,
        "solvers": ["md", "exact"],
        "check": sinai_check,
        "diameter": sinai_diameter,
        "inside": sinai_inside,
        **make_piecewise_geometry(sinai_pieces),
    },
    "mushroom": {
        "name": "mushroom",
        "description": "Mushroom billiard: a semicircular hat of radius R on a "
        "centered stem of width w and height h",
        "parameters": ["R", "w", "h"],
        "position": "s",
        "closed": True,
        "solvers": ["md", "exact"],
        "check": mushroom_check,
        "diameter": mushroom_diameter,
        "inside": mushroom_inside,
        **make_piecewise_geometry(mushroom_pieces),
    },
}


# Billiards defined by users, by the functions or pieces that define them, so that the
# same definition is compiled only once.
_USER_BILLIARDS: Dict[Tuple[Any, ...], Dict[str, Any]] = {}

# Pieces closer than this touch, as the ends of connected pieces do.
_TOUCH_TOLERANCE = 1e-9

_INVALID_FIGURE = "Invalid billiard boundary"

_CONVEXITY_POINTS = 5000


def _find_billiard(name: Any) -> Dict[str, Any]:
    return _USER_BILLIARDS[name] if isinstance(name, tuple) else _BILLIARDS[name]


@lru_cache(maxsize=None)
def _collision_solver(name: Any, solver: str, md_steps: int) -> Callable:
    billiard = _find_billiard(name)
    if solver == "md":
        return make_md_solver(billiard["inside"], billiard["diameter"], md_steps)
    if "solver_functions" in billiard:
        return billiard["solver_functions"][solver]

    return billiard["exact_solver"]


@lru_cache(maxsize=None)
def _billiard_maps(
    name: Any, coordinates: str, solver: str, md_steps: int
) -> Tuple[Callable, Callable, Callable]:
    # Each map is compiled once for every billiard, coordinates, and solver, and for
    # every number of steps with molecular dynamics.
    billiard = _find_billiard(name)
    collide = _collision_solver(name, solver, md_steps)

    if coordinates == "birkhoff":
        return make_birkhoff_maps(
            billiard["check"],
            collide,
            billiard["boundary_point"],
            billiard["tangent"],
            billiard["curvature"],
            billiard["project"],
            billiard["arc_length"],
            billiard["arc_position"],
        )

    return make_angle_maps(
        billiard["check"],
        collide,
        billiard["boundary_point"],
        billiard["tangent"],
        billiard["curvature"],
        billiard["arc_speed"],
        billiard["project"],
    )


def _require_jitted(function: Any, name: str) -> None:
    if not isinstance(function, CPUDispatcher):
        raise TypeError(f"{name} must be a function decorated with @njit")


def _validate_polar_functions(
    radius: Callable,
    dradius: Callable,
    ddradius: Callable | None,
    max_radius: Callable | None,
    solver: str,
    convexity: Callable,
    parameters: NDArray[np.float64],
) -> None:
    """Check the functions of a polar boundary along it, for the given parameters."""
    h = 1e-5
    thetas = np.linspace(0.0, 2.0 * np.pi, 257)
    r = np.array([radius(theta, parameters) for theta in thetas])
    dr = np.array([dradius(theta, parameters) for theta in thetas])

    def draw(
        axes: Any,
        marks: NDArray[np.bool_],
        angles: NDArray[np.float64] = thetas,
        radii: NDArray[np.float64] = r,
    ) -> None:
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        axes.plot(x, y, "k", lw=1.2)
        axes.plot(x[marks], y[marks], "o", color="tab:red", ms=4.0, mfc="none")
        axes.set_aspect("equal")

    if np.any(r <= 0.0):
        raise _invalid_boundary(
            "radius must be positive along the boundary",
            lambda axes: draw(axes, r <= 0.0),
        )
    if abs(r[0] - r[-1]) > 1e-9 * abs(r[0]):
        ends = np.zeros(thetas.size, dtype=bool)
        ends[[0, -1]] = True
        raise _invalid_boundary(
            "radius must have period 2 pi", lambda axes: draw(axes, ends)
        )

    def matches(values: NDArray[np.float64], derivative: Callable) -> bool:
        numerical = np.array(
            [
                (derivative(theta + h, parameters) - derivative(theta - h, parameters))
                / (2.0 * h)
                for theta in thetas
            ]
        )
        scale = 1.0 + np.max(np.abs(numerical))
        return bool(np.max(np.abs(values - numerical)) <= 1e-4 * scale)

    if not matches(dr, radius):
        raise ValueError("dradius is not the derivative of radius")

    if ddradius is not None:
        ddr = np.array([ddradius(theta, parameters) for theta in thetas])
        if not matches(ddr, dradius):
            raise ValueError("ddradius is not the derivative of dradius")

    if max_radius is not None and max_radius(parameters) < np.max(r) * (1.0 - 1e-12):
        raise ValueError("max_radius is smaller than the radius along the boundary")

    if solver == "circle":
        # A sampled check, on a grid fine enough for dents of about 1e-3 in the angle.
        fine = np.linspace(0.0, 2.0 * np.pi, _CONVEXITY_POINTS)
        concave = convexity(fine, parameters) < -1e-9
        if np.any(concave):
            radii = np.array([radius(theta, parameters) for theta in fine])
            raise _invalid_boundary(
                "solver='circle' requires a convex boundary. On a boundary that is not "
                "convex, use solver='md'",
                lambda axes: draw(axes, concave, fine, radii),
            )


def _warn_if_arc_length_is_coarse(
    radius: Callable,
    dradius: Callable,
    num_panels: int,
    perimeter: float,
    parameters: NDArray[np.float64],
) -> None:
    """Warn if twice the panels changes the perimeter of a polar boundary."""
    nodes, weights = np.polynomial.legendre.leggauss(12)
    width = np.pi / num_panels
    total = 0.0
    for j in range(2 * num_panels):
        center = (j + 0.5) * width
        for node, weight in zip(nodes, weights):
            theta = center + 0.5 * width * node
            total += (
                0.5
                * width
                * weight
                * np.hypot(radius(theta, parameters), dradius(theta, parameters))
            )
    if abs(total - perimeter) > 1e-10 * total:
        warnings.warn(
            f"the arc length with num_panels={num_panels} has an error of about "
            f"{abs(total - perimeter):.1e}. Increase num_panels.",
            UserWarning,
            stacklevel=4,
        )


def _fixed_pieces_table(pieces: Sequence[Sequence[Any]]) -> NDArray[np.float64]:
    """Build the table of a boundary from `("line", ...)` and `("arc", ...)` entries."""
    if len(pieces) == 0:
        raise ValueError("pieces must not be empty")

    table = new_pieces(len(pieces))
    for row, piece in enumerate(pieces):
        if len(piece) == 0 or piece[0] not in ("line", "arc"):
            raise ValueError(f"piece {row} must start with 'line' or 'arc'")
        values = piece[1:]
        if piece[0] == "line" and len(values) != 4:
            raise ValueError(f"line {row} must be ('line', x0, y0, x1, y1)")
        if piece[0] == "arc" and len(values) != 5:
            raise ValueError(
                f"arc {row} must be ('arc', center_x, center_y, radius, first_angle, span)"
            )
        if not all(isinstance(value, Real) for value in values):
            raise TypeError(f"piece {row} must have numeric values")
        values = [float(value) for value in values]
        if piece[0] == "line":
            set_line(table, row, *values)
        else:
            set_arc(table, row, *values)

    return table


def _pieces_components(table: NDArray[np.float64]) -> List[Tuple[int, int]]:
    """Return the rows `(first, last)` of each connected part of a boundary."""
    components = []
    first = 0
    for row in range(table.shape[0] - 1):
        end = piece_point(table, row, table[row, 2])
        start = piece_point(table, row + 1, 0.0)
        if np.hypot(end[0] - start[0], end[1] - start[1]) > _TOUCH_TOLERANCE:
            components.append((first, row))
            first = row + 1
    components.append((first, table.shape[0] - 1))

    return components


def _on_piece(
    table: NDArray[np.float64], row: int, x: float, y: float, margin: float
) -> bool:
    """Tell if a point of the line or circle of a piece is within the piece."""
    length = table[row, 2]
    if table[row, 0] == 0.0:
        position = (x - table[row, 3]) * table[row, 5] + (y - table[row, 4]) * table[row, 6]
    else:
        radius = table[row, 5]
        angle = np.arctan2(y - table[row, 4], x - table[row, 3])
        swept = (table[row, 7] * (angle - table[row, 6])) % (2.0 * np.pi)
        position = radius * swept
        if position > length + margin:
            position -= 2.0 * np.pi * radius

    return -margin <= position <= length + margin


def _same_line_or_circle(table: NDArray[np.float64], i: int, j: int) -> bool:
    if table[i, 0] != table[j, 0]:
        return False
    if table[i, 0] == 1.0:
        return (
            np.hypot(table[i, 3] - table[j, 3], table[i, 4] - table[j, 4])
            <= _TOUCH_TOLERANCE
            and abs(table[i, 5] - table[j, 5]) <= _TOUCH_TOLERANCE
        )
    parallel = abs(table[i, 5] * table[j, 6] - table[i, 6] * table[j, 5]) <= 1e-12
    offset = (table[j, 3] - table[i, 3]) * table[i, 6] - (
        table[j, 4] - table[i, 4]
    ) * table[i, 5]

    return parallel and abs(offset) <= _TOUCH_TOLERANCE


def _line_circle_points(
    x0: float, y0: float, dx: float, dy: float, cx: float, cy: float, radius: float
) -> List[Tuple[float, float]]:
    along = (cx - x0) * dx + (cy - y0) * dy
    distance = abs((cx - x0) * dy - (cy - y0) * dx)
    if distance > radius + _TOUCH_TOLERANCE:
        return []
    # A tangent line meets the circle once, whatever the round-off in the distance.
    half = (
        0.0
        if abs(distance - radius) <= _TOUCH_TOLERANCE
        else np.sqrt(radius * radius - distance * distance)
    )

    return [
        (x0 + (along + side * half) * dx, y0 + (along + side * half) * dy)
        for side in (1.0, -1.0)
    ]


def _circle_circle_points(
    x0: float, y0: float, r0: float, x1: float, y1: float, r1: float
) -> List[Tuple[float, float]]:
    distance = np.hypot(x1 - x0, y1 - y0)
    if (
        distance > r0 + r1 + _TOUCH_TOLERANCE
        or distance < abs(r0 - r1) - _TOUCH_TOLERANCE
        or distance == 0.0
    ):
        return []
    along = (r0 * r0 - r1 * r1 + distance * distance) / (2.0 * distance)
    # Tangent circles meet once, whatever the round-off in the distance.
    tangent = min(abs(distance - r0 - r1), abs(distance - abs(r0 - r1)))
    half = (
        0.0
        if tangent <= _TOUCH_TOLERANCE
        else np.sqrt(max(r0 * r0 - along * along, 0.0))
    )
    ux = (x1 - x0) / distance
    uy = (y1 - y0) / distance

    return [
        (
            x0 + along * ux - side * half * uy,
            y0 + along * uy + side * half * ux,
        )
        for side in (1.0, -1.0)
    ]


def _pieces_meeting_point(
    table: NDArray[np.float64], i: int, j: int, joints: List[Tuple[float, float]]
) -> Tuple[float, float] | None:
    """Return a point shared by two pieces, other than the joints between them."""
    if _same_line_or_circle(table, i, j):
        # Only a stretch in common counts, so test the ends and middle of each piece
        # against the interior of the other.
        for row, other in ((i, j), (j, i)):
            for u in (0.0, 0.5 * table[row, 2], table[row, 2]):
                x, y = piece_point(table, row, u)
                if _on_piece(table, other, x, y, -_TOUCH_TOLERANCE):
                    return x, y
        return None

    if table[i, 0] == 0.0 and table[j, 0] == 0.0:
        cross = table[i, 5] * table[j, 6] - table[i, 6] * table[j, 5]
        if abs(cross) <= 1e-12:
            return None
        u = (
            (table[j, 3] - table[i, 3]) * table[j, 6]
            - (table[j, 4] - table[i, 4]) * table[j, 5]
        ) / cross
        points = [(table[i, 3] + u * table[i, 5], table[i, 4] + u * table[i, 6])]
    elif table[i, 0] == 1.0 and table[j, 0] == 1.0:
        points = _circle_circle_points(
            table[i, 3], table[i, 4], table[i, 5], table[j, 3], table[j, 4], table[j, 5]
        )
    else:
        line, arc = (i, j) if table[i, 0] == 0.0 else (j, i)
        points = _line_circle_points(
            table[line, 3],
            table[line, 4],
            table[line, 5],
            table[line, 6],
            table[arc, 3],
            table[arc, 4],
            table[arc, 5],
        )

    for x, y in points:
        if (
            _on_piece(table, i, x, y, _TOUCH_TOLERANCE)
            and _on_piece(table, j, x, y, _TOUCH_TOLERANCE)
            and all(
                np.hypot(x - joint[0], y - joint[1]) > _TOUCH_TOLERANCE
                for joint in joints
            )
        ):
            return x, y

    return None


def _pieces_name(first: int, last: int) -> str:
    return f"piece {first}" if first == last else f"pieces {first} to {last}"


def _pieces_problem(
    table: NDArray[np.float64],
) -> Tuple[str, List[int], List[Tuple[float, float]]] | None:
    """Return what is wrong with a boundary, the pieces, and the points involved."""
    for row in range(table.shape[0]):
        if table[row, 0] == 1.0 and table[row, 5] <= 0.0:
            point = (table[row, 3], table[row, 4])
            return f"piece {row} must have a positive radius", [row], [point]
        if table[row, 2] <= 0.0:
            point = (table[row, 3], table[row, 4])
            return f"piece {row} must have a positive length", [row], [point]
        if table[row, 0] == 1.0 and table[row, 2] / table[row, 5] > 2.0 * np.pi * (
            1.0 + 1e-12
        ):
            return f"arc {row} must sweep at most 2 pi", [row], []

    components = _pieces_components(table)
    following = {}
    for first, last in components:
        end = piece_point(table, last, table[last, 2])
        start = piece_point(table, first, 0.0)
        if np.hypot(end[0] - start[0], end[1] - start[1]) > _TOUCH_TOLERANCE:
            return (
                "the pieces must be connected: each part of the boundary must end "
                f"where it starts, but piece {last} ends at "
                f"({end[0]:.6g}, {end[1]:.6g}) and piece {first} starts at "
                f"({start[0]:.6g}, {start[1]:.6g})",
                [first, last],
                [start, end],
            )
        for row in range(first, last + 1):
            following[row] = row + 1 if row < last else first

    for i in range(table.shape[0]):
        for j in range(i + 1, table.shape[0]):
            joints = []
            if following[i] == j:
                joints.append(piece_point(table, j, 0.0))
            if following[j] == i:
                joints.append(piece_point(table, i, 0.0))
            point = _pieces_meeting_point(table, i, j, joints)
            if point is not None:
                return (
                    f"pieces {i} and {j} cross or touch at "
                    f"({point[0]:.6g}, {point[1]:.6g})",
                    [i, j],
                    [point],
                )

    areas = []
    for first, last in components:

        # The signed area of the part, from the boundary integral of (x dy - y dx) / 2.
        area = 0.0
        for row in range(first, last + 1):
            x0, y0 = piece_point(table, row, 0.0)
            x1, y1 = piece_point(table, row, table[row, 2])
            if table[row, 0] == 0.0:
                area += 0.5 * (x0 * y1 - x1 * y0)
            else:
                radius = table[row, 5]
                center_x = table[row, 3]
                center_y = table[row, 4]
                span = table[row, 7] * table[row, 2] / radius
                area += 0.5 * (
                    radius * radius * span
                    + center_x * (y1 - y0)
                    - center_y * (x1 - x0)
                )
        areas.append(area)

    outer = int(np.argmax(np.abs(areas)))
    wrong = [
        j
        for j, area in enumerate(areas)
        if (area <= 0.0 if j == outer else area >= 0.0)
    ]
    if wrong:
        first, last = components[wrong[0]]
        return (
            "the pieces run the wrong way: the outer boundary must run counterclockwise, "
            "and the boundary of a hole clockwise, but the part with "
            f"{_pieces_name(first, last)} runs the other way",
            list(range(first, last + 1)),
            [],
        )

    outer_first, outer_last = components[outer]
    for j, (first, last) in enumerate(components):
        if j == outer:
            continue
        rows = list(range(first, last + 1))
        point = piece_point(table, first, 0.0)
        if not inside_pieces(table[outer_first : outer_last + 1], *point):
            return (
                f"the hole with {_pieces_name(first, last)} is outside the billiard",
                rows,
                [point],
            )
        for k, (other_first, other_last) in enumerate(components):
            if k not in (j, outer) and inside_pieces(
                table[other_first : other_last + 1], *point
            ):
                return (
                    f"the hole with {_pieces_name(first, last)} is inside the hole "
                    f"with {_pieces_name(other_first, other_last)}",
                    rows,
                    [point],
                )

    return None


def _invalid_boundary(message: str, draw: Callable) -> ValueError:
    """Draw a boundary that is not valid, and return the error to raise."""
    # One figure is reused, so that a loop over invalid boundaries doesn't open many.
    # The error matters more than the figure, so a failure to draw does not replace it.
    try:
        import matplotlib.pyplot as plt

        figure = plt.figure(_INVALID_FIGURE, clear=True)
        axes = figure.add_subplot()
        draw(axes)
        axes.set_title(message, fontsize=9, color="tab:red", wrap=True)
    except Exception:
        return ValueError(message)

    return ValueError(
        f"{message}. The boundary is drawn in the figure '{_INVALID_FIGURE}': "
        "notebooks display it, and a script needs plt.show()."
    )


def _draw_pieces(
    axes: Any,
    table: NDArray[np.float64],
    rows: Sequence[int] = (),
    points: Sequence[Tuple[float, float]] = (),
) -> None:
    """Draw numbered pieces with their directions, and mark those with a problem."""
    extent = []
    drawn = []
    for row in range(table.shape[0]):
        length = table[row, 2]
        if length <= 0.0 or (table[row, 0] == 1.0 and table[row, 5] <= 0.0):
            continue
        count = 2 if table[row, 0] == 0.0 else 129
        curve = np.array(
            [piece_point(table, row, u) for u in np.linspace(0.0, length, count)]
        )
        color = "tab:red" if row in rows else "black"
        axes.plot(curve[:, 0], curve[:, 1], color=color, lw=2.0 if row in rows else 1.2)
        extent.append(curve)
        drawn.append((row, color))
    extent = np.vstack(extent) if extent else np.zeros((1, 2))
    size = max(float(np.ptp(extent, axis=0).max()), 1e-12)

    for row, color in drawn:
        length = table[row, 2]
        # An arrow that stays on the piece, and the number on the side of the billiard.
        step = min(0.01 * size, 0.1 * length)
        back = np.array(piece_point(table, row, 0.5 * length - step))
        middle = np.array(piece_point(table, row, 0.5 * length))
        ahead = np.array(piece_point(table, row, 0.5 * length + step))
        axes.annotate(
            "",
            xy=ahead,
            xytext=back,
            arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.2},
        )
        direction = (ahead - back) / np.hypot(*(ahead - back))
        left = np.array([-direction[1], direction[0]])
        axes.text(
            *(middle + 0.04 * size * left),
            str(row),
            color=color,
            fontsize=8,
            ha="center",
            va="center",
        )

    for point in points:
        axes.plot(*point, "o", color="tab:red", ms=7.0, mfc="none", mew=1.5)
    axes.set_aspect("equal")


def _validate_pieces(table: NDArray[np.float64]) -> None:
    """Check that a table of pieces is a closed boundary that runs in the right direction."""
    if table.ndim != 2 or table.shape[0] == 0 or table.shape[1] != new_pieces(1).shape[1]:
        raise ValueError("the table of pieces has the wrong shape")

    problem = _pieces_problem(table)
    if problem is not None:
        message, rows, points = problem
        raise _invalid_boundary(
            message, lambda axes: _draw_pieces(axes, table, rows, points)
        )


def _rebuild_billiard(
    constructor: str | None,
    arguments: Tuple[Any, ...],
    options: Dict[str, Any],
    parameters: NDArray[np.float64] | None,
    frequency_map: Callable | None,
) -> "Billiard":
    """Build a billiard again from what defines it, after it was pickled."""
    build = Billiard if constructor is None else getattr(Billiard, constructor)
    billiard = build(*arguments, **options)
    if parameters is not None and parameters.size > 0:
        billiard.set_parameters(parameters)
    if frequency_map is not None:
        billiard.set_frequency_map(frequency_map)

    return billiard


class Billiard(DiscreteDynamicalSystem):
    """
    Billiard, as the map from one collision with the boundary to the next.

    It inherits from `DiscreteDynamicalSystem`, so all the methods of that class work on
    a billiard, with these exceptions:

    - `classify_orbit` and `rotation_number` need a frequency map, and a billiard does
      not have one unless you set it with `set_frequency_map`. Without it,
      `classify_orbit` raises an error, and `rotation_number` falls back on its
      deprecated computation with a `DeprecationWarning`.
    - `available_models` lists the models of `DiscreteDynamicalSystem`. Use
      `available_billiards` for the billiards.
    - `diffusion_coefficient`, `mean_squared_displacement`, and `root_mean_squared` run,
      but the position and the angle are bounded, so their results do not describe
      diffusion.

    The available billiards, with their parameters in order, are:

    - `"oval"`: `R(theta) = 1 + eps cos(p theta)`, with `eps` in `[0, 1]` and an integer
      `p`. The position is `theta`. The cardioid is `eps = 1` and `p = 1`, which has a
      cusp at `theta = pi`, where a reflection is not defined, as at a corner. With
      `eps = 1` and `p > 1`, the boundary meets the origin at `p` points, which
      separate `p` closed lobes, and a trajectory stays in the lobe where it starts.
    - `"stadium"`: semicircles of radius `r` joined by straight segments of length
      `l`. The position is the arc length, from the start of the right semicircle.
    - `"sinai"`: a disk of radius `r` centered at `(x_d, y_d)` in the unit square with
      reflecting walls. The position is the arc length, along the walls from the origin
      and then around the disk from its rightmost point.
    - `"mushroom"`: a semicircular hat of radius `R` on a centered stem of width `w`
      and height `h`. The position is the arc length, from the bottom left of the stem.

    Other billiards can be defined with `Billiard.from_polar` and `Billiard.from_pieces`.

    Lines and arcs meet at corners, where a reflection is not defined. A trajectory
    reaches one exactly with probability zero, and takes the reflection of a neighboring
    piece.

    By default, the state is `(q, alpha)`, where `q` locates the collision on the
    boundary and `alpha`, in `(0, pi)`, is the angle of the outgoing trajectory,
    measured counterclockwise from the tangent of the boundary. This map preserves
    the measure `sin(alpha) ds dalpha`, where `s` is the arc length along the
    boundary, and not the area in `(q, alpha)`.

    The Birkhoff coordinates `(s, cos(alpha))` are canonical, so the map preserves
    area in them.

    Parameters
    ----------
    name : str
        Name of the billiard, one of `Billiard.available_billiards()`.
    parameters : numeric_like_t | None, optional
        Parameters of the billiard. They can also be set with `set_parameters`
        or passed to each method.
    solver : str
        Method that finds the next collision, one of
        `Billiard.available_solvers(name)`. It is required, as the right one depends on
        the boundary and the problem. A solver that is not valid for the parameters
        raises a `ValueError` when the map is used.

        - `"md"` is molecular dynamics: the particle advances in small steps until it
          is outside the billiard, and the collision is located by bisection. It works
          on every billiard, but it is the slowest, and a large step can miss a
          collision, so `md_steps` must suit the boundary.
        - `"exact"` finds the collision in closed form, on boundaries made of lines and
          arcs, and on the oval billiard with `p = 1`, including the cardioid.
        - `"circle"` is the tangent method from a seed on a circle around the billiard
          (Algorithm 1 of da Costa et al., 2022). It works only on a convex `R(theta)`,
          such as the oval billiard with `eps <= 1 / (1 + p^2)`.
        - `"polygon"` is the tangent method from a seed on a polygon around one wedge of
          the billiard (Algorithm 2 of da Costa et al., 2022). It uses the formula of
          the oval billiard, and requires `p >= 2` and `1 / (1 + p^2) < eps < 1`.

        By kind of boundary:

        - Lines and arcs (stadium, Sinai, mushroom): `"exact"`.
        - A convex `R(theta)`: `"circle"`.
        - A general `R(theta)` that is not convex: `"md"`.
        - The oval billiard: `"exact"` if `p = 1`, `"circle"` up to
          `eps = 1 / (1 + p^2)`, `"polygon"` above it, and `"md"` if `eps = 1` and
          `p > 1`.

        By kind of problem:

        - Long orbits, ensembles, and basins: the fastest valid solver above. `"md"` is
          about 2 times slower on lines and arcs, and 6 to 20 times slower on `R(theta)`.
        - A scan of parameters across `eps = 1 / (1 + p^2)`: `"circle"` raises past it, so
          use `"md"`, or `"polygon"` on that side.
        - Checking a new boundary or a result: compare with `"md"` and a large
          `md_steps`.

        If `"circle"` or `"polygon"` does not converge, as for a trajectory that almost
        touches the boundary, that collision is found by molecular dynamics with a fine
        step on the oval billiard, and raises a `ValueError` on a user-defined boundary.
    coordinates : str, optional
        Coordinates of the state. Default is `"angles"`, the state `(q, alpha)`.
        With `"birkhoff"`, the state is `(s, cos(alpha))`, and positions and sizes
        of holes are arc lengths. When the arc length has no closed form, as in the
        oval billiard, each collision computes it by quadrature, which adds to the
        cost. Near grazing, `cos(alpha)` resolves `alpha` only to about `1e-8`.
    md_steps : int_t, optional
        Number of molecular dynamics steps across the diameter of the billiard, used
        only when the solver is molecular dynamics. Default is `500`. A collision and a return to the interior within one
        step are not detected, so a trajectory that barely touches a part of the
        boundary that bulges inward can pass through it. Larger values are safer
        and slower.

    Raises
    ------
    ValueError
        - If `name`, `solver`, or `coordinates` is not available.
        - If `md_steps` is not positive.
    TypeError
        If `md_steps` is not an integer.

    Notes
    -----
    Each map is compiled by Numba the first time it is used with a given billiard,
    coordinates, and solver, and, when the solver is molecular dynamics, `md_steps`.

    Examples
    --------
    >>> Billiard.available_solvers("oval")
    ['md', 'exact', 'circle', 'polygon']
    >>> bl = Billiard("oval", parameters=[0.08, 3], solver="circle")

    A hole in the boundary is an interval of the position, whatever the value of
    `alpha`:

    >>> index, time = bl.escape_analysis(
    ...     [2.0, 1.0], 10_000, exits=[0.2, 0.0], hole_size=[0.2, np.inf]
    ... )

    In Birkhoff coordinates, the state is the arc length and `cos(alpha)`:

    >>> bl = Billiard("oval", parameters=[0.08, 3], solver="circle", coordinates="birkhoff")
    >>> s, cos_alpha = bl.trajectory([2.0, 0.5], 1)[0]
    """

    def __init__(
        self,
        name: str,
        parameters: numeric_like_t | None = None,
        *,
        solver: str,
        coordinates: str = "angles",
        md_steps: int_t = 500,
    ) -> None:
        if not isinstance(name, tuple) and name not in _BILLIARDS:
            raise ValueError(
                f"Billiard '{name}' not implemented. "
                f"Available billiards: {self.available_billiards()}"
            )
        billiard = _find_billiard(name)
        if solver not in billiard["solvers"]:
            raise ValueError(
                f"Solver '{solver}' not available for the {billiard['name']} billiard. "
                f"Available solvers: {billiard['solvers']}."
                f"{billiard.get('solver_hint', '')}"
            )
        if coordinates not in ("angles", "birkhoff"):
            raise ValueError("coordinates must be either 'angles' or 'birkhoff'")
        validate_positive(md_steps, "md_steps", Integral)

        self.__name = billiard["name"]
        self.__billiard = billiard
        self.__recipe: Tuple[str | None, Tuple[Any, ...], Dict[str, Any]] = (
            None,
            (name,),
            {},
        )
        self.__validated: set = set()
        self.__solver = solver
        self.__coordinates = coordinates
        self.__md_steps = int(md_steps)

        mapping, jacobian, backwards_mapping = _billiard_maps(
            name,
            coordinates,
            solver,
            self.__md_steps if solver == "md" else 0,
        )
        super().__init__(
            mapping=mapping,
            # Without one, the system differentiates the map numerically.
            jacobian=jacobian if billiard.get("analytic_jacobian", True) else None,
            backwards_mapping=backwards_mapping,
            system_dimension=2,
            number_of_parameters=len(self.__billiard["parameters"]),
        )

        if parameters is not None:
            self.set_parameters(parameters)
        elif not billiard["parameters"]:
            self.__validate_parameters(np.zeros(0))

    @classmethod
    def from_polar(
        cls,
        radius: Callable,
        dradius: Callable,
        ddradius: Callable | None = None,
        max_radius: Callable | None = None,
        check: Callable | None = None,
        number_of_parameters: int_t | None = None,
        parameters: numeric_like_t | None = None,
        *,
        solver: str,
        coordinates: str = "angles",
        md_steps: int_t = 500,
        num_panels: int_t = 32,
    ) -> "Billiard":
        """
        Define a billiard by its boundary in polar coordinates, `R(theta)`.

        The boundary must surround the origin and be crossed once by each ray from it.
        The position on the boundary is `theta`. Every function must be decorated with
        `@njit`.

        Parameters
        ----------
        radius : Callable
            Function `radius(theta, parameters)`, positive and with period `2 pi`.
        dradius : Callable
            Function `dradius(theta, parameters)`, the derivative of `radius`.
        ddradius : Callable | None, optional
            Function `ddradius(theta, parameters)`, the second derivative of `radius`,
            which gives the analytic Jacobian. If None, the map is differentiated
            numerically, which is less accurate and slower.
        max_radius : Callable | None, optional
            Function `max_radius(parameters)`, an upper bound of `radius`. It is needed by
            `solver="circle"`. If None, the size of the billiard is sampled at each
            collision.
        check : Callable | None, optional
            Function `check(parameters)` that raises if the parameters are invalid.
        number_of_parameters : int_t | None, optional
            Number of parameters. If None, it is the number in `parameters`, or zero.
        parameters : numeric_like_t | None, optional
            Parameters of the billiard.
        solver : str
            Required. `"md"` uses molecular dynamics, which works on every boundary, with
            `md_steps` chosen to suit it. `"circle"` is the tangent method of Algorithm 1
            of da Costa et al. (2022), much faster, but valid only on a convex
            boundary. It requires `max_radius`. Convexity is checked on a grid of 5000
            angles, for the given or stored parameters, which can miss a narrower dent,
            and a wrong solver can give wrong collisions without an error.
        coordinates : str, optional
            `"angles"` (default) or `"birkhoff"`, as in `Billiard`.
        md_steps : int_t, optional
            Number of molecular dynamics steps across the billiard, as in `Billiard`.
        num_panels : int_t, optional
            Number of Gauss-Legendre panels of the arc length, used by the Birkhoff
            coordinates. Default is `32`. A boundary with sharp features needs more, and
            a `UserWarning` tells when twice the panels changes the perimeter.

        Returns
        -------
        Billiard
            The billiard.

        Raises
        ------
        TypeError
            If a function is not decorated with `@njit`.
        ValueError
            - If `solver` is not available.
            - If, for the given or stored parameters, `radius` is not positive and
              periodic, a derivative does not match, `max_radius` is too small, or
              `solver="circle"` is used on a boundary that is not convex. Parameters
              passed to `trajectory` and similar methods are not checked this way. The
              boundary is drawn in a figure if the radius or the convexity is at fault.

        Examples
        --------
        >>> @njit
        ... def radius(theta, parameters):
        ...     return 1.0 + parameters[0] * np.cos(3.0 * theta)
        >>> @njit
        ... def dradius(theta, parameters):
        ...     return -3.0 * parameters[0] * np.sin(3.0 * theta)
        >>> bl = Billiard.from_polar(radius, dradius, parameters=[0.1], solver="md")
        """
        _require_jitted(radius, "radius")
        _require_jitted(dradius, "dradius")
        for function, label in (
            (ddradius, "ddradius"),
            (max_radius, "max_radius"),
            (check, "check"),
        ):
            if function is not None:
                _require_jitted(function, label)
        validate_positive(num_panels, "num_panels", Integral)

        count = cls.__parameter_count(number_of_parameters, parameters)
        key = (
            "polar",
            radius,
            dradius,
            ddradius,
            max_radius,
            check,
            count,
            int(num_panels),
        )
        if key not in _USER_BILLIARDS:
            geometry = make_polar_geometry(
                radius, dradius, ddradius, max_radius, int(num_panels)
            )
            solvers = ["md"]
            solver_functions = {}
            if max_radius is not None:
                solvers.append("circle")
                solver_functions["circle"] = geometry["circle_solver"]

            def validate(
                values: NDArray[np.float64], chosen: str, chosen_coordinates: str
            ) -> None:
                _validate_polar_functions(
                    radius,
                    dradius,
                    ddradius,
                    max_radius,
                    chosen,
                    geometry["convexity"],
                    values,
                )
                if chosen_coordinates == "birkhoff":
                    _warn_if_arc_length_is_coarse(
                        radius,
                        dradius,
                        int(num_panels),
                        geometry["perimeter"](values),
                        values,
                    )

            _USER_BILLIARDS[key] = {
                "name": "polar",
                "description": "User-defined billiard, R(theta)",
                "parameters": [f"parameter {j}" for j in range(count)],
                "position": "theta",
                "position_range": (0.0, 2.0 * np.pi),
                "period": 2.0 * np.pi,
                "solvers": solvers,
                "solver_hint": (
                    "" if max_radius is not None else " 'circle' requires max_radius."
                ),
                "solver_functions": solver_functions,
                "analytic_jacobian": ddradius is not None,
                "check": check if check is not None else no_check,
                "validate": validate,
                **geometry,
            }

        billiard = cls(
            key,
            parameters=parameters,
            solver=solver,
            coordinates=coordinates,
            md_steps=md_steps,
        )
        billiard.__recipe = (
            "from_polar",
            (radius, dradius),
            {
                "ddradius": ddradius,
                "max_radius": max_radius,
                "check": check,
                "number_of_parameters": count,
                "num_panels": int(num_panels),
            },
        )

        return billiard

    @classmethod
    def from_pieces(
        cls,
        pieces: Callable | Sequence[Sequence[Any]],
        inside: Callable | None = None,
        check: Callable | None = None,
        number_of_parameters: int_t | None = None,
        parameters: numeric_like_t | None = None,
        *,
        solver: str,
        coordinates: str = "angles",
        md_steps: int_t = 500,
    ) -> "Billiard":
        """
        Define a billiard by a boundary made of lines and circular arcs.

        The position on the boundary is the arc length, in the order of the pieces. The
        direction of the pieces matters: the outer boundary must run counterclockwise, and
        the boundary of a hole, such as a disk, clockwise.

        Parameters
        ----------
        pieces : Callable | Sequence[Sequence[Any]]
            The pieces, as a sequence of `("line", x0, y0, x1, y1)` and
            `("arc", center_x, center_y, radius, first_angle, span)`, where a negative
            `span` runs clockwise. For a boundary that depends on parameters, a function
            `pieces(parameters)` decorated with `@njit`, which returns a table built with
            `new_pieces`, `set_line`, and `set_arc` from
            `pynamicalsys.discrete_time.billiards`.
        inside : Callable | None, optional
            Function `inside(x, y, t, parameters)` decorated with `@njit`, that tells
            if a point is inside the billiard, only used by molecular dynamics. If None,
            it comes from the pieces, which is slower than a formula for the shape.
        check : Callable | None, optional
            Function `check(parameters)` decorated with `@njit`, that raises if the
            parameters are invalid.
        number_of_parameters : int_t | None, optional
            Number of parameters. If None, it is the number in `parameters`, or zero.
        parameters : numeric_like_t | None, optional
            Parameters of the billiard.
        solver : str
            Required. `"exact"` finds the collision in closed form, and is the choice for
            these boundaries. `"md"` uses molecular dynamics, which is slower and can miss
            a collision near a corner, but is useful to compare with.
        coordinates : str, optional
            `"angles"` (default) or `"birkhoff"`, as in `Billiard`.
        md_steps : int_t, optional
            Number of molecular dynamics steps across the billiard, as in `Billiard`.

        Returns
        -------
        Billiard
            The billiard.

        Raises
        ------
        TypeError
            If `inside`, `check`, or a function `pieces` is not decorated with `@njit`,
            or a piece has values that are not numbers.
        ValueError
            - If `solver` is not available.
            - If a piece is malformed or has no length or radius.
            - If, for the given or stored parameters, the pieces are not connected,
              cross or touch each other, run in the wrong direction, or a hole
              is outside the billiard or inside another hole. The boundary is drawn in
              a figure, as with `plot_pieces`, with the problem marked in red.

        Examples
        --------
        A square with a corner rounded off:

        >>> bl = Billiard.from_pieces([
        ...     ("line", 0.0, 0.0, 1.0, 0.0),
        ...     ("arc", 1.0, 1.0, 1.0, -np.pi / 2.0, np.pi / 2.0),
        ...     ("line", 2.0, 1.0, 0.0, 1.0),
        ...     ("line", 0.0, 1.0, 0.0, 0.0),
        ... ], solver="exact")
        >>> bl.perimeter() == 1.0 + np.pi / 2.0 + 2.0 + 1.0
        np.True_
        """
        for function, label in ((inside, "inside"), (check, "check")):
            if function is not None:
                _require_jitted(function, label)

        count = cls.__parameter_count(number_of_parameters, parameters)
        if callable(pieces):
            _require_jitted(pieces, "pieces")
            pieces_function = pieces
            definition: Any = pieces
        else:
            table = _fixed_pieces_table(pieces)
            _validate_pieces(table)
            pieces_function = None
            definition = table.tobytes()

        key = ("pieces", definition, inside, check, count)
        if key not in _USER_BILLIARDS:
            if pieces_function is None:
                pieces_function = make_fixed_pieces(table)
            function = pieces_function

            def validate(
                values: NDArray[np.float64], chosen: str, chosen_coordinates: str
            ) -> None:
                _validate_pieces(function(values))

            def closed(values: NDArray[np.float64]) -> bool:
                return len(_pieces_components(function(values))) == 1

            defaults = make_pieces_inside_and_size(function)
            _USER_BILLIARDS[key] = {
                "name": "pieces",
                "description": "User-defined billiard, made of lines and arcs",
                "parameters": [f"parameter {j}" for j in range(count)],
                "position": "s",
                "closed": closed,
                "solvers": ["md", "exact"],
                "check": check if check is not None else no_check,
                "validate": validate,
                **make_piecewise_geometry(function),
                "diameter": defaults["diameter"],
                "inside": inside if inside is not None else defaults["inside"],
            }

        billiard = cls(
            key,
            parameters=parameters,
            solver=solver,
            coordinates=coordinates,
            md_steps=md_steps,
        )
        billiard.__recipe = (
            "from_pieces",
            (pieces,),
            {"inside": inside, "check": check, "number_of_parameters": count},
        )

        return billiard

    @staticmethod
    def plot_pieces(
        pieces: Callable | Sequence[Sequence[Any]],
        parameters: numeric_like_t | None = None,
        ax: Any | None = None,
    ) -> Any:
        """
        Draw a boundary made of lines and arcs, whether it is valid or not.

        Each piece is numbered, and has an arrow with its direction. The number lies on the
        inner side of the billiard when the piece runs in the right direction.

        Parameters
        ----------
        pieces : Callable | Sequence[Sequence[Any]]
            The pieces, as in `from_pieces`.
        parameters : numeric_like_t | None, optional
            Parameters of a function `pieces`.
        ax : matplotlib.axes.Axes | None, optional
            Axes to draw on. If None, a new figure is created.

        Returns
        -------
        matplotlib.axes.Axes
            The axes with the drawing.

        Examples
        --------
        >>> ax = Billiard.plot_pieces([
        ...     ("line", 0.0, 0.0, 1.0, 1.0),
        ...     ("line", 1.0, 1.0, 1.0, 0.0),
        ...     ("line", 1.0, 0.0, 0.0, 1.0),
        ...     ("line", 0.0, 1.0, 0.0, 0.0),
        ... ])
        """
        import matplotlib.pyplot as plt

        if callable(pieces):
            if parameters is None:
                raise ValueError("a function pieces requires parameters")
            table = pieces(np.atleast_1d(np.asarray(parameters, dtype=np.float64)))
        else:
            table = _fixed_pieces_table(pieces)

        if ax is None:
            _, ax = plt.subplots()
        _draw_pieces(ax, table)

        return ax

    def __reduce__(self) -> Tuple[Any, ...]:
        # Compiled maps are slow to pickle, which would dominate parallel runs with
        # one task for each trajectory. What defines the billiard is small, and each
        # process compiles the maps only once.
        constructor, arguments, options = self.__recipe

        return (
            _rebuild_billiard,
            (
                constructor,
                arguments,
                {
                    **options,
                    "solver": self.__solver,
                    "coordinates": self.__coordinates,
                    "md_steps": self.__md_steps,
                },
                self.get_parameters(),
                self._DiscreteDynamicalSystem__frequency_map,
            ),
        )

    @staticmethod
    def __parameter_count(
        number_of_parameters: int_t | None, parameters: numeric_like_t | None
    ) -> int:
        if number_of_parameters is not None:
            if isinstance(number_of_parameters, bool) or not isinstance(
                number_of_parameters, Integral
            ):
                raise TypeError("number_of_parameters must be an integer")
            if number_of_parameters < 0:
                raise ValueError("number_of_parameters must not be negative")
            return int(number_of_parameters)
        if parameters is not None:
            return int(np.atleast_1d(parameters).size)

        return 0

    def __validate_parameters(self, parameters: NDArray[np.float64]) -> None:
        # Sampling along the boundary is slow, so it is done once for each set of
        # parameters.
        if "validate" not in self.__billiard:
            return
        values = tuple(np.atleast_1d(parameters).tolist())
        if values not in self.__validated:
            self.__billiard["check"](parameters)
            self.__billiard["validate"](parameters, self.__solver, self.__coordinates)
            self.__validated.add(values)

    def set_parameters(self, parameters: numeric_like_t) -> None:
        """
        Set the parameters of the billiard.

        Parameters
        ----------
        parameters : numeric_like_t
            The parameters.

        Raises
        ------
        ValueError
            If the parameters do not match the number expected or, for a billiard
            defined by a user, the boundary is not valid for them.
        """
        super().set_parameters(parameters)
        self.__validate_parameters(self.get_parameters())

    @classmethod
    def available_billiards(cls) -> List[str]:
        """Return the names of the available billiards."""
        return list(_BILLIARDS.keys())

    @classmethod
    def available_solvers(cls, name: str) -> List[str]:
        """Return the solvers available for the billiard `name`."""
        if name not in _BILLIARDS:
            raise ValueError(
                f"Billiard '{name}' not implemented. "
                f"Available billiards: {cls.available_billiards()}"
            )
        return list(_BILLIARDS[name]["solvers"])

    @property
    def info(self) -> Dict[str, Any]:
        """
        Return a dictionary with information about the billiard.

        The entry `"period"` is the period of the position, the perimeter when the
        position is an arc length. It is None if the position does not wrap or if the
        perimeter is needed and no parameters are stored.
        """
        if self.__coordinates == "birkhoff":
            state = ["s", "cos(alpha)"]
        else:
            state = [self.__billiard["position"], "alpha"]

        stored = not self.__billiard["parameters"] or self.get_parameters() is not None
        if stored or (
            self.__coordinates == "angles" and self.__billiard["position"] == "theta"
        ):
            period = self.__period(None)
        else:
            period = None
        if period == np.inf:
            period = None

        return {
            "name": self.__name,
            "description": self.__billiard["description"],
            "parameters": list(self.__billiard["parameters"]),
            "state": state,
            "period": period,
            "available_solvers": list(self.__billiard["solvers"]),
            "solver": self.__solver,
            "coordinates": self.__coordinates,
            "md_steps": self.__md_steps,
        }

    def __resolve_parameters(
        self, parameters: numeric_like_t | None
    ) -> NDArray[np.float64]:
        if parameters is None:
            parameters = self.get_parameters()
        if not self.__billiard["parameters"]:
            parameters_arr = np.zeros(0)
        else:
            parameters_arr = validate_parameters(
                parameters, len(self.__billiard["parameters"])
            )
        self.__billiard["check"](parameters_arr)
        self.__validate_parameters(parameters_arr)

        return parameters_arr

    def __period(self, parameters: numeric_like_t | None) -> float:
        closed = self.__billiard.get("closed", True)
        if callable(closed):
            closed = closed(self.__resolve_parameters(parameters))
        if not closed:
            return np.inf
        if self.__coordinates == "birkhoff" or self.__billiard["position"] == "s":
            return self.perimeter(parameters)

        return self.__billiard["period"]

    def perimeter(self, parameters: numeric_like_t | None = None) -> float:
        """
        Return the length of the boundary of the billiard.

        Parameters
        ----------
        parameters : numeric_like_t | None, optional
            Parameters of the billiard. If None, the stored parameters are used.

        Returns
        -------
        float
            The perimeter.

        Raises
        ------
        ValueError
            If the parameters are invalid.
        """
        return float(self.__billiard["perimeter"](self.__resolve_parameters(parameters)))

    def boundary(
        self, num_points: int_t = 1000, parameters: numeric_like_t | None = None
    ) -> NDArray[np.float64]:
        """
        Return points along the boundary of the billiard, for example to plot it.

        Parameters
        ----------
        num_points : int_t, optional
            Number of points. Default is `1000`.
        parameters : numeric_like_t | None, optional
            Parameters of the billiard. If None, the stored parameters are used.

        Returns
        -------
        NDArray[np.float64]
            Points of shape `(n, 2)`, in order along the boundary, so the first and
            last points coincide on a closed boundary. Here `n` is `num_points`,
            except on a boundary made of lines and arcs, where each piece gets a
            number of points close to its share of `num_points`, including its
            ends. Separate parts of a boundary are separated by a row of NaN, which
            plots as a gap.

        Raises
        ------
        ValueError
            - If `num_points` is not positive.
            - If the parameters are invalid.
        TypeError
            If `num_points` is not an integer.
        """
        validate_positive(num_points, "num_points", Integral)
        parameters_arr = self.__resolve_parameters(parameters)
        if "pieces" in self.__billiard:
            return self.__piecewise_boundary(int(num_points), parameters_arr)

        lower, upper = self.__billiard["position_range"]
        positions = np.linspace(lower, upper, int(num_points))
        point = self.__billiard["boundary_point"]

        return np.array([point(q, 0.0, parameters_arr) for q in positions])

    def __piecewise_boundary(
        self, num_points: int, parameters: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        table = self.__billiard["pieces"](parameters)
        total = np.sum(table[:, 2])
        points = []
        for row in range(table.shape[0]):
            length = table[row, 2]
            if length == 0.0:
                continue
            count = max(2, int(round(num_points * length / total)))
            part = [piece_point(table, row, u) for u in np.linspace(0.0, length, count)]
            if points and not np.allclose(points[-1], part[0]):
                points.append((np.nan, np.nan))
            points.extend(part)

        return np.array(points)

    def to_cartesian(
        self,
        states: NDArray[np.float64] | Sequence[float] | Sequence[Sequence[float]],
        parameters: numeric_like_t | None = None,
    ) -> NDArray[np.float64]:
        """
        Convert collisions to their positions in the plane of the billiard.

        Parameters
        ----------
        states : NDArray[np.float64] | Sequence[float] | Sequence[Sequence[float]]
            One state of shape `(2,)`, or states of shape `(n, 2)`, such as the
            output of `trajectory`.
        parameters : numeric_like_t | None, optional
            Parameters of the billiard. If None, the stored parameters are used.

        Returns
        -------
        NDArray[np.float64]
            Positions `(x, y)`, with the same shape as `states`.

        Raises
        ------
        ValueError
            - If `states` has an invalid shape.
            - If the parameters are invalid.
        """
        states_arr = np.asarray(states, dtype=np.float64)
        if states_arr.ndim not in (1, 2) or states_arr.shape[-1] != 2:
            raise ValueError("states must have shape (2,) or (n, 2)")
        parameters_arr = self.__resolve_parameters(parameters)
        point = self.__billiard["boundary_point"]
        arc_position = self.__billiard["arc_position"]

        def position(first: float) -> Tuple[float, float]:
            if self.__coordinates == "birkhoff":
                first = arc_position(first, 0.0, parameters_arr)
            return point(first, 0.0, parameters_arr)

        if states_arr.ndim == 1:
            return np.array(position(states_arr[0]))

        return np.array([position(first) for first in states_arr[:, 0]])

    def escape_analysis(
        self,
        u: numeric_like_t,
        max_time: int_t,
        exits: NDArray[np.float64]
        | Sequence[Sequence[float]]
        | Sequence[NDArray[np.float64]],
        parameters: numeric_like_t | None = None,
        escape: str = "entering",
        hole_size: numeric_like_t | None = None,
        periods: numeric_like_t | None = None,
    ) -> Tuple[int, int_t]:
        """
        Compute the escape index and escape time for a single trajectory.

        This is `DiscreteDynamicalSystem.escape_analysis`, except that `periods`
        defaults to the period of the boundary for the position and to no
        wrapping for the other coordinate. When the position is an arc length, the
        period is the perimeter, and the centers and sizes of the holes are arc
        lengths. The position of the Sinai billiard does not wrap, since its boundary
        has two separate parts, so a hole across the corner at `s = 0` takes two holes.

        A hole in the boundary is an interval of the position, whatever the value
        of the other coordinate: `hole_size=[h, np.inf]`. See
        `DiscreteDynamicalSystem.escape_analysis` for the parameters.

        Returns
        -------
        Tuple[int, int]
            `(exit_index, escape_time)` or, if `escape == "exiting"`,
            `(face_index, escape_time)`.
        """
        if periods is None and escape == "entering":
            periods = [self.__period(parameters), np.inf]

        return super().escape_analysis(
            u,
            max_time,
            exits,
            parameters=parameters,
            escape=escape,
            hole_size=hole_size,
            periods=periods,
        )
