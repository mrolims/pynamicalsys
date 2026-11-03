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
    make_angle_maps,
    make_birkhoff_maps,
    make_fixed_pieces,
    make_md_solver,
    make_oval_auto_solver,
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
    oval_auto_choice,
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
        "solvers": ["auto", "md", "exact", "circle", "polygon"],
        "solver_functions": {
            "exact": oval_exact_solver,
            "circle": oval_circle_solver,
            "polygon": oval_polygon_solver,
        },
        "make_auto_solver": make_oval_auto_solver,
        "auto_choice": oval_auto_choice,
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
        "solvers": ["auto", "md", "exact"],
        "auto_solver": "exact",
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
        "solvers": ["auto", "md", "exact"],
        "auto_solver": "exact",
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
        "solvers": ["auto", "md", "exact"],
        "auto_solver": "exact",
        "check": mushroom_check,
        "diameter": mushroom_diameter,
        "inside": mushroom_inside,
        **make_piecewise_geometry(mushroom_pieces),
    },
}


# Billiards defined by users, by the functions or pieces that define them, so that the
# same definition is compiled only once.
_USER_BILLIARDS: Dict[Tuple[Any, ...], Dict[str, Any]] = {}


def _find_billiard(name: Any) -> Dict[str, Any]:
    return _USER_BILLIARDS[name] if isinstance(name, tuple) else _BILLIARDS[name]


@lru_cache(maxsize=None)
def _collision_solver(name: Any, solver: str, md_steps: int) -> Callable:
    billiard = _find_billiard(name)
    if solver == "auto":
        if "make_auto_solver" in billiard:
            return billiard["make_auto_solver"](md_steps)
        solver = billiard["auto_solver"]
    if solver == "md":
        return make_md_solver(billiard["inside"], billiard["diameter"], md_steps)
    if "solver_functions" in billiard:
        return billiard["solver_functions"][solver]

    return billiard["exact_solver"]


def _uses_md_steps(name: Any, solver: str) -> bool:
    billiard = _find_billiard(name)
    if solver == "auto":
        return "make_auto_solver" in billiard or billiard["auto_solver"] == "md"

    return solver == "md"


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
    parameters: NDArray[np.float64],
) -> None:
    """Check the functions of a polar boundary along it, for the given parameters."""
    h = 1e-5
    thetas = np.linspace(0.0, 2.0 * np.pi, 257)
    r = np.array([radius(theta, parameters) for theta in thetas])
    dr = np.array([dradius(theta, parameters) for theta in thetas])

    if np.any(r <= 0.0):
        raise ValueError("radius must be positive along the boundary")
    if abs(r[0] - r[-1]) > 1e-9 * abs(r[0]):
        raise ValueError("radius must have period 2 pi")

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
    else:
        ddr = np.array(
            [
                (dradius(theta + h, parameters) - dradius(theta - h, parameters))
                / (2.0 * h)
                for theta in thetas
            ]
        )

    if max_radius is not None and max_radius(parameters) < np.max(r) * (1.0 - 1e-12):
        raise ValueError("max_radius is smaller than the radius along the boundary")

    if solver == "circle" and np.any(r * r + 2.0 * dr * dr - r * ddr < -1e-9):
        raise ValueError("solver='circle' requires a convex boundary")


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
        if np.hypot(end[0] - start[0], end[1] - start[1]) > 1e-9:
            components.append((first, row))
            first = row + 1
    components.append((first, table.shape[0] - 1))

    return components


def _validate_pieces(table: NDArray[np.float64]) -> None:
    """Check that a table of pieces is a boundary with the billiard on its left."""
    if table.ndim != 2 or table.shape[0] == 0 or table.shape[1] != new_pieces(1).shape[1]:
        raise ValueError("the table of pieces has the wrong shape")

    for row in range(table.shape[0]):
        if table[row, 0] == 1.0 and table[row, 5] <= 0.0:
            raise ValueError(f"piece {row} must have a positive radius")
        if table[row, 2] <= 0.0:
            raise ValueError(f"piece {row} must have a positive length")
        if table[row, 0] == 1.0 and table[row, 2] / table[row, 5] > 2.0 * np.pi * (
            1.0 + 1e-12
        ):
            raise ValueError(f"arc {row} must sweep at most 2 pi")

    areas = []
    for first, last in _pieces_components(table):
        end = piece_point(table, last, table[last, 2])
        start = piece_point(table, first, 0.0)
        if np.hypot(end[0] - start[0], end[1] - start[1]) > 1e-9:
            raise ValueError(
                "the pieces must be connected: each part of the boundary must end "
                "where it starts"
            )

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
    if areas[outer] <= 0.0 or any(
        area >= 0.0 for j, area in enumerate(areas) if j != outer
    ):
        raise ValueError(
            "the billiard must be on the left of each piece: the outer boundary runs "
            "counterclockwise, and the boundary of a hole runs clockwise"
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

    A billiard is a discrete dynamical system, so every method of
    `DiscreteDynamicalSystem` is available, such as `trajectory`, `lyapunov`,
    and `escape_analysis`.

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
    solver : str, optional
        Method that finds the next collision, one of
        `Billiard.available_solvers(name)`. A solver that is not valid for the
        parameters raises a `ValueError` when the map is used.

        - `"auto"` (default) uses the fastest valid solver for the parameters of each
          call. `info["active_solver"]` tells which one for the stored parameters.
        - `"md"` is molecular dynamics: the particle advances in small steps until it
          is outside the billiard, and the collision is located by bisection. It works
          on every billiard, but it is the slowest, and a large step can miss a
          collision.
        - `"exact"` finds the collision in closed form. It is available on boundaries
          made of lines and arcs, and on the oval billiard with `p = 1`.
        - `"circle"`, on the oval billiard with `eps <= 1 / (1 + p^2)`, where the
          curvature of the boundary does not change sign, is the tangent method from a
          seed on a circle around the billiard (Algorithm 1 of da Costa et al., 2022).
        - `"polygon"`, on the oval billiard with `p >= 2` and
          `1 / (1 + p^2) < eps < 1`, is the tangent method from a seed on a polygon around one wedge of the billiard
          (Algorithm 2 of da Costa et al., 2022).

        If `"circle"` or `"polygon"` does not converge, as for a trajectory that almost
        touches the boundary, that collision is found by molecular dynamics with a fine
        step.
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
    coordinates, and solver, and, when the solver can be molecular dynamics, `md_steps`.

    Examples
    --------
    >>> bl = Billiard("oval", parameters=[0.08, 3])
    >>> bl.available_solvers("oval")
    ['auto', 'md', 'exact', 'circle', 'polygon']
    >>> bl.info["active_solver"]
    'circle'

    A hole in the boundary is an interval of the position, whatever the value of
    `alpha`:

    >>> index, time = bl.escape_analysis(
    ...     [2.0, 1.0], 10_000, exits=[0.2, 0.0], hole_size=[0.2, np.inf]
    ... )

    In Birkhoff coordinates, the state is the arc length and `cos(alpha)`:

    >>> bl = Billiard("oval", parameters=[0.08, 3], coordinates="birkhoff")
    >>> s, cos_alpha = bl.trajectory([2.0, 0.5], 1)[0]
    """

    def __init__(
        self,
        name: str,
        parameters: numeric_like_t | None = None,
        solver: str = "auto",
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
                f"Available solvers: {billiard['solvers']}"
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
            self.__md_steps if _uses_md_steps(name, solver) else 0,
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
        solver: str = "auto",
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
        solver : str, optional
            `"auto"` (default) and `"md"` use molecular dynamics. `"circle"` is the tangent
            method of Algorithm 1 of da Costa et al. (2022), which is faster, but valid
            only on a convex boundary. It requires `max_radius`.
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
              passed to each method are not checked this way.

        Examples
        --------
        >>> @njit
        ... def radius(theta, parameters):
        ...     return 1.0 + parameters[0] * np.cos(3.0 * theta)
        >>> @njit
        ... def dradius(theta, parameters):
        ...     return -3.0 * parameters[0] * np.sin(3.0 * theta)
        >>> bl = Billiard.from_polar(radius, dradius, parameters=[0.1])
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
            solvers = ["auto", "md"]
            solver_functions = {}
            if max_radius is not None:
                solvers.append("circle")
                solver_functions["circle"] = geometry["circle_solver"]

            def validate(
                values: NDArray[np.float64], chosen: str, chosen_coordinates: str
            ) -> None:
                _validate_polar_functions(
                    radius, dradius, ddradius, max_radius, chosen, values
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
                "auto_solver": "md",
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
        solver: str = "auto",
        coordinates: str = "angles",
        md_steps: int_t = 500,
    ) -> "Billiard":
        """
        Define a billiard by a boundary made of lines and circular arcs.

        The position on the boundary is the arc length, in the order of the pieces, and
        the billiard must be on the left of each piece. So the outer boundary runs
        counterclockwise, and the boundary of a hole, such as a disk, runs clockwise.

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
        solver : str, optional
            `"auto"` (default) or `"exact"` find the collision in closed form. `"md"`
            uses molecular dynamics.
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
            - If, for the given or stored parameters, the pieces are not connected, or
              the billiard is not on their left.

        Examples
        --------
        A square with a corner rounded off:

        >>> bl = Billiard.from_pieces([
        ...     ("line", 0.0, 0.0, 1.0, 0.0),
        ...     ("arc", 1.0, 1.0, 1.0, -np.pi / 2.0, np.pi / 2.0),
        ...     ("line", 2.0, 1.0, 0.0, 1.0),
        ...     ("line", 0.0, 1.0, 0.0, 0.0),
        ... ])
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
                "solvers": ["auto", "md", "exact"],
                "auto_solver": "exact",
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

        The entry `"active_solver"` is the solver in use, which is None if the solver is
        `"auto"`, depends on the parameters, and none are stored. The entry `"period"`
        is the period of the position, the perimeter when the
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
            "active_solver": self.__active_solver(),
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

    def __active_solver(self) -> str | None:
        if self.__solver != "auto":
            return self.__solver
        if "auto_choice" not in self.__billiard:
            return self.__billiard["auto_solver"]
        parameters = self.get_parameters()

        return None if parameters is None else self.__billiard["auto_choice"](parameters)

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
