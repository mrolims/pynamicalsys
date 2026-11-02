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

from functools import lru_cache
from numbers import Integral
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from pynamicalsys.common.types import int_t, numeric_like_t, numeric_t
from pynamicalsys.common.validators import validate_parameters, validate_positive
from pynamicalsys.core.discrete_dynamical_systems import DiscreteDynamicalSystem
from pynamicalsys.discrete_time.billiards import (
    make_angle_maps,
    make_birkhoff_maps,
    make_md_solver,
    make_piecewise_geometry,
    mushroom_check,
    mushroom_diameter,
    mushroom_inside,
    mushroom_pieces,
    oval_arc_length,
    oval_arc_position,
    oval_arc_speed,
    oval_boundary_point,
    oval_check,
    oval_curvature,
    oval_diameter,
    oval_inside,
    oval_perimeter,
    oval_project,
    oval_tangent,
    piece_point,
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
        "description": "Oval billiard, R(theta) = 1 + eps cos(p theta)",
        "parameters": ["eps", "p"],
        "position": "theta",
        "position_range": (0.0, 2.0 * np.pi),
        "period": 2.0 * np.pi,
        "solvers": ["md"],
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


@lru_cache(maxsize=None)
def _collision_solver(name: str, solver: str, md_steps: int) -> Callable:
    billiard = _BILLIARDS[name]
    if solver == "exact":
        return billiard["exact_solver"]

    return make_md_solver(billiard["inside"], billiard["diameter"], md_steps)


@lru_cache(maxsize=None)
def _billiard_maps(
    name: str, coordinates: str, solver: str, md_steps: int
) -> Tuple[Callable, Callable, Callable]:
    # Each map is compiled once for every billiard, coordinates, and solver, and for
    # every number of steps with molecular dynamics.
    billiard = _BILLIARDS[name]
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


class Billiard(DiscreteDynamicalSystem):
    """
    Predefined billiard, as the map from one collision with the boundary to the next.

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
        `Billiard.available_solvers(name)`. Default is `"md"`, molecular
        dynamics: the particle advances in small steps until it is outside the
        billiard, and the collision is located by bisection. It works on every
        billiard. With `"exact"`, available on boundaries made of lines and arcs,
        the collision comes from the closed-form intersection of the trajectory with
        each piece, so no collision can be missed.
    coordinates : str, optional
        Coordinates of the state. Default is `"angles"`, the state `(q, alpha)`.
        With `"birkhoff"`, the state is `(s, cos(alpha))`, and positions and sizes
        of holes are arc lengths. When the arc length has no closed form, as in the
        oval billiard, each collision computes it by quadrature, which adds to the
        cost. Near grazing, `cos(alpha)` resolves `alpha` only to about `1e-8`.
    md_steps : int_t, optional
        Number of molecular dynamics steps across the diameter of the billiard, used
        only by `solver="md"`. Default is `500`. A collision and a return to the interior within one
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
    coordinates, and solver, and with `solver="md"`, `md_steps`.

    Examples
    --------
    >>> bl = Billiard("oval", parameters=[0.08, 3])
    >>> bl.available_solvers("oval")
    ['md']

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
        solver: str = "md",
        coordinates: str = "angles",
        md_steps: int_t = 500,
    ) -> None:
        if name not in _BILLIARDS:
            raise ValueError(
                f"Billiard '{name}' not implemented. "
                f"Available billiards: {self.available_billiards()}"
            )
        if solver not in _BILLIARDS[name]["solvers"]:
            raise ValueError(
                f"Solver '{solver}' not available for the {name} billiard. "
                f"Available solvers: {self.available_solvers(name)}"
            )
        if coordinates not in ("angles", "birkhoff"):
            raise ValueError("coordinates must be either 'angles' or 'birkhoff'")
        validate_positive(md_steps, "md_steps", Integral)

        self.__name = name
        self.__billiard = _BILLIARDS[name]
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
            jacobian=jacobian,
            backwards_mapping=backwards_mapping,
            system_dimension=2,
            number_of_parameters=len(self.__billiard["parameters"]),
        )

        if parameters is not None:
            self.set_parameters(parameters)

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

        if self.__wraps_with_perimeter() and self.get_parameters() is None:
            period = None
        else:
            period = self.__period(None)
        if period == np.inf:
            period = None

        return {
            "name": self.__name,
            "description": self.__billiard["description"],
            "parameters": list(self.__billiard["parameters"]),
            "state": state,
            "period": period,
            "solver": self.__solver,
            "coordinates": self.__coordinates,
            "md_steps": self.__md_steps,
        }

    def __resolve_parameters(
        self, parameters: numeric_like_t | None
    ) -> NDArray[np.float64]:
        if parameters is None:
            parameters = self.get_parameters()
        parameters_arr = validate_parameters(
            parameters, len(self.__billiard["parameters"])
        )
        self.__billiard["check"](parameters_arr)

        return parameters_arr

    def __wraps_with_perimeter(self) -> bool:
        return self.__billiard.get("closed", True) and (
            self.__coordinates == "birkhoff" or self.__billiard["position"] == "s"
        )

    def __period(self, parameters: numeric_like_t | None) -> float:
        if not self.__billiard.get("closed", True):
            return np.inf
        if self.__wraps_with_perimeter():
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
