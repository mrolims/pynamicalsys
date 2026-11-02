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
    oval_arc_speed,
    oval_boundary_point,
    oval_check,
    oval_curvature,
    oval_diameter,
    oval_inside,
    oval_project,
    oval_tangent,
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
    },
}


@lru_cache(maxsize=None)
def _angle_maps(name: str, md_steps: int) -> Tuple[Callable, Callable, Callable]:
    # Each map is compiled once for every billiard and number of steps.
    billiard = _BILLIARDS[name]

    return make_angle_maps(
        billiard["check"],
        billiard["diameter"],
        billiard["inside"],
        billiard["boundary_point"],
        billiard["tangent"],
        billiard["curvature"],
        billiard["arc_speed"],
        billiard["project"],
        md_steps,
    )


class Billiard(DiscreteDynamicalSystem):
    """
    Predefined billiard, as the map from one collision with the boundary to the next.

    A billiard is a discrete dynamical system, so every method of
    `DiscreteDynamicalSystem` is available, such as `trajectory`, `lyapunov`,
    and `escape_analysis`.

    The state is `(q, alpha)`, where `q` locates the collision on the boundary
    and `alpha`, in `(0, pi)`, is the angle of the outgoing trajectory, measured
    counterclockwise from the tangent of the boundary.

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
        billiard, and the collision is located by bisection.
    coordinates : str, optional
        Coordinates of the state. Default is `"angles"`, the state `(q, alpha)`.
    md_steps : int_t, optional
        Number of molecular dynamics steps across the diameter of the billiard.
        Default is `500`. A collision and a return to the interior within one
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
    Each map is compiled by Numba the first time it is used with a given billiard
    and `md_steps`.

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
        if coordinates != "angles":
            raise ValueError("coordinates must be 'angles'")
        validate_positive(md_steps, "md_steps", Integral)

        self.__name = name
        self.__billiard = _BILLIARDS[name]
        self.__solver = solver
        self.__coordinates = coordinates
        self.__md_steps = int(md_steps)

        mapping, jacobian, backwards_mapping = _angle_maps(name, self.__md_steps)
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
        """Return a dictionary with information about the billiard."""
        return {
            "name": self.__name,
            "description": self.__billiard["description"],
            "parameters": list(self.__billiard["parameters"]),
            "state": [self.__billiard["position"], "alpha"],
            "period": self.__billiard["period"],
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
            Points of shape `(num_points, 2)`, from one end of the position range
            to the other, so the first and last points coincide on a closed
            boundary.

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
        lower, upper = self.__billiard["position_range"]
        positions = np.linspace(lower, upper, int(num_points))
        point = self.__billiard["boundary_point"]

        return np.array([point(q, 0.0, parameters_arr) for q in positions])

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

        if states_arr.ndim == 1:
            return np.array(point(states_arr[0], 0.0, parameters_arr))

        return np.array([point(q, 0.0, parameters_arr) for q in states_arr[:, 0]])

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
        wrapping for the other coordinate.

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
            periods = [self.__billiard["period"], np.inf]

        return super().escape_analysis(
            u,
            max_time,
            exits,
            parameters=parameters,
            escape=escape,
            hole_size=hole_size,
            periods=periods,
        )
