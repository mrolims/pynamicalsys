# continuous_dynamical_systems.py

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
from numbers import Integral, Real
from typing import Any, Callable, Dict, List, Sequence, Union
from IPython.display import Math

import numpy as np
from numba.core.errors import NumbaExperimentalFeatureWarning
from numpy.typing import NDArray

from pynamicalsys.common.types import int_t, numeric_t, numeric_like_t

from pynamicalsys.continuous_time.step import evolve_system

from pynamicalsys.continuous_time.step_methods import (
    estimate_initial_step,
    rk4_step_wrapped,
    rk45_step_wrapped,
)

from pynamicalsys.continuous_time.trajectory import (
    generate_trajectory,
    ensemble_trajectories,
)

from pynamicalsys.continuous_time.poincare import (
    generate_poincare_section,
    ensemble_poincare_section,
)

from pynamicalsys.continuous_time.stroboscopic import (
    generate_stroboscopic_map,
    ensemble_stroboscopic_map,
)

from pynamicalsys.continuous_time.maxima_map import (
    generate_maxima_map,
    ensemble_maxima_map,
)

from pynamicalsys.continuous_time.basins import basin_of_attraction

from pynamicalsys.continuous_time.lyapunov import (
    lyapunov_exponents,
    maximum_lyapunov_exponent,
)

from pynamicalsys.continuous_time.clv import clv_angles, compute_clvs

from pynamicalsys.continuous_time.sali import sali

from pynamicalsys.continuous_time.ldi import ldi_k

from pynamicalsys.continuous_time.gali import gali_k

from pynamicalsys.continuous_time.rte import (
    recurrence_time_entropy as recurrence_time_entropy_core,
)

from pynamicalsys.continuous_time.hurst import hurst_exponent_wrapped

from pynamicalsys.continuous_time.models import (
    henon_heiles,
    henon_heiles_jacobian,
    lorenz_jacobian,
    lorenz_system,
    rossler_system,
    rossler_system_4D,
    rossler_system_4D_jacobian,
    rossler_system_jacobian,
    duffing,
    duffing_jacobian,
)


from pynamicalsys.continuous_time.validators import (
    validate_times,
)

from pynamicalsys.common.validators import (
    validate_initial_conditions,
    validate_parameters,
    validate_positive,
    validate_non_negative,
    validate_clv_pairs,
    validate_clv_subspaces,
)


class ContinuousDynamicalSystem:
    """
    Class for defining, integrating, and analyzing continuous-time dynamical systems.

    This class represents systems of ordinary differential equations of the form

        du/dt = f(t, u; parameters),

    where `u` is the state vector and `f` is the vector field. A system can be
    created either from one of the built-in models or from user-supplied
    equations of motion, with an optional Jacobian for tangent-space and
    stability computations.

    The class provides fixed-step and adaptive-step integration together with
    tools for trajectory generation and nonlinear-dynamics analysis, including
    Poincaré sections, stroboscopic maps, maxima maps, Lyapunov exponents,
    covariant Lyapunov vectors (CLVs), SALI, LDI, GALI, recurrence time
    entropy, Hurst exponent estimation, and basin-of-attraction analysis.

    Parameters
    ----------
    model : str or None, optional
        Name of a built-in continuous-time model.
    equations_of_motion : callable or None, optional
        User-defined vector field with signature
        `f(time, u, parameters) -> NDArray[np.float64]`.
    jacobian : callable or None, optional
        Jacobian of the vector field with signature
        `J(time, u, parameters) -> NDArray[np.float64]`.
    system_dimension : int or None, optional
        Dimension of the state space for a custom system.
    parameters : array_like or None, optional
        Parameter vector for the system.
    number_of_parameters : int or None, optional
        Number of parameters expected by the custom system.

    Notes
    -----
    - A system can be created either from a built-in model or from custom
      equations of motion, but not both at the same time.
    - The Jacobian is optional for trajectory-level analysis, but it is required
      for Lyapunov exponents, CLVs, SALI, LDI, and GALI.
    - Built-in models and supported integrators can be queried with the
      corresponding class methods.

    See Also
    --------
    HamiltonianSystem : Class for separable Hamiltonian systems.
    DiscreteDynamicalSystem : Class for discrete-time maps.
    """

    __AVAILABLE_MODELS: Dict[str, Dict[str, Any]] = {
        "lorenz system": {
            "description": "3D Lorenz system",
            "equation": Math(
                r"""
        \dot{x} = \sigma (y - x), \quad
        \dot{y} = x (\rho - z) - y, \quad
        \dot{z} = xy - \beta z
        """
            ),
            "equation_readable": "x' = σ(y − x), y' = x(ρ − z) − y, z' = xy − βz",
            "notes": "Classic Lorenz 1963 model of atmospheric convection. Exhibits chaotic dynamics for some parameter values.",
            "has_jacobian": True,
            "has_variational_equations": True,
            "equations_of_motion": lorenz_system,
            "jacobian": lorenz_jacobian,
            "dimension": 3,
            "number_of_parameters": 3,
            "parameters": ["sigma", "rho", "beta"],
        },
        "henon heiles": {
            "description": "Two d.o.f. Hénon–Heiles system",
            "equation": Math(
                r"""
        H = \frac{1}{2}(p_x^2 + p_y^2) + 
            \frac{1}{2}(x^2 + y^2) + 
            x^2 y - \frac{1}{3}y^3
        """
            ),
            "equation_readable": "H = ½(pₓ² + pᵧ²) + ½(x² + y²) + x²y − y³/3",
            "notes": "Hamiltonian system modeling stellar motion near a galactic center; classic example of a mixed chaotic/regular system.",
            "has_jacobian": True,
            "has_variational_equations": True,
            "equations_of_motion": henon_heiles,
            "jacobian": henon_heiles_jacobian,
            "dimension": 4,
            "number_of_parameters": 0,
            "parameters": [],
        },
        "rossler system": {
            "description": "3D Rössler system",
            "equation": Math(
                r"""
        \dot{x} = -y - z, \quad
        \dot{y} = x + a y, \quad
        \dot{z} = b + z(x - c)
        """
            ),
            "equation_readable": "x' = −y − z, y' = x + a y, z' = b + z(x − c)",
            "notes": "Continuous-time chaotic system proposed by Otto Rössler (1976).",
            "has_jacobian": True,
            "has_variational_equations": True,
            "equations_of_motion": rossler_system,
            "jacobian": rossler_system_jacobian,
            "dimension": 3,
            "number_of_parameters": 3,
            "parameters": ["a", "b", "c"],
        },
        "4d rossler system": {
            "description": "4D Rössler system",
            "equation": Math(
                r"""
        \dot{x} = -y - z, \quad
        \dot{y} = x + a y + w, \quad
        \dot{z} = b + z(x - c), \quad
        \dot{w} = -d y
        """
            ),
            "equation_readable": "x' = −y − z, y' = x + a y + w, z' = b + z(x − c), w' = −d y",
            "notes": "A 4D generalization of the Rössler attractor with an added variable w.",
            "has_jacobian": True,
            "has_variational_equations": True,
            "equations_of_motion": rossler_system_4D,
            "jacobian": rossler_system_4D_jacobian,
            "dimension": 4,
            "number_of_parameters": 4,
            "parameters": ["a", "b", "c", "d"],
        },
        "duffing": {
            "description": "Duffing oscillator (nonlinear forced damped oscillator)",
            "equation": Math(
                r"\ddot{x} + \delta \dot{x} - \alpha x + \beta x^3 = \gamma \cos(\omega t)"
            ),
            "equation_readable": "x'' + δ x' − α x + β x³ = γ cos(ω t)",
            "notes": "A nonlinear oscillator with a double-well potential, forced and damped; exhibits chaos under some parameters.",
            "has_jacobian": True,
            "has_variational_equations": True,
            "equations_of_motion": duffing,
            "jacobian": duffing_jacobian,
            "dimension": 2,
            "number_of_parameters": 5,
            "parameters": ["delta", "alpha", "beta", "gamma", "omega"],
        },
    }

    __AVAILABLE_INTEGRATORS: Dict[str, Dict[str, Any]] = {
        "rk4": {
            "description": "4th order Runge-Kutta method with fixed step size",
            "integrator": rk4_step_wrapped,
            "estimate_initial_step": False,
        },
        "rk45": {
            "description": "Adaptive 4th/5th order Runge-Kutta-Fehlberg method (RK45) with embedded error estimation",
            "integrator": rk45_step_wrapped,
            "estimate_initial_step": True,
        },
    }

    def __init__(
        self,
        model: str | None = None,
        equations_of_motion: Callable | None = None,
        jacobian: Callable | None = None,
        system_dimension: int | None = None,
        parameters: numeric_like_t | None = None,
        number_of_parameters: int | None = None,
    ) -> None:
        self.__equations_of_motion: Callable
        self.__jacobian: Callable | None
        self.__system_dimension: int
        self.__parameters: NDArray[np.float64] | None
        self.__number_of_parameters: int
        self.__model: str
        self.__atol: np.float64
        self.__rtol: np.float64
        self.__time_step: np.float64

        warnings.filterwarnings("ignore", category=NumbaExperimentalFeatureWarning)

        if model is not None and equations_of_motion is not None:
            raise ValueError("Cannot specify both model and custom system")

        if model is not None:
            model = model.lower()
            if model not in self.__AVAILABLE_MODELS:
                available = "\n".join(
                    f"- {name}: {info['description']}"
                    for name, info in self.__AVAILABLE_MODELS.items()
                )
                raise ValueError(
                    f"Model '{model}' not implemented. Available models:\n{available}"
                )

            model_info = self.__AVAILABLE_MODELS[model]

            self.__model = model
            self.__equations_of_motion = model_info["equations_of_motion"]
            self.__jacobian = model_info["jacobian"]
            self.__system_dimension = int(model_info["dimension"])
            self.__parameters = None
            self.__number_of_parameters = int(model_info["number_of_parameters"])

            if jacobian is not None:
                if not callable(jacobian):
                    raise TypeError("Custom Jacobian must be callable or None")
                self.__jacobian = jacobian

        elif equations_of_motion is not None and system_dimension is not None:
            if not callable(equations_of_motion):
                raise TypeError("Custom equations_of_motion must be callable")

            if jacobian is not None and not callable(jacobian):
                raise TypeError("Custom Jacobian must be callable or None")

            validate_positive(system_dimension, "system_dimension", Integral)

            if number_of_parameters is not None:
                validate_non_negative(
                    number_of_parameters, "number_of_parameters", Integral
                )

            validated_parameters: NDArray[np.float64] | None
            validated_number_of_parameters: int

            if parameters is not None:
                parameters_input: numeric_like_t = parameters

                if isinstance(parameters_input, (int, float, np.integer, np.floating)):
                    validated_number_of_parameters = 1
                else:
                    parameters_arr = np.asarray(parameters_input, dtype=np.float64)
                    if parameters_arr.ndim != 1:
                        raise ValueError(
                            f"`parameters` must be a 1D array or scalar. Got shape {parameters_arr.shape}."
                        )
                    validated_number_of_parameters = int(parameters_arr.size)

                if (
                    number_of_parameters is not None
                    and validated_number_of_parameters != number_of_parameters
                ):
                    raise ValueError(
                        f"Expected {number_of_parameters} parameter(s), but got {validated_number_of_parameters}."
                    )

                validated_parameters = validate_parameters(
                    parameters_input, validated_number_of_parameters
                )
            else:
                if number_of_parameters is None:
                    raise ValueError(
                        "For a custom system, you must provide either parameters or number_of_parameters."
                    )

                validated_number_of_parameters = int(number_of_parameters)
                validated_parameters = None
            self.__model = "custom"
            self.__equations_of_motion = equations_of_motion
            self.__jacobian = jacobian
            self.__system_dimension = int(system_dimension)
            self.__parameters = validated_parameters
            self.__number_of_parameters = validated_number_of_parameters

        else:
            raise ValueError(
                "Must specify either a model name or a custom system with equations_of_motion, system_dimension, and parameters or number_of_parameters."
            )

        self.__integrator = "rk4"
        self.__integrator_func = rk4_step_wrapped
        self.__time_step = np.float64(1e-2)
        self.__atol = np.float64(1e-6)
        self.__rtol = np.float64(1e-3)

    @classmethod
    def available_models(cls) -> List[str]:
        """Return a list of available models."""
        return list(cls.__AVAILABLE_MODELS.keys())

    @classmethod
    def available_integrators(cls) -> List[str]:
        """Return a list of available integrators."""
        return list(cls.__AVAILABLE_INTEGRATORS.keys())

    @property
    def info(self) -> Dict[str, Any]:
        """Return a dictionary with information about the current model."""

        if self.__model is None:
            raise ValueError(
                "The 'info' property is only available when a model is provided."
            )

        model = self.__model.lower()

        return self.__AVAILABLE_MODELS[model]

    @property
    def integrator_info(self):
        """Return a dictionary with information about the current integrator."""
        integrator = self.__integrator.lower()

        return self.__AVAILABLE_INTEGRATORS[integrator]

    def integrator(
        self,
        integrator: str,
        time_step: np.float64 = np.float64(1e-2),
        atol: np.float64 = np.float64(1e-6),
        rtol: np.float64 = np.float64(1e-3),
    ):
        """Define the integrator.

        Parameters
        ----------
        integrator : str
            The integrator name. Available options are 'rk4' and 'rk45'
        time_step : np.float64, optional
            The integration time step when `integrator='rk4'`, by default 1e-2
        atol : np.float64, optional
            The absolute tolerance used when `integrator='rk45'`, by default 1e-6
        rtol : np.float64, optional
            The relative tolerance used when `integrator='rk45'`, by default 1e-3

        Raises
        ------
        ValueError
            If `time_step`, `atol`, or `rtol` are negative.
            If `integrator` is not available.
        TypeError
            - If `integrator` is not a string.
            - If `time_step`, `atol`, or `rtol` are not valid numbers

        Examples
        --------
        >>> from pynamicalsys import ContinuousDynamicalSystem as cds
        >>> cds.available_integrators()
        ['rk4', 'rk45']
        >>> ds = cds(model="lorenz system")
        >>> ds.integrator("rk4", time_step=0.001) #  To use the RK4 integrator
        >>> ds.integrator("rk45", atol=1e-10, rtol=1e-8) #  To use the RK45 integrator
        """

        if not isinstance(integrator, str):
            raise ValueError("integrator must be a string.")
        validate_non_negative(time_step, "time_step", type_=Real)
        validate_non_negative(atol, "atol", type_=Real)
        validate_non_negative(rtol, "rtol", type_=Real)

        if integrator in self.__AVAILABLE_INTEGRATORS:
            self.__integrator = integrator.lower()
            integrator_info = self.__AVAILABLE_INTEGRATORS[self.__integrator]
            self.__integrator_func = integrator_info["integrator"]
            self.__time_step = time_step
            self.__atol = atol
            self.__rtol = rtol

        else:
            integrator = integrator.lower()
            if integrator not in self.__AVAILABLE_INTEGRATORS:
                available = "\n".join(
                    f"- {name}: {info['description']}"
                    for name, info in self.__AVAILABLE_INTEGRATORS.items()
                )
                raise ValueError(
                    f"Integrator '{integrator}' not implemented. Available integrators:\n{available}"
                )

    def set_parameters(
        self, parameters: Union[NDArray[np.float64], Sequence[float], float]
    ) -> None:
        """
        Set the parameter vector of the dynamical system.

        This method validates and stores the model parameters. The input can
        be a scalar, a sequence of floats, or a NumPy array. It is internally
        converted into a ``float64`` NumPy array of the appropriate size.

        Parameters
        ----------
        parameters : float or sequence of float or ndarray of shape (P,)
            The parameter set to be used by the system.

        Returns
        -------
        None
        """
        parameters = validate_parameters(parameters, self.__number_of_parameters)
        self.__parameters = parameters

    def get_parameters(self) -> NDArray[np.float64] | None:
        """
        Return the current parameter vector of the dynamical system.

        Returns
        -------
        ndarray of float64, shape (P,)
            The parameter vector currently stored in the system.
        """
        return self.__parameters

    def __get_initial_time_step(
        self, u: NDArray[np.float64], parameters: NDArray[np.float64]
    ) -> np.float64:
        if self.integrator_info["estimate_initial_step"]:
            time_step = estimate_initial_step(
                np.float64(0.0),
                u,
                parameters,
                self.__equations_of_motion,
                atol=self.__atol,
                rtol=self.__rtol,
            )
        else:
            time_step = self.__time_step

        return np.float64(time_step)

    def evolve_system(
        self,
        u: numeric_like_t,
        total_time: numeric_t,
        parameters: numeric_like_t | None = None,
    ) -> NDArray[np.float64]:
        """
        Evolve the dynamical system from the given initial condition over a specified time interval.

        Parameters
        ----------
        u : numeric_like_t
            Initial condition of the system. It must define a 1D state vector whose
            length matches the system dimension.
        total_time : numeric_t
            Total integration time.
        parameters : numeric_like_t | None, optional
            Parameters of the system. If `None`, the parameters stored in the
            instance are used.

        Returns
        -------
        NDArray[np.float64]
            State of the system at `time = total_time`.

        Raises
        ------
        ValueError
            - If `u` does not match the system dimension.
            - If the number of parameters does not match the expected number.
            - If `total_time` is negative.
        TypeError
            - If `total_time` is not a valid real number.

        Examples
        --------
        >>> from pynamicalsys import ContinuousDynamicalSystem as cds
        >>> ds = cds(model="lorenz system")
        >>> ds.integrator("rk4", time_step=0.01)
        >>> parameters = [10.0, 28.0, 8.0 / 3.0]
        >>> u0 = [1.0, 1.0, 1.0]
        >>> ds.evolve_system(u0, 10.0, parameters=parameters)

        >>> ds.integrator("rk45", atol=1e-8, rtol=1e-6)
        >>> ds.evolve_system(u0, 10.0, parameters=parameters)
        """
        u = validate_initial_conditions(
            u, self.__system_dimension, allow_ensemble=False
        )

        if parameters is None:
            parameters = self.__parameters
        parameters = validate_parameters(parameters, self.__number_of_parameters)

        _, total_time = validate_times(None, total_time)

        time_step = self.__get_initial_time_step(u, parameters)

        total_time += time_step

        return evolve_system(
            u=u,
            parameters=parameters,
            total_time=total_time,
            equations_of_motion=self.__equations_of_motion,
            time_step=time_step,
            atol=self.__atol,
            rtol=self.__rtol,
            integrator=self.__integrator_func,
        )

    def trajectory(
        self,
        u: numeric_like_t,
        total_time: numeric_t,
        parameters: numeric_like_t | None = None,
        transient_time: numeric_t | None = None,
    ) -> NDArray[np.float64] | list[NDArray[np.float64]]:
        """
        Compute the trajectory of the dynamical system over a specified time interval.

        Parameters
        ----------
        u : numeric_like_t
            Initial condition or ensemble of initial conditions. It must define either
            a 1D state vector of length equal to the system dimension or a 2D array
            whose rows are valid initial conditions.
        total_time : numeric_t
            Total integration time.
        parameters : numeric_like_t | None, optional
            Parameters of the system. If `None`, the parameters stored in the
            instance are used.
        transient_time : numeric_t | None, optional
            Initial integration time discarded before storing the trajectory.

        Returns
        -------
        NDArray[np.float64] | list[NDArray[np.float64]]
            The computed trajectory.

            - If `u` is one initial condition, returns an array of shape
              `(num_steps, neq + 1)`, where the first column contains time and the
              remaining columns contain the state coordinates.
            - If `u` is an ensemble of initial conditions and all trajectories have the
              same number of stored time steps, the result may be represented as arrays
              with common shape `(num_ic, num_steps, neq + 1)`.
            - If `u` is an ensemble of initial conditions and the trajectories have
              different numbers of stored time steps, returns a list of arrays, where
              each element has shape `(num_steps_i, neq + 1)`.

        Raises
        ------
        ValueError
            - If `u` does not match the system dimension.
            - If the number of parameters does not match the expected number.
            - If `total_time` is negative.
            - If `transient_time` is negative or not smaller than `total_time`.
        TypeError
            - If `total_time` or `transient_time` is not a valid real number.

        Examples
        --------
        >>> from pynamicalsys import ContinuousDynamicalSystem as cds
        >>> ds = cds(model="lorenz system")
        >>> u0 = [0.1, 0.1, 0.1]
        >>> parameters = [10.0, 28.0, 8.0 / 3.0]
        >>> traj = ds.trajectory(u0, 700.0, parameters=parameters, transient_time=500.0)

        >>> u0s = [
        ...     [0.1, 0.1, 0.1],
        ...     [0.2, 0.2, 0.2],
        ...     [0.3, 0.3, 0.3],
        ... ]
        >>> trajs = ds.trajectory(u0s, 700.0, parameters=parameters, transient_time=500.0)
        """
        u = validate_initial_conditions(u, self.__system_dimension)

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        else:
            parameters = validate_parameters(parameters, self.__number_of_parameters)

        transient_time, total_time = validate_times(transient_time, total_time)

        time_step = self.__get_initial_time_step(u, parameters)

        if u.ndim == 1:
            return generate_trajectory(
                u=u,
                parameters=parameters,
                total_time=total_time,
                equations_of_motion=self.__equations_of_motion,
                transient_time=transient_time,
                time_step=time_step,
                atol=self.__atol,
                rtol=self.__rtol,
                integrator=self.__integrator_func,
            )

        return ensemble_trajectories(
            u=u,
            parameters=parameters,
            total_time=total_time,
            equations_of_motion=self.__equations_of_motion,
            transient_time=transient_time,
            time_step=time_step,
            atol=self.__atol,
            rtol=self.__rtol,
            integrator=self.__integrator_func,
        )

    def poincare_section(
        self,
        u: numeric_like_t,
        num_intersections: int_t,
        section_index: int_t,
        section_value: numeric_t,
        parameters: numeric_like_t | None = None,
        transient_time: numeric_t | None = None,
        crossing: int_t = 1,
    ) -> NDArray[np.float64]:
        """
        Compute the Poincaré section of the dynamical system for given initial conditions.

        A Poincaré section records the points where a trajectory intersects a chosen
        hypersurface in phase space. This reduces a continuous flow to a lower-dimensional
        map, making it easier to identify periodic, quasi-periodic, or chaotic motion.

        Parameters
        ----------
        u : numeric_like_t
            Initial condition or ensemble of initial conditions. It must define either
            a 1D state vector of length equal to the system dimension or a 2D array
            whose rows are valid initial conditions.
        num_intersections : int_t
            Number of intersections to record.
        section_index : int_t
            Index of the coordinate defining the Poincaré section.
        section_value : numeric_t
            Value of the coordinate at which the section is defined.
        parameters : numeric_like_t | None, optional
            Parameters of the system. If `None`, the parameters stored in the
            instance are used.
        transient_time : numeric_t | None, optional
            Initial integration time discarded before recording the section.
        crossing : int_t, optional
            Type of crossing to consider:
            - `1`  : positive crossings
            - `-1` : negative crossings
            - `0`  : all crossings

        Returns
        -------
        NDArray[np.float64]
            Poincaré section points.

            - If `u` is one initial condition, returns an array of shape
              `(num_intersections, neq + 1)`, where the first column is the crossing
              time and the remaining columns are the state coordinates.
            - If `u` is an ensemble of initial conditions, returns an array of shape
              `(num_ic, num_intersections, neq + 1)`.

        Raises
        ------
        ValueError
            - If `u` does not match the system dimension.
            - If the number of parameters does not match the expected number.
            - If `num_intersections` is negative.
            - If `section_index` is outside the valid coordinate range.
            - If `transient_time` is negative or not smaller than the total integration time
              when such a comparison is required elsewhere.
            - If `crossing` is not one of `-1`, `0`, or `1`.
        TypeError
            - If `section_value` is not a valid real number.
            - If `num_intersections`, `section_index`, or `crossing` are not integers.
            - If `transient_time` is not a valid real number.

        Examples
        --------
        >>> from pynamicalsys import ContinuousDynamicalSystem as cds
        >>> ds = cds(model="lorenz system")
        >>> u0 = [0.1, 0.1, 0.1]
        >>> parameters = [10.0, 28.0, 8.0 / 3.0]
        >>> ps = ds.poincare_section(
        ...     u0,
        ...     num_intersections=500,
        ...     section_index=2,
        ...     section_value=25.0,
        ...     parameters=parameters,
        ... )

        >>> u0s = [[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]]
        >>> ps_ensemble = ds.poincare_section(
        ...     u0s,
        ...     num_intersections=500,
        ...     section_index=2,
        ...     section_value=25.0,
        ...     parameters=parameters,
        ... )
        """
        u = validate_initial_conditions(u, self.__system_dimension)

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        else:
            parameters = validate_parameters(parameters, self.__number_of_parameters)

        validate_non_negative(num_intersections, "num_intersections", Integral)
        validate_non_negative(section_index, "section_index", Integral)

        if section_index >= self.__system_dimension:
            raise ValueError("section_index must be smaller than system_dimension")

        if isinstance(section_value, bool) or not isinstance(section_value, Real):
            raise TypeError("section_value must be a valid real number")
        section_value = np.float64(section_value)

        transient_time, _ = validate_times(transient_time, 1.0e300)

        if isinstance(crossing, bool) or not isinstance(crossing, Integral):
            raise TypeError("crossing must be an integer")
        if crossing not in (-1, 0, 1):
            raise ValueError(
                "crossing must be -1 (downward), 0 (all crossings), or 1 (upward)"
            )

        time_step = self.__get_initial_time_step(u, parameters)

        if u.ndim == 1:
            return generate_poincare_section(
                u=u,
                parameters=parameters,
                num_intersections=int(num_intersections),
                equations_of_motion=self.__equations_of_motion,
                transient_time=transient_time,
                time_step=time_step,
                atol=self.__atol,
                rtol=self.__rtol,
                integrator=self.__integrator_func,
                section_index=int(section_index),
                section_value=section_value,
                crossing=int(crossing),
            )

        return ensemble_poincare_section(
            u=u,
            parameters=parameters,
            num_intersections=int(num_intersections),
            equations_of_motion=self.__equations_of_motion,
            transient_time=transient_time,
            time_step=time_step,
            atol=self.__atol,
            rtol=self.__rtol,
            integrator=self.__integrator_func,
            section_index=int(section_index),
            section_value=section_value,
            crossing=int(crossing),
        )

    def stroboscopic_map(
        self,
        u: numeric_like_t,
        num_samples: int_t,
        sampling_time: numeric_t,
        parameters: numeric_like_t | None = None,
        transient_time: numeric_t | None = None,
    ) -> NDArray[np.float64]:
        """
        Compute the stroboscopic map of the dynamical system for given initial conditions.

        A stroboscopic map samples the state of a continuous-time system at fixed time
        intervals. This converts the flow into a discrete sequence that can reveal
        periodicity, phase locking, and bifurcations.

        Parameters
        ----------
        u : numeric_like_t
            Initial condition or ensemble of initial conditions. It must define either
            a 1D state vector of length equal to the system dimension or a 2D array
            whose rows are valid initial conditions.
        num_samples : int_t
            Number of samples to record.
        sampling_time : numeric_t
            Time interval between consecutive samples.
        parameters : numeric_like_t | None, optional
            Parameters of the system. If `None`, the parameters stored in the
            instance are used.
        transient_time : numeric_t | None, optional
            Initial integration time discarded before recording the map.

        Returns
        -------
        NDArray[np.float64]
            Stroboscopic map points.

            - If `u` is one initial condition, returns an array of shape
              `(num_samples, neq + 1)`, where the first column is time and the
              remaining columns are the sampled state coordinates.
            - If `u` is an ensemble of initial conditions, returns an array of shape
              `(num_ic, num_samples, neq + 1)`.

        Raises
        ------
        ValueError
            - If `u` does not match the system dimension.
            - If the number of parameters does not match the expected number.
            - If `num_samples` is negative.
            - If `sampling_time` is negative.
            - If `transient_time` is negative.
        TypeError
            - If `num_samples` is not an integer.
            - If `sampling_time` or `transient_time` is not a valid real number.

        Examples
        --------
        >>> from pynamicalsys import ContinuousDynamicalSystem as cds
        >>> ds = cds(model="lorenz system")
        >>> u0 = [0.1, 0.1, 0.1]
        >>> parameters = [10.0, 28.0, 8.0 / 3.0]
        >>> smap = ds.stroboscopic_map(
        ...     u0,
        ...     num_samples=500,
        ...     sampling_time=0.1,
        ...     parameters=parameters,
        ... )

        >>> u0s = [[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]]
        >>> smap_ensemble = ds.stroboscopic_map(
        ...     u0s,
        ...     num_samples=500,
        ...     sampling_time=0.1,
        ...     parameters=parameters,
        ... )
        """
        u = validate_initial_conditions(u, self.__system_dimension)

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        else:
            parameters = validate_parameters(parameters, self.__number_of_parameters)

        validate_non_negative(num_samples, "num_samples", Integral)
        num_samples = int(num_samples)
        validate_non_negative(sampling_time, "sampling_time", Real)
        sampling_time = np.float64(sampling_time)

        transient_time, _ = validate_times(transient_time, 1.0e300)

        time_step = self.__get_initial_time_step(u, parameters)

        if u.ndim == 1:
            return generate_stroboscopic_map(
                u=u,
                parameters=parameters,
                num_intersections=num_samples,
                sampling_time=sampling_time,
                equations_of_motion=self.__equations_of_motion,
                transient_time=transient_time,
                time_step=time_step,
                atol=self.__atol,
                rtol=self.__rtol,
                integrator=self.__integrator_func,
            )

        return ensemble_stroboscopic_map(
            u=u,
            parameters=parameters,
            num_intersections=num_samples,
            sampling_time=sampling_time,
            equations_of_motion=self.__equations_of_motion,
            transient_time=transient_time,
            time_step=time_step,
            atol=self.__atol,
            rtol=self.__rtol,
            integrator=self.__integrator_func,
        )

    def maxima_map(
        self,
        u: numeric_like_t,
        num_points: int_t,
        maxima_index: int_t,
        parameters: numeric_like_t | None = None,
        transient_time: numeric_t | None = None,
    ) -> NDArray[np.float64]:
        """
        Compute the maxima map of the dynamical system for given initial conditions.

        A maxima map records the local maxima of a chosen system variable along the
        trajectory. By plotting successive maxima, one obtains a discrete return map
        that can reveal oscillation amplitudes, period-doubling cascades, and other
        nonlinear behavior.

        Parameters
        ----------
        u : numeric_like_t
            Initial condition or ensemble of initial conditions. It must define either
            a 1D state vector of length equal to the system dimension or a 2D array
            whose rows are valid initial conditions.
        num_points : int_t
            Number of maxima points to record.
        maxima_index : int_t
            Index of the state variable whose maxima are recorded.
        parameters : numeric_like_t | None, optional
            Parameters of the system. If `None`, the parameters stored in the
            instance are used.
        transient_time : numeric_t | None, optional
            Initial integration time discarded before recording the maxima map.

        Returns
        -------
        NDArray[np.float64]
            Maxima map points.

            - If `u` is one initial condition, returns an array of shape
              `(num_points, neq + 1)`, where the first column is time and the
              remaining columns are the state coordinates at each detected maximum.
            - If `u` is an ensemble of initial conditions, returns an array of shape
              `(num_ic, num_points, neq + 1)`.

        Raises
        ------
        ValueError
            - If `u` does not match the system dimension.
            - If the number of parameters does not match the expected number.
            - If `num_points` is negative.
            - If `maxima_index` is outside the valid coordinate range.
            - If `transient_time` is negative.
        TypeError
            - If `num_points` or `maxima_index` is not an integer.
            - If `transient_time` is not a valid real number.

        Examples
        --------
        >>> from pynamicalsys import ContinuousDynamicalSystem as cds
        >>> ds = cds(model="lorenz system")
        >>> u0 = [0.1, 0.1, 0.1]
        >>> parameters = [10.0, 28.0, 8.0 / 3.0]
        >>> mmap = ds.maxima_map(u0, 500, 0, parameters=parameters)
        >>> mmap.shape
        (500, 4)

        >>> u0s = [[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]]
        >>> mmap_ensemble = ds.maxima_map(u0s, 500, 0, parameters=parameters)
        >>> mmap_ensemble.shape
        (2, 500, 4)
        """
        u = validate_initial_conditions(u, self.__system_dimension)

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        else:
            parameters = validate_parameters(parameters, self.__number_of_parameters)

        validate_non_negative(num_points, "num_points", Integral)
        validate_non_negative(maxima_index, "maxima_index", Integral)

        if maxima_index >= self.__system_dimension:
            raise ValueError("maxima_index must be smaller than system_dimension")

        transient_time, _ = validate_times(transient_time, 1.0e300)

        time_step = self.__get_initial_time_step(u, parameters)

        if u.ndim == 1:
            return generate_maxima_map(
                u=u,
                parameters=parameters,
                num_peaks=int(num_points),
                maxima_index=int(maxima_index),
                equations_of_motion=self.__equations_of_motion,
                transient_time=transient_time,
                time_step=time_step,
                atol=self.__atol,
                rtol=self.__rtol,
                integrator=self.__integrator_func,
            )

        return ensemble_maxima_map(
            u=u,
            parameters=parameters,
            num_peaks=int(num_points),
            maxima_index=int(maxima_index),
            equations_of_motion=self.__equations_of_motion,
            transient_time=transient_time,
            time_step=time_step,
            atol=self.__atol,
            rtol=self.__rtol,
            integrator=self.__integrator_func,
        )

    def basin_of_attraction(
        self,
        u: numeric_like_t,
        num_intersections: int_t,
        parameters: numeric_like_t | None = None,
        transient_time: numeric_t | None = None,
        map_type: str = "SM",
        section_index: int_t | None = None,
        section_value: numeric_t | None = None,
        crossing: int_t | None = None,
        sampling_time: numeric_t | None = None,
        eps: numeric_t = 0.05,
        min_samples: int_t = 1,
    ) -> NDArray[np.int32]:
        """
        Compute the basin of attraction for a set of initial conditions.

        Parameters
        ----------
        u : numeric_like_t
            Ensemble of initial conditions for the dynamical system. It must define
            either a valid 2D array whose rows are initial conditions, or a single
            initial condition if single-trajectory analysis is intended.
        num_intersections : int_t
            Number of intersections or samples used to construct the reduced map.
        parameters : numeric_like_t | None, optional
            System parameters. If `None`, the parameters stored in the instance are used.
        transient_time : numeric_t | None, optional
            Initial integration time discarded before analyzing the trajectories.
        map_type : str, optional
            Type of reduced map to construct:
            - `"SM"` for stroboscopic map
            - `"PS"` for Poincaré section
        section_index : int_t | None, optional
            Index of the section coordinate when `map_type="PS"`.
        section_value : numeric_t | None, optional
            Value of the Poincaré section when `map_type="PS"`.
        crossing : int_t | None, optional
            Crossing orientation when `map_type="PS"`:
            - `-1` for downward crossings
            - `0` for all crossings
            - `1` for upward crossings
        sampling_time : numeric_t | None, optional
            Sampling time when `map_type="SM"`.
        eps : numeric_t, optional
            Maximum neighborhood radius used by DBSCAN.
        min_samples : int_t, optional
            Minimum number of points required to form a DBSCAN cluster.

        Returns
        -------
        NDArray[np.int32]
            Integer labels indicating which attractor each initial condition belongs to.
            The label `-1` indicates noise.

        Raises
        ------
        ValueError
            - If `u` does not match the system dimension.
            - If the number of parameters does not match the expected number.
            - If `num_intersections`, `eps`, or `min_samples` is negative.
            - If `map_type` is not `"SM"` or `"PS"`.
            - If `section_index` is outside the valid coordinate range.
            - If `crossing` is not one of `-1`, `0`, or `1`.
        TypeError
            - If `map_type` is not a string.
            - If `section_value`, `sampling_time`, or `transient_time` is not a valid real number.
            - If `num_intersections`, `section_index`, `crossing`, or `min_samples` is not an integer.

        Notes
        -----
        The basin of attraction is estimated by first constructing either a
        stroboscopic map or a Poincaré section, then clustering trajectory centroids
        with DBSCAN. Trajectories whose centroids belong to the same cluster are
        assigned to the same attractor basin.
        """
        u = validate_initial_conditions(u, self.__system_dimension)

        validate_non_negative(num_intersections, "num_intersections", Integral)

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        else:
            parameters = validate_parameters(parameters, self.__number_of_parameters)

        transient_time, _ = validate_times(transient_time, 1.0e300)

        if not isinstance(map_type, str):
            raise TypeError("map_type must be a string")
        if map_type not in ("SM", "PS"):
            raise ValueError("map_type must be either 'SM' or 'PS'")

        if section_index is not None:
            validate_non_negative(section_index, "section_index", Integral)
            if section_index >= self.__system_dimension:
                raise ValueError("section_index must be smaller than system_dimension")

        if section_value is not None:
            if isinstance(section_value, bool) or not isinstance(section_value, Real):
                raise TypeError("section_value must be a valid real number")

        if crossing is not None:
            if isinstance(crossing, bool) or not isinstance(crossing, Integral):
                raise TypeError("crossing must be an integer")
            if crossing not in (-1, 0, 1):
                raise ValueError(
                    "crossing must be -1 (downward), 0 (all crossings), or 1 (upward)"
                )

        if sampling_time is not None:
            validate_non_negative(sampling_time, "sampling_time", Real)

        validate_non_negative(eps, "eps", Real)
        validate_non_negative(min_samples, "min_samples", Integral)

        if map_type == "PS":
            if section_index is None or section_value is None or crossing is None:
                raise ValueError(
                    "section_index, section_value, and crossing must be provided when map_type='PS'"
                )
        else:
            if sampling_time is None:
                raise ValueError("sampling_time must be provided when map_type='SM'")

        time_step = self.__get_initial_time_step(u, parameters)

        return basin_of_attraction(
            u=u,
            parameters=parameters,
            num_intersections=int(num_intersections),
            equations_of_motion=self.__equations_of_motion,
            transient_time=transient_time,
            time_step=time_step,
            atol=self.__atol,
            rtol=self.__rtol,
            integrator=self.__integrator_func,
            select_map=map_type,
            section_index=None if section_index is None else int(section_index),
            section_value=None if section_value is None else np.float64(section_value),
            crossing=None if crossing is None else int(crossing),
            sampling_time=None if sampling_time is None else np.float64(sampling_time),
            eps=np.float64(eps),
            min_samples=int(min_samples),
        )

    def lyapunov(
        self,
        u: numeric_like_t,
        total_time: numeric_t,
        parameters: numeric_like_t | None = None,
        transient_time: numeric_t | None = None,
        num_exponents: int_t | None = None,
        return_history: bool = False,
        seed: int_t = 13,
        log_base: numeric_t = np.e,
        method: str = "QR",
        endpoint: bool = True,
    ) -> np.float64 | NDArray[np.float64]:
        """
        Calculate the Lyapunov exponents of the dynamical system.

        The Lyapunov exponents measure the average exponential rates of divergence
        or convergence of nearby trajectories.

        Parameters
        ----------
        u : numeric_like_t
            Initial condition of the system. It must define a 1D state vector whose
            length matches the system dimension.
        total_time : numeric_t
            Total integration time.
        parameters : numeric_like_t | None, optional
            Parameters of the system. If `None`, the parameters stored in the
            instance are used.
        transient_time : numeric_t | None, optional
            Initial integration time discarded before computing the exponents.
        num_exponents : int_t | None, optional
            Number of Lyapunov exponents to compute. If `None`, the full spectrum
            is computed.
        return_history : bool, optional
            If `True`, return the time evolution of the exponents.
        seed : int_t, optional
            Seed used to initialize the deviation vectors.
        log_base : numeric_t, optional
            Base of the logarithm used to rescale the exponents.
        method : str, optional
            QR decomposition method:
            - `"QR"`: custom modified Gram-Schmidt QR
            - `"QR_HH"`: `numpy.linalg.qr` based on Householder reflections
        endpoint : bool, optional
            Whether to include the endpoint `time = total_time`.

        Returns
        -------
        np.float64 | NDArray[np.float64]
            - If `return_history=True`, returns a 2D array whose first column
              contains the sampled times and whose remaining columns contain
              the Lyapunov exponents at those times.
            - If `return_history=False` and `num_exponents == 1`, returns the
              largest Lyapunov exponent as a scalar.
            - If `return_history=False` and `num_exponents > 1`, returns a 1D
              array containing the final Lyapunov exponents.

        Raises
        ------
        ValueError
            - If the Jacobian is not available.
            - If `u` does not match the system dimension.
            - If the number of parameters does not match the expected number.
            - If `num_exponents` exceeds the system dimension.
            - If `method` is not `"QR"` or `"QR_HH"`.
            - If `log_base == 1`.
        TypeError
            - If `method` is not a string.
            - If `total_time`, `transient_time`, or `log_base` is not a valid real number.
            - If `num_exponents` or `seed` is not an integer.

        Notes
        -----
        By default, the computation uses the custom QR routine based on modified
        Gram-Schmidt. If `method="QR_HH"`, the wrapper passes `numpy.linalg.qr`
        to the low-level routine.
        """
        if self.__jacobian is None:
            raise ValueError(
                "Jacobian function is required to compute Lyapunov exponents"
            )

        u = validate_initial_conditions(
            u, self.__system_dimension, allow_ensemble=False
        )

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        else:
            parameters = validate_parameters(parameters, self.__number_of_parameters)

        transient_time, total_time = validate_times(transient_time, total_time)

        time_step = self.__get_initial_time_step(u, parameters)

        if num_exponents is None:
            num_exponents = self.__system_dimension
        else:
            validate_positive(num_exponents, "num_exponents", Integral)
            if num_exponents > self.__system_dimension:
                raise ValueError("num_exponents must be <= system_dimension")

        if endpoint:
            total_time += time_step

        if not isinstance(method, str):
            raise TypeError("method must be a string")

        method = method.upper()
        if method not in ["QR", "QR_HH"]:
            raise ValueError("method must be 'QR' or 'QR_HH'")

        validate_non_negative(log_base, "log_base", Real)
        if log_base == 1:
            raise ValueError("The logarithm function is not defined with base 1")

        validate_non_negative(seed, "seed", Integral)

        if num_exponents == 1:
            result = maximum_lyapunov_exponent(
                u=u,
                parameters=parameters,
                total_time=total_time,
                equations_of_motion=self.__equations_of_motion,
                jacobian=self.__jacobian,
                transient_time=transient_time,
                time_step=time_step,
                atol=self.__atol,
                rtol=self.__rtol,
                integrator=self.__integrator_func,
                return_history=return_history,
                seed=int(seed),
            )
        else:
            result = lyapunov_exponents(
                u=u,
                parameters=parameters,
                total_time=total_time,
                equations_of_motion=self.__equations_of_motion,
                jacobian=self.__jacobian,
                num_exponents=int(num_exponents),
                transient_time=transient_time,
                time_step=time_step,
                atol=self.__atol,
                rtol=self.__rtol,
                integrator=self.__integrator_func,
                return_history=return_history,
                seed=int(seed),
                method=method,
            )

        if return_history:
            return result / np.log(log_base)

        if num_exponents == 1:
            return np.float64(result[0, 0] / np.log(log_base))

        return result[0] / np.log(log_base)

    def CLV(
        self,
        u: numeric_like_t,
        total_time: numeric_t,
        parameters: numeric_like_t | None = None,
        num_clvs: int_t | None = None,
        transient_time: numeric_t | None = None,
        warmup_time: numeric_t = 0.0,
        tail_time: numeric_t = 0.0,
        qr_time_step: numeric_t | None = None,
        seed: int_t = 13,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Compute covariant Lyapunov vectors (CLVs) along a trajectory of this
        continuous-time dynamical system.

        The CLVs form a covariant time-dependent basis of the tangent space. The
        i-th CLV is associated with the i-th Lyapunov exponent and gives the local
        expanding or contracting direction.

        Parameters
        ----------
        u : numeric_like_t
            Initial condition. It must define a 1D state vector whose length matches
            the system dimension.
        total_time : numeric_t
            Total integration time over which the CLVs are returned.
        parameters : numeric_like_t | None, optional
            System parameters. If `None`, the parameters stored in the instance are used.
        num_clvs : int_t | None, optional
            Number of CLVs to compute. If `None`, the full set of
            `system_dimension` CLVs is computed.
        transient_time : numeric_t, optional
            Initial integration time discarded before starting the CLV computation.
        warmup_time : numeric_t, optional
            Forward warmup time used to drive the tangent basis toward the backward
            Lyapunov vectors.
        tail_time : numeric_t, optional
            Additional forward integration time after the main storage window, used
            to initialize the backward recursion.
        qr_time_step : numeric_t | None, optional
            Time between successive QR factorizations. If `None`, the initial
            integration step size is used.
        seed : int_t, optional
            Seed used to initialize the backward recursion.

        Returns
        -------
        tuple[NDArray[np.float64], NDArray[np.float64]]
            - `clvs`: array of shape `(N + 1, system_dimension, num_clvs)`
            - `traj`: array of shape `(N + 1, system_dimension + 1)` with time in
              the first column and state variables in the remaining columns

        Raises
        ------
        ValueError
            - If the Jacobian is not available.
            - If `u` does not match the system dimension.
            - If the number of parameters does not match the expected number.
            - If `total_time`, `transient_time`, `warmup_time`, or `tail_time` is negative.
            - If `num_clvs` is not in `[1, system_dimension]`.
            - If `qr_time_step` is not positive when provided.
        TypeError
            - If `seed` is not an integer.
            - If any time-like argument is not a valid real number.
        """
        if self.__jacobian is None:
            raise ValueError("Jacobian function is required to compute CLVs")

        u = validate_initial_conditions(
            u, self.__system_dimension, allow_ensemble=False
        )

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        else:
            parameters = validate_parameters(parameters, self.__number_of_parameters)

        transient_time, total_time = validate_times(transient_time, total_time, Real)
        validate_non_negative(warmup_time, "warmup_time", Real)
        validate_non_negative(tail_time, "tail_time", Real)
        validate_non_negative(seed, "seed", Integral)

        warmup_time = np.float64(warmup_time)
        tail_time = np.float64(tail_time)

        time_step = self.__get_initial_time_step(u, parameters)

        if num_clvs is None:
            num_clvs = self.__system_dimension
        else:
            validate_positive(num_clvs, "num_clvs", Integral)
            if num_clvs > self.__system_dimension:
                raise ValueError("num_clvs must be <= system_dimension")

        if qr_time_step is None:
            qr_time_step = time_step
        else:
            validate_positive(qr_time_step, "qr_time_step", Real)
            qr_time_step = np.float64(qr_time_step)

        return compute_clvs(
            u=u,
            parameters=parameters,
            total_time=total_time,
            equations_of_motion=self.__equations_of_motion,
            jacobian=self.__jacobian,
            num_clvs=int(num_clvs),
            transient_time=transient_time,
            warmup_time=warmup_time,
            tail_time=tail_time,
            time_step=time_step,
            qr_time_step=qr_time_step,
            atol=self.__atol,
            rtol=self.__rtol,
            integrator=self.__integrator_func,
            seed=int(seed),
        )

    def CLV_angles(
        self,
        u: numeric_like_t,
        total_time: numeric_t,
        parameters: numeric_like_t | None = None,
        subspaces: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...] | None = None,
        pairs: tuple[tuple[int, int], ...] | None = None,
        transient_time: numeric_t | None = None,
        warmup_time: numeric_t = 0,
        tail_time: numeric_t = 0,
        qr_time_step: numeric_t | None = None,
        seed: int_t = 13,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Compute CLV-based angle diagnostics for this continuous-time dynamical system.

        This method computes CLVs along a trajectory and returns time series of
        angles between user-specified CLV subspaces and/or CLV direction pairs.

        Parameters
        ----------
        u : numeric_like_t
            Initial condition. It must define a 1D state vector whose length matches
            the system dimension.
        total_time : numeric_t
            Total integration time used for the angle diagnostics.
        parameters : numeric_like_t | None, optional
            System parameters. If `None`, the parameters stored in the instance are used.
        subspaces : tuple[tuple[tuple[int, ...], tuple[int, ...]], ...] | None, optional
            Requested subspace-angle diagnostics. Each entry is a pair `(A, B)`, where
            `A` and `B` are tuples of CLV indices defining two subspaces.
        pairs : tuple[tuple[int, int], ...] | None, optional
            Requested pairwise CLV angles. Each entry `(i, j)` returns the angle
            between CLV `i` and CLV `j` at each sampled time.
        transient_time : numeric_t, optional
            Initial integration time discarded before starting the diagnostic.
        warmup_time : numeric_t, optional
            Forward QR warm-up duration passed to the CLV computation.
        tail_time : numeric_t, optional
            Backward-recursion convergence duration passed to the CLV computation.
        qr_time_step : numeric_t | None, optional
            Time spacing between successive QR re-orthonormalizations and CLV samples.
            If `None`, the initial integration step size is used.
        seed : int_t, optional
            Seed forwarded to the CLV computation.

        Returns
        -------
        tuple[NDArray[np.float64], NDArray[np.float64]]
            - `angles`: array of shape `(N + 1, M)` containing the requested angle time series
            - `traj`: sampled trajectory array with time in the first column

        Raises
        ------
        ValueError
            - If the Jacobian is not available.
            - If `total_time` is negative.
            - If `transient_time` is negative or not smaller than `total_time`.
            - If `warmup_time` or `tail_time` is negative.
            - If both `subspaces` and `pairs` are missing or empty.
            - If any CLV subspace or pair specification is invalid.
            - If `qr_time_step` is not positive when provided.
        TypeError
            - If `u` cannot be interpreted as a valid initial condition.
            - If `parameters` cannot be interpreted as valid system parameters.
            - If `seed` is not an integer.
        """
        if self.__jacobian is None:
            raise ValueError("Jacobian function is required to compute CLV angles")

        u = validate_initial_conditions(
            u, self.__system_dimension, allow_ensemble=False
        )

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        else:
            parameters = validate_parameters(parameters, self.__number_of_parameters)

        transient_time, total_time = validate_times(transient_time, total_time, Real)
        transient_time = np.float64(transient_time)

        validate_non_negative(warmup_time, "warmup_time", Real)
        validate_non_negative(tail_time, "tail_time", Real)
        validate_non_negative(seed, "seed", Integral)

        warmup_time = np.float64(warmup_time)
        tail_time = np.float64(tail_time)

        time_step = self.__get_initial_time_step(u, parameters)

        if qr_time_step is None:
            qr_time_step = time_step
        else:
            validate_positive(qr_time_step, "qr_time_step", Real)
            qr_time_step = np.float64(qr_time_step)

        pairs = validate_clv_pairs(pairs, self.__system_dimension)
        subspaces = validate_clv_subspaces(subspaces, self.__system_dimension)

        return clv_angles(
            u=u,
            parameters=parameters,
            total_time=total_time,
            equations_of_motion=self.__equations_of_motion,
            jacobian=self.__jacobian,
            transient_time=transient_time,
            warmup_time=warmup_time,
            tail_time=tail_time,
            time_step=time_step,
            qr_time_step=qr_time_step,
            atol=self.__atol,
            rtol=self.__rtol,
            integrator=self.__integrator_func,
            seed=int(seed),
            subspaces=subspaces,
            pairs=pairs,
        )

    def SALI(
        self,
        u: numeric_like_t,
        total_time: numeric_t,
        parameters: numeric_like_t | None = None,
        transient_time: numeric_t | None = None,
        return_history: bool = False,
        seed: int_t = 13,
        threshold: numeric_t = 1e-16,
        endpoint: bool = True,
    ) -> NDArray[np.float64]:
        """
        Calculate the Smaller Alignment Index (SALI) for a continuous-time dynamical system.

        Parameters
        ----------
        u : numeric_like_t
            Initial condition of the system. It must define a 1D state vector whose
            length matches the system dimension.
        total_time : numeric_t
            Total integration time.
        parameters : numeric_like_t | None, optional
            Parameters of the system. If `None`, the parameters stored in the
            instance are used.
        transient_time : numeric_t | None, optional
            Initial integration time discarded before computing SALI.
        return_history : bool, optional
            If `True`, return the time evolution of SALI.
        seed : int_t, optional
            Seed used to initialize the deviation vectors.
        threshold : numeric_t, optional
            Early stopping threshold. If SALI becomes smaller than `threshold`,
            the computation is terminated.
        endpoint : bool, optional
            Whether to include the endpoint `time = total_time`.

        Returns
        -------
        NDArray[np.float64]
            - If `return_history=False`, returns an array containing the final time
              and the final SALI value.
            - If `return_history=True`, returns the time history of SALI.

        Raises
        ------
        ValueError
            - If the Jacobian is not available.
            - If `u` does not match the system dimension.
            - If the number of parameters does not match the expected number.
            - If `total_time`, `transient_time`, or `threshold` is negative.
        TypeError
            - If `total_time`, `transient_time`, or `threshold` is not a valid real number.
            - If `seed` is not an integer.

        Examples
        --------
        >>> from pynamicalsys import ContinuousDynamicalSystem as cds
        >>> ds = cds(model="lorenz system")
        >>> u0 = [0.1, 0.1, 0.1]
        >>> parameters = [16.0, 45.92, 4.0]
        >>> ds.SALI(u0, 1000.0, parameters=parameters, transient_time=500.0)

        >>> sali_hist = ds.SALI(
        ...     u0,
        ...     1000.0,
        ...     parameters=parameters,
        ...     transient_time=500.0,
        ...     return_history=True,
        ... )
        """
        if self.__jacobian is None:
            raise ValueError("Jacobian function is required to compute SALI")

        u = validate_initial_conditions(
            u, self.__system_dimension, allow_ensemble=False
        )

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        else:
            parameters = validate_parameters(parameters, self.__number_of_parameters)

        transient_time, total_time = validate_times(transient_time, total_time)

        validate_non_negative(threshold, "threshold", Real)
        validate_non_negative(seed, "seed", Integral)

        threshold = np.float64(threshold)

        time_step = self.__get_initial_time_step(u, parameters)

        if endpoint:
            total_time += time_step

        result = sali(
            u=u,
            parameters=parameters,
            total_time=total_time,
            equations_of_motion=self.__equations_of_motion,
            jacobian=self.__jacobian,
            transient_time=transient_time,
            time_step=time_step,
            atol=self.__atol,
            rtol=self.__rtol,
            integrator=self.__integrator_func,
            return_history=return_history,
            seed=int(seed),
            threshold=threshold,
        )

        if return_history:
            return result

        return result[0]

    def LDI(
        self,
        u: numeric_like_t,
        total_time: numeric_t,
        k: int_t,
        parameters: numeric_like_t | None = None,
        transient_time: numeric_t | None = None,
        return_history: bool = False,
        seed: int_t = 13,
        threshold: numeric_t = 1e-16,
        endpoint: bool = True,
    ) -> NDArray[np.float64]:
        """
        Calculate the Linear Dependence Index (LDI_k) for a continuous-time dynamical system.

        Parameters
        ----------
        u : numeric_like_t
            Initial condition of the system. It must define a 1D state vector whose
            length matches the system dimension.
        total_time : numeric_t
            Total integration time.
        k : int_t
            Number of deviation vectors used in the computation.
        parameters : numeric_like_t | None, optional
            Parameters of the system. If `None`, the parameters stored in the
            instance are used.
        transient_time : numeric_t | None, optional
            Initial integration time discarded before computing LDI.
        return_history : bool, optional
            If `True`, return the time evolution of LDI.
        seed : int_t, optional
            Seed used to initialize the deviation vectors.
        threshold : numeric_t, optional
            Early stopping threshold. If LDI becomes smaller than `threshold`,
            the computation is terminated.
        endpoint : bool, optional
            Whether to include the endpoint `time = total_time`.

        Returns
        -------
        NDArray[np.float64]
            - If `return_history=False`, returns an array containing the final time
              and the final LDI value.
            - If `return_history=True`, returns the time history of LDI.

        Raises
        ------
        ValueError
            - If the Jacobian is not available.
            - If `u` does not match the system dimension.
            - If the number of parameters does not match the expected number.
            - If `total_time`, `transient_time`, or `threshold` is negative.
            - If `k < 2`.
            - If `k > system_dimension`.
        TypeError
            - If `total_time`, `transient_time`, or `threshold` is not a valid real number.
            - If `k` or `seed` is not an integer.
        """
        if self.__jacobian is None:
            raise ValueError("Jacobian function is required to compute LDI")

        u = validate_initial_conditions(
            u, self.__system_dimension, allow_ensemble=False
        )

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        else:
            parameters = validate_parameters(parameters, self.__number_of_parameters)

        transient_time, total_time = validate_times(transient_time, total_time)

        validate_positive(k, "k", Integral)
        if k < 2:
            raise ValueError("k must be >= 2")
        if k > self.__system_dimension:
            raise ValueError("k must be <= system_dimension")

        validate_non_negative(threshold, "threshold", Real)
        validate_non_negative(seed, "seed", Integral)

        threshold = np.float64(threshold)

        time_step = self.__get_initial_time_step(u, parameters)

        if endpoint:
            total_time += time_step

        result = ldi_k(
            u=u,
            parameters=parameters,
            total_time=total_time,
            equations_of_motion=self.__equations_of_motion,
            jacobian=self.__jacobian,
            number_deviation_vectors=int(k),
            transient_time=transient_time,
            time_step=time_step,
            atol=self.__atol,
            rtol=self.__rtol,
            integrator=self.__integrator_func,
            return_history=return_history,
            seed=int(seed),
            threshold=threshold,
        )

        if return_history:
            return result

        return result[0]

    def GALI(
        self,
        u: numeric_like_t,
        total_time: numeric_t,
        k: int_t,
        parameters: numeric_like_t | None = None,
        transient_time: numeric_t | None = None,
        return_history: bool = False,
        method: str = "QR",
        seed: int_t = 13,
        threshold: numeric_t = 1e-16,
        endpoint: bool = True,
    ) -> NDArray[np.float64]:
        """
        Calculate the Generalized Alignment Index (GALI_k) for a continuous-time
        dynamical system.

        Parameters
        ----------
        u : numeric_like_t
            Initial condition of the system. It must define a 1D state vector whose
            length matches the system dimension.
        total_time : numeric_t
            Total integration time.
        k : int_t
            Number of deviation vectors used in the computation.
        parameters : numeric_like_t | None, optional
            Parameters of the system. If `None`, the parameters stored in the
            instance are used.
        transient_time : numeric_t | None, optional
            Initial integration time discarded before computing GALI.
        return_history : bool, optional
            If `True`, return the time evolution of GALI.
        method : str, optional
            Method used to compute GALI. Supported options are:
            - `"DET"`   : determinant of the Gram matrix
            - `"QR"`    : custom QR routine
            - `"QR_HH"` : `numpy.linalg.qr`
        seed : int_t, optional
            Seed used to initialize the deviation vectors.
        threshold : numeric_t, optional
            Early stopping threshold. If GALI becomes smaller than `threshold`,
            the computation is terminated.
        endpoint : bool, optional
            Whether to include the endpoint `time = total_time`.

        Returns
        -------
        NDArray[np.float64]
            - If `return_history=False`, returns an array containing the final time
              and the final GALI value.
            - If `return_history=True`, returns the time history of GALI.

        Raises
        ------
        ValueError
            - If the Jacobian is not available.
            - If `u` does not match the system dimension.
            - If the number of parameters does not match the expected number.
            - If `total_time`, `transient_time`, or `threshold` is negative.
            - If `k < 2`.
            - If `k > system_dimension`.
            - If `method` is not one of `"DET"`, `"QR"`, or `"QR_HH"`.
        TypeError
            - If `total_time`, `transient_time`, or `threshold` is not a valid real number.
            - If `k` or `seed` is not an integer.
            - If `method` is not a string.
        """
        if self.__jacobian is None:
            raise ValueError("Jacobian function is required to compute GALI")

        u = validate_initial_conditions(
            u, self.__system_dimension, allow_ensemble=False
        )

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        else:
            parameters = validate_parameters(parameters, self.__number_of_parameters)

        transient_time, total_time = validate_times(transient_time, total_time)

        validate_positive(k, "k", Integral)
        if k < 2:
            raise ValueError("k must be >= 2")
        if k > self.__system_dimension:
            raise ValueError("k must be <= system_dimension")

        validate_non_negative(threshold, "threshold", Real)
        validate_non_negative(seed, "seed", Integral)

        if not isinstance(method, str):
            raise TypeError("method must be a string")

        method = method.upper()
        if method not in ("DET", "QR", "QR_HH"):
            raise ValueError("method must be 'DET', 'QR', or 'QR_HH'")

        threshold = np.float64(threshold)

        time_step = self.__get_initial_time_step(u, parameters)

        if endpoint:
            total_time += time_step

        result = gali_k(
            u=u,
            parameters=parameters,
            total_time=total_time,
            equations_of_motion=self.__equations_of_motion,
            jacobian=self.__jacobian,
            number_deviation_vectors=int(k),
            transient_time=transient_time,
            time_step=time_step,
            atol=self.__atol,
            rtol=self.__rtol,
            integrator=self.__integrator_func,
            return_history=return_history,
            method=method,
            seed=int(seed),
            threshold=threshold,
        )

        if return_history:
            return result

        return result[0]

    def recurrence_time_entropy(
        self,
        u: numeric_like_t,
        num_intersections: int_t,
        parameters: numeric_like_t | None = None,
        transient_time: numeric_t | None = None,
        map_type: str = "SM",
        section_index: int_t | None = None,
        section_value: numeric_t | None = None,
        crossing: int_t | None = None,
        sampling_time: numeric_t | None = None,
        maxima_index: int_t | None = None,
        **kwargs: Any,
    ) -> (
        float
        | tuple[float, NDArray[np.float64]]
        | tuple[float, NDArray[np.uint8]]
        | tuple[float, NDArray[np.float64], NDArray[np.uint8]]
        | tuple[float, NDArray[np.float64], NDArray[np.float64]]
        | tuple[float, NDArray[np.uint8], NDArray[np.float64]]
        | tuple[float, NDArray[np.float64], NDArray[np.uint8], NDArray[np.float64]]
    ):
        """
        Compute the recurrence time entropy (RTE) for a continuous-time dynamical system.

        Parameters
        ----------
        u : numeric_like_t
            Initial condition. It must define a 1D state vector whose length matches
            the system dimension.
        num_intersections : int_t
            Number of reduced-map points used in the recurrence analysis.
        parameters : numeric_like_t | None, optional
            Parameters of the system. If `None`, the parameters stored in the
            instance are used.
        transient_time : numeric_t | None, optional
            Initial integration time discarded before generating the reduced map.
        map_type : str, optional
            Reduced map used to generate the data:
            - `"PS"` for Poincaré section
            - `"SM"` for stroboscopic map
            - `"MM"` for maxima map
        section_index : int_t | None, optional
            Coordinate index defining the Poincaré section when `map_type="PS"`.
        section_value : numeric_t | None, optional
            Section value for the Poincaré section when `map_type="PS"`.
        crossing : int_t | None, optional
            Crossing orientation for the Poincaré section when `map_type="PS"`.
        sampling_time : numeric_t | None, optional
            Sampling interval for the stroboscopic map when `map_type="SM"`.
        maxima_index : int_t | None, optional
            Index of the state variable whose maxima are used when `map_type="MM"`.
        **kwargs : Any
            Additional keyword arguments passed to `RTEConfig`, including:
            - `metric`
            - `std_metric`
            - `threshold`
            - `threshold_mode`
            - `threshold_std`
            - `lmin`
            - `return_final_state`
            - `return_recmat`
            - `return_p`

        Returns
        -------
        float or tuple
            The RTE value, optionally followed by:
            - the final reduced-map point
            - the recurrence matrix
            - the white-vertical-line distribution

        Raises
        ------
        ValueError
            - If the initial condition does not match the system dimension.
            - If the number of parameters does not match the expected number.
            - If `num_intersections` is negative.
            - If `map_type` is not one of `"PS"`, `"SM"`, or `"MM"`.
            - If required arguments for the selected map type are missing.
            - If `section_index` or `maxima_index` is outside the valid coordinate range.
            - If `crossing` is not one of `-1`, `0`, or `1`.
        TypeError
            - If `map_type` is not a string.
            - If `section_value`, `sampling_time`, or `transient_time` is not a valid real number.
            - If `section_index`, `crossing`, `maxima_index`, or `num_intersections` is not an integer.
        """
        u = validate_initial_conditions(
            u, self.__system_dimension, allow_ensemble=False
        )

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        else:
            parameters = validate_parameters(parameters, self.__number_of_parameters)

        validate_non_negative(num_intersections, "num_intersections", Integral)
        transient_time, _ = validate_times(transient_time, np.float64(1.0e300))

        if not isinstance(map_type, str):
            raise TypeError("map_type must be a string")

        map_type = map_type.upper()

        if map_type == "PS":
            if section_index is None or section_value is None or crossing is None:
                raise ValueError(
                    "When map_type='PS', section_index, section_value, and crossing must be provided"
                )

            validate_non_negative(section_index, "section_index", Integral)
            if section_index >= self.__system_dimension:
                raise ValueError("section_index must be in [0, system_dimension)")

            if isinstance(section_value, bool) or not isinstance(section_value, Real):
                raise TypeError("section_value must be a valid real number")

            if isinstance(crossing, bool) or not isinstance(crossing, Integral):
                raise TypeError("crossing must be an integer")
            if crossing not in (-1, 0, 1):
                raise ValueError("crossing must be -1, 0, or 1")

        elif map_type == "SM":
            if sampling_time is None:
                raise ValueError("When map_type='SM', sampling_time must be provided")
            validate_positive(sampling_time, "sampling_time", Real)

        elif map_type == "MM":
            if maxima_index is None:
                raise ValueError("When map_type='MM', maxima_index must be provided")
            validate_non_negative(maxima_index, "maxima_index", Integral)
            if maxima_index >= self.__system_dimension:
                raise ValueError("maxima_index must be in [0, system_dimension)")

        else:
            raise ValueError("map_type must be 'PS', 'SM', or 'MM'")

        time_step = self.__get_initial_time_step(u, parameters)

        return recurrence_time_entropy_core(
            u=u,
            parameters=parameters,
            num_points=int(num_intersections),
            transient_time=transient_time,
            equations_of_motion=self.__equations_of_motion,
            time_step=time_step,
            atol=self.__atol,
            rtol=self.__rtol,
            integrator=self.__integrator_func,
            map_type=map_type,
            section_index=None if section_index is None else int(section_index),
            section_value=None if section_value is None else np.float64(section_value),
            crossing=None if crossing is None else int(crossing),
            sampling_time=None if sampling_time is None else np.float64(sampling_time),
            maxima_index=None if maxima_index is None else int(maxima_index),
            **kwargs,
        )

    def hurst_exponent(
        self,
        u: numeric_like_t,
        num_intersections: int_t,
        parameters: numeric_like_t | None = None,
        transient_time: numeric_t | None = None,
        wmin: int_t = 2,
        map_type: str = "SM",
        section_index: int_t | None = None,
        section_value: numeric_t | None = None,
        crossing: int_t | None = None,
        sampling_time: numeric_t | None = None,
        maxima_index: int_t | None = None,
    ) -> NDArray[np.float64]:
        """
        Estimate the Hurst exponent from a reduced map generated by the continuous-time system.

        Parameters
        ----------
        u : numeric_like_t
            Initial condition. It must define a 1D state vector whose length matches
            the system dimension.
        num_intersections : int_t
            Number of reduced-map points used in the Hurst analysis.
        parameters : numeric_like_t | None, optional
            Parameters of the system. If `None`, the parameters stored in the
            instance are used.
        transient_time : numeric_t | None, optional
            Initial integration time discarded before generating the reduced map.
        wmin : int_t, optional
            Minimum window size used in the rescaled-range calculation.
        map_type : str, optional
            Reduced map used to generate the data:
            - `"PS"` for Poincaré section
            - `"SM"` for stroboscopic map
            - `"MM"` for maxima map
        section_index : int_t | None, optional
            Coordinate index defining the Poincaré section when `map_type="PS"`.
        section_value : numeric_t | None, optional
            Section value for the Poincaré section when `map_type="PS"`.
        crossing : int_t | None, optional
            Crossing orientation for the Poincaré section when `map_type="PS"`.
        sampling_time : numeric_t | None, optional
            Sampling interval for the stroboscopic map when `map_type="SM"`.
        maxima_index : int_t | None, optional
            Index of the state variable whose maxima are used when `map_type="MM"`.

        Returns
        -------
        NDArray[np.float64]
            Estimated Hurst exponent values for the reduced-map coordinates.

        Raises
        ------
        ValueError
            - If the initial condition does not match the system dimension.
            - If the number of parameters does not match the expected number.
            - If `num_intersections` is negative.
            - If `map_type` is not one of `"PS"`, `"SM"`, or `"MM"`.
            - If required arguments for the selected map type are missing.
            - If `section_index` or `maxima_index` is outside the valid coordinate range.
            - If `crossing` is not one of `-1`, `0`, or `1`.
            - If `sampling_time` is not positive.
            - If `wmin < 2` or `wmin >= num_intersections // 2`.
        TypeError
            - If `map_type` is not a string.
            - If `section_value`, `sampling_time`, or `transient_time` is not a valid real number.
            - If `section_index`, `crossing`, `maxima_index`, `wmin`, or `num_intersections`
              is not an integer.
        """
        u = validate_initial_conditions(
            u, self.__system_dimension, allow_ensemble=False
        )

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        else:
            parameters = validate_parameters(parameters, self.__number_of_parameters)

        validate_non_negative(num_intersections, "num_intersections", Integral)
        validate_positive(wmin, "wmin", Integral)
        transient_time, _ = validate_times(transient_time, np.float64(1.0e300))

        if not isinstance(map_type, str):
            raise TypeError("map_type must be a string")

        map_type = map_type.upper()

        if map_type == "PS":
            if section_index is None or section_value is None or crossing is None:
                raise ValueError(
                    "When map_type='PS', section_index, section_value, and crossing must be provided"
                )

            validate_non_negative(section_index, "section_index", Integral)
            if section_index >= self.__system_dimension:
                raise ValueError("section_index must be in [0, system_dimension)")

            if isinstance(section_value, bool) or not isinstance(section_value, Real):
                raise TypeError("section_value must be a valid real number")

            if isinstance(crossing, bool) or not isinstance(crossing, Integral):
                raise TypeError("crossing must be an integer")
            if crossing not in (-1, 0, 1):
                raise ValueError("crossing must be -1, 0, or 1")

        elif map_type == "SM":
            if sampling_time is None:
                raise ValueError("When map_type='SM', sampling_time must be provided")
            validate_positive(sampling_time, "sampling_time", Real)

        elif map_type == "MM":
            if maxima_index is None:
                raise ValueError("When map_type='MM', maxima_index must be provided")
            validate_non_negative(maxima_index, "maxima_index", Integral)
            if maxima_index >= self.__system_dimension:
                raise ValueError("maxima_index must be in [0, system_dimension)")

        else:
            raise ValueError("map_type must be 'PS', 'SM', or 'MM'")

        if wmin < 2 or wmin >= num_intersections // 2:
            raise ValueError(
                f"`wmin` must be an integer >= 2 and < num_intersections // 2. Got {wmin}."
            )

        time_step = self.__get_initial_time_step(u, parameters)

        return hurst_exponent_wrapped(
            u=u,
            parameters=parameters,
            num_points=int(num_intersections),
            equations_of_motion=self.__equations_of_motion,
            time_step=time_step,
            atol=self.__atol,
            rtol=self.__rtol,
            integrator=self.__integrator_func,
            map_type=map_type,
            section_index=None if section_index is None else int(section_index),
            section_value=None if section_value is None else np.float64(section_value),
            crossing=None if crossing is None else int(crossing),
            sampling_time=None if sampling_time is None else np.float64(sampling_time),
            maxima_index=None if maxima_index is None else int(maxima_index),
            wmin=int(wmin),
            transient_time=transient_time,
        )
