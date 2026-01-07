# continuous_dynamical_systems.py

# Copyright (C) 2025 Matheus Rolim Sales
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
from typing import Any, Callable, Dict, List, Optional, Sequence, Union, Tuple
from IPython.display import Math

import numpy as np
from numba.core.errors import NumbaExperimentalFeatureWarning
from numpy.typing import NDArray

from pynamicalsys.common.utils import householder_qr, qr

from pynamicalsys.continuous_time.chaotic_indicators import (
    LDI,
    SALI,
    GALI,
    lyapunov_exponents,
    maximum_lyapunov_exponent,
    recurrence_time_entropy,
    hurst_exponent_wrapped,
    compute_clvs,
    clv_angles,
)

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

from pynamicalsys.continuous_time.numerical_integrators import (
    estimate_initial_step,
    rk4_step_wrapped,
    rk45_step_wrapped,
)

from pynamicalsys.continuous_time.trajectory_analysis import (
    evolve_system,
    generate_maxima_map,
    generate_trajectory,
    ensemble_trajectories,
    generate_poincare_section,
    ensemble_poincare_section,
    generate_stroboscopic_map,
    ensemble_stroboscopic_map,
    generate_maxima_map,
    ensemble_maxima_map,
    basin_of_attraction,
)

from pynamicalsys.continuous_time.validators import (
    validate_times,
)

from pynamicalsys.common.validators import (
    validate_initial_conditions,
    validate_parameters,
    validate_non_negative,
    validate_clv_pairs,
    validate_clv_subspaces,
)


class ContinuousDynamicalSystem:
    """Class representing a continuous-time dynamical system with various models and methods for analysis.

    This class allows users to work with predefined dynamical models or with user-provided equations of motion, compute trajectories, Lyapunov exponents and more dynamical analyses.

    Parameters
    ----------
    model : str, optional
        Name of the predefined model to use (e.g. "lorenz system").
    equations_of_motion : callable, optional
        Custom function that describes the equations of motion with signature f(time, state, parameters) -> array_like. If provided, `model` must be None.
    jacobian : callable, optional
        Custom function that describes the Jacobian matrix of the system with signature J(time, state, parameters) -> array_like
    system_dimension : int, optional
        Dimension of the system (number of equations). Required if using custom equations of motion and not a predefined model.
    number_of_parameters : int, optional
        Number of parameters of the system. Required if using custom equations of motion and not a predefined model.

    Raises
    ------
    ValueError
        - If neither model nor equations_of_motion is provided, or if provided model is not implemented.
        - If `system_dimension` is negative.
        - If `number_of_parameters` is negative.

    TypeError
        - If `equations_of_motion` or `jacobian` are not callable.
        - If `system_dimension` or `number_of_parameters` are not valid integers.

    Notes
    -----
    - When providing custom functions, either provide both `equations_of_motion` and `jacobian`, or just the `equations_of_motion`.
    - When providing custom functions, the equations of motion functions signature should be f(time, u, parameters) -> NDArray[np.float64].
    - The class supports various predefined models, such as the Lorenz and Rössler system.
    - The available models can be queried using the 'available_models' class method.

    Examples
    --------
    >>> from pynamicalsys import ContinuousDynamicalSystem as cds
    >>> #  Using predefined model
    >>> ds = cds(model="lorenz system")
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
        model: Optional[str] = None,
        equations_of_motion: Optional[Callable] = None,
        jacobian: Optional[Callable] = None,
        system_dimension: Optional[int] = None,
        parameters: Optional[Sequence] = None,
        number_of_parameters: Optional[int] = None,
    ) -> None:

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
            self.__system_dimension = model_info["dimension"]
            self.__parameters = None
            self.__number_of_parameters = model_info["number_of_parameters"]

            if jacobian is not None:
                self.__jacobian = jacobian

        elif (
            equations_of_motion is not None
            and system_dimension is not None
            and (parameters is not None or number_of_parameters is not None)
        ):
            self.__equations_of_motion = equations_of_motion
            self.__jacobian = jacobian

            validate_non_negative(system_dimension, "system_dimension", Integral)
            if number_of_parameters is not None:
                validate_non_negative(
                    number_of_parameters, "number_of_parameters", Integral
                )

            self.__system_dimension = system_dimension
            self.__parameters = parameters
            if self.__parameters is not None:
                self.__number_of_parameters = len(self.__parameters)
                self.__parameters = validate_parameters(
                    self.__parameters, self.__number_of_parameters
                )
            else:
                self.__number_of_parameters = number_of_parameters

            if not callable(self.__equations_of_motion):
                raise TypeError("Custom mapping must be callable")

            if self.__jacobian is not None and not callable(self.__jacobian):
                raise TypeError("Custom Jacobian must be callable or None")
        else:
            raise ValueError(
                "Must specify either a model name or custom system function with its dimension and number of paramters."
            )

        self.__integrator = "rk4"
        self.__integrator_func = rk4_step_wrapped
        self.__time_step = 1e-2
        self.__atol = 1e-6
        self.__rtol = 1e-3

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

    def integrator(self, integrator, time_step=1e-2, atol=1e-6, rtol=1e-3):
        """Define the integrator.

        Parameters
        ----------
        integrator : str
            The integrator name. Available options are 'rk4' and 'rk45'
        time_step : float, optional
            The integration time step when `integrator='rk4'`, by default 1e-2
        atol : float, optional
            The absolute tolerance used when `integrator='rk45'`, by default 1e-6
        rtol : float, optional
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

    def get_parameters(self) -> NDArray[np.float64]:
        """
        Return the current parameter vector of the dynamical system.

        Returns
        -------
        ndarray of float64, shape (P,)
            The parameter vector currently stored in the system.
        """
        return self.__parameters

    def __get_initial_time_step(self, u, parameters):
        if self.integrator_info["estimate_initial_step"]:
            time_step = estimate_initial_step(
                0.0,
                u,
                parameters,
                self.__equations_of_motion,
                atol=self.__atol,
                rtol=self.__rtol,
            )
        else:
            time_step = self.__time_step

        return time_step

    def evolve_system(
        self,
        u: Union[NDArray[np.float64], Sequence[float]],
        total_time: float,
        parameters: Union[None, Sequence[float], NDArray[np.float64]] = None,
    ) -> NDArray[np.float64]:
        """
        Evolve the dynamical system from the given initial conditions over a specified time period.

        Parameters
        ----------
        u : Union[NDArray[np.float64], Sequence[float]]
            Initial conditions of the system. Must match the system's dimension.
        total_time : float
            Total time over which to evolve the system.
        parameters : Union[None, Sequence[float], NDArray[np.float64]], optional
            Parameters of the system, by default None. Can be a scalar, a sequence of floats or a numpy array.

        Returns
        -------
        result : NDArray[np.float64]
            The state of the system at time = total_time.

        Raises
        ------
        ValueError
            - If the initial condition is not valid, i.e., if the dimensions do not match.
            - If the number of parameters does not match.
            - If `parameters` is not a scalar, 1D list, or 1D array.
        TypeError
            - If `total_time` is not a valid number.

        Examples
        --------
        >>> from pynamicalsys import ContinuousDynamicalSystem as cds
        >>> ds = cds(model="lorenz system")
        >>> ds.integrator("rk4", time_step=0.01)
        >>> parameters = [10, 28, 8/3]
        >>> u = [1.0, 1.0, 1.0]
        >>> total_time = 10
        >>> ds.evolve_system(u, total_time, parameters=parameters)
        >>> ds.integrator("rk45", atol=1e-8, rtol=1e-6)
        >>> ds.evolve_system(u, total_time, parameters=parameters)
        """

        u = validate_initial_conditions(u, self.__system_dimension)
        u = u.copy()

        if parameters is None:
            parameters = self.__parameters
        else:
            parameters = validate_parameters(parameters, self.__number_of_parameters)

        _, total_time = validate_times(1, total_time)

        time_step = self.__get_initial_time_step(u, parameters)

        total_time += time_step

        return evolve_system(
            u,
            parameters,
            total_time,
            self.__equations_of_motion,
            time_step=time_step,
            atol=self.__atol,
            rtol=self.__rtol,
            integrator=self.__integrator_func,
        )

    def trajectory(
        self,
        u: Union[NDArray[np.float64], Sequence[float]],
        total_time: float,
        parameters: Union[None, Sequence[float], NDArray[np.float64]] = None,
        transient_time: Optional[float] = None,
    ) -> NDArray[np.float64]:
        """
        Compute the trajectory of the dynamical system over a specified time period.

        Parameters
        ----------
        u : Union[NDArray[np.float64], Sequence[float]]
            Initial conditions of the system. Must match the system's dimension.
        total_time : float
            Total time over which to evolve the system (including transient).
        parameters : Union[None, Sequence[float], NDArray[np.float64]], optional
            Parameters of the system, by default None. Can be a scalar, a sequence of floats or a numpy array.
        transient_time : float
            Initial time to discard.

        Returns
        -------
        result : NDArray[np.float64]
            The trajectory of the system.

            - For a single initial condition (u.ndim = 1), return a 2D array of shape (number_of_steps, neq + 1), where the first column is the time samples and the remaining columns are the coordinates of the system
            - For multiple initial conditions (u.ndim = 2), return a 3D array of shape (num_ic, number_of_steps, neq + 1).

        Raises
        ------
        ValueError
            - If the initial condition is not valid, i.e., if the dimensions do not match.
            - If the number of parameters does not match.
            - If `parameters` is not a scalar, 1D list, or 1D array.
        TypeError
            - If `total_time` or `transient_time` are not valid numbers.

        Examples
        --------
        >>> from pynamicalsys import ContinuousDynamicalSystem as cds
        >>> ds = cds(model="lorenz system")
        >>> u = [0.1, 0.1, 0.1]  # Initial condition
        >>> parameters = [10, 28, 8/3]
        >>> total_time = 700
        >>> transient_time = 500
        >>> trajectory = ds.trajectory(u, total_time, parameters=parameters, transient_time=transient_time)
        (11000, 4)
        >>> u = [[0.1, 0.1, 0.1],
        ... [0.2, 0.2, 0.2],
        ... [0.3, 0.3, 0.3]]  # Three initial conditions
        >>> trajectories = ds.trajectory(u, total_time, parameters=parameters, transient_time=transient_time)
        (3, 20000, 4)
        """

        u = validate_initial_conditions(u, self.__system_dimension)
        u = u.copy()

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        else:
            parameters = validate_parameters(parameters, self.__number_of_parameters)

        transient_time, total_time = validate_times(transient_time, total_time)

        time_step = self.__get_initial_time_step(u, parameters)

        if u.ndim == 1:
            result = generate_trajectory(
                u,
                parameters,
                total_time,
                self.__equations_of_motion,
                transient_time=transient_time,
                time_step=time_step,
                atol=self.__atol,
                rtol=self.__rtol,
                integrator=self.__integrator_func,
            )
            return np.array(result)
        else:
            return ensemble_trajectories(
                u,
                parameters,
                total_time,
                self.__equations_of_motion,
                transient_time=transient_time,
                time_step=time_step,
                atol=self.__atol,
                rtol=self.__rtol,
                integrator=self.__integrator_func,
            )

    def poincare_section(
        self,
        u: Union[NDArray[np.float64], Sequence[float]],
        num_intersections: int,
        section_index: int,
        section_value: float,
        parameters: Union[None, Sequence[float], NDArray[np.float64]] = None,
        transient_time: Optional[float] = None,
        crossing: int = 1,
    ) -> NDArray[np.float64]:
        """
        Compute the Poincaré section of the dynamical system for given initial conditions.

        A Poincaré section records the points where a trajectory intersects a chosen hypersurface
        in phase space (e.g. x = constant). This reduces a continuous flow to a lower-dimensional
        map, making it easier to identify periodic orbits, quasi-periodic motion, or chaotic
        structures.

        Parameters
        ----------
        u : Union[NDArray[np.float64], Sequence[float]]
            Initial conditions of the system. Must match the system's dimension.
        num_intersections : int
            Number of intersections to record in the Poincaré section.
        section_index : int
            Index of the coordinate to define the Poincaré section (0-based).
        section_value : float
            Value of the coordinate at which the section is defined.
        parameters : Union[None, Sequence[float], NDArray[np.float64]], optional
            Parameters of the system, by default None. Can be a scalar, a sequence of floats, or a numpy array.
        transient_time : float, optional
            Initial time to discard before recording the section.
        crossing : int, default=1
            Specifies the type of crossing to consider:
            - 1 : positive crossing (from below to above section_value)
            - -1 : negative crossing (from above to below section_value)
            - 0 : all crossings

        Returns
        -------
        result : NDArray[np.float64]
            The Poincaré section points.

            - For a single initial condition (u.ndim = 1), returns a 2D array of shape
              (num_intersections, neq), where each row is a system state at a crossing.
            - For multiple initial conditions (u.ndim = 2), returns a 3D array of shape
              (num_ic, num_intersections, neq).

        Raises
        ------
        ValueError
            - If the initial condition dimension does not match the system dimension.
            - If the number of parameters does not match the system.
            - If section_index is larger than the system dimension.
        TypeError
            - If `section_value` is not a real number.
            - If `num_intersections` or `transient_time` are not valid numbers.

        Examples
        --------
        >>> from pynamicalsys import ContinuousDynamicalSystem as cds
        >>> ds = cds(model="lorenz system")
        >>> u = [0.1, 0.1, 0.1]  # Initial condition
        >>> parameters = [10, 28, 8/3]
        >>> num_intersections = 500
        >>> section_index = 2
        >>> section_value = 25.0
        >>> ps = ds.poincare_section(u, num_intersections, section_index, section_value, parameters=parameters)
        (500, 3)
        >>> u = [[0.1, 0.1, 0.1],
        ...      [0.2, 0.2, 0.2]]  # Two initial conditions
        >>> ps_ensemble = ds.poincare_section(u, num_intersections, section_index, section_value, parameters=parameters)
        (2, 500, 3)
        """

        u = validate_initial_conditions(u, self.__system_dimension)
        u = u.copy()

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        else:
            parameters = validate_parameters(parameters, self.__number_of_parameters)

        validate_non_negative(num_intersections, "num_intersections", Integral)

        validate_non_negative(section_index, "section_index", Integral)
        if section_index > self.__system_dimension:
            raise ValueError("section_index must be smaller than the sustem_dimension")

        if not isinstance(section_value, Real):
            raise TypeError("section_value must be a valid real number")

        if transient_time is not None:
            validate_non_negative(transient_time, "transient_time", Real)

        if not isinstance(crossing, Integral):
            raise TypeError("crossing must be an integer number")
        elif crossing not in [-1, 0, 1]:
            raise ValueError(
                "crossing must be -1 (downward crossings), 0 (all crossings), or 1 (upward crossing)"
            )

        time_step = self.__get_initial_time_step(u, parameters)

        if u.ndim == 1:
            return generate_poincare_section(
                u,
                parameters,
                num_intersections,
                self.__equations_of_motion,
                transient_time,
                time_step,
                self.__atol,
                self.__rtol,
                self.__integrator_func,
                section_index,
                section_value,
                crossing,
            )
        else:
            return ensemble_poincare_section(
                u,
                parameters,
                num_intersections,
                self.__equations_of_motion,
                transient_time,
                time_step,
                self.__atol,
                self.__rtol,
                self.__integrator_func,
                section_index,
                section_value,
                crossing,
            )

    def stroboscopic_map(
        self,
        u: Union[NDArray[np.float64], Sequence[float]],
        num_samples: int,
        sampling_time: float,
        parameters: Union[None, Sequence[float], NDArray[np.float64]] = None,
        transient_time: Optional[float] = None,
    ) -> NDArray[np.float64]:
        """
        Compute the stroboscopic map of the dynamical system for given initial conditions.

        A stroboscopic map samples the state of a time-periodic or driven system at fixed time
        intervals (typically one driving period). This converts the continuous-time dynamics
        into a discrete-time sequence that highlights periodicity, phase locking, and
        bifurcations.

        Parameters
        ----------
        u : Union[NDArray[np.float64], Sequence[float]]
            Initial conditions of the system. Must match the system's dimension.
        num_samples : int
            Number of samples to record in the stroboscopic map.
        sampling_time : float
            Time interval between consecutive samples.
        parameters : Union[None, Sequence[float], NDArray[np.float64]], optional
            Parameters of the system, by default None. Can be a scalar, a sequence of floats, or a numpy array.
        transient_time : float, optional
            Initial time to discard before recording the map.

        Returns
        -------
        result : NDArray[np.float64]
            The stroboscopic map points.

            - For a single initial condition (u.ndim = 1), returns a 2D array of shape
              (num_samples, neq + 1), where the first column is the time and the remaining
              columns are the system coordinates at each sampled time.
            - For multiple initial conditions (u.ndim = 2), returns a 3D array of shape
              (num_ic, num_samples, neq + 1).

        Raises
        ------
        ValueError
            - If the initial condition dimension does not match the system dimension.
            - If the number of parameters does not match the system.
        TypeError
            - If `num_samples` or `sampling_time` are not valid numbers.
            - If `transient_time` is provided and is not a valid number.

        Examples
        --------
        >>> from pynamicalsys import ContinuousDynamicalSystem as cds
        >>> ds = cds(model="lorenz system")
        >>> u = [0.1, 0.1, 0.1]  # Initial condition
        >>> parameters = [10, 28, 8/3]
        >>> num_samples = 500
        >>> sampling_time = 0.1
        >>> smap = ds.stroboscopic_map(u, num_samples, sampling_time, parameters=parameters)
        (500, 4)
        >>> u = [[0.1, 0.1, 0.1],
        ...      [0.2, 0.2, 0.2]]  # Two initial conditions
        >>> smap_ensemble = ds.stroboscopic_map(u, num_samples, sampling_time, parameters=parameters)
        (2, 500, 4)
        """

        u = validate_initial_conditions(u, self.__system_dimension)
        u = u.copy()

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        else:
            parameters = validate_parameters(parameters, self.__number_of_parameters)

        validate_non_negative(num_samples, "num_samples", Integral)

        validate_non_negative(sampling_time, "sampling_time", Real)

        if transient_time is not None:
            validate_non_negative(transient_time, "transient_time", Real)

        time_step = self.__get_initial_time_step(u, parameters)

        if u.ndim == 1:
            return generate_stroboscopic_map(
                u,
                parameters,
                num_samples,
                sampling_time,
                self.__equations_of_motion,
                transient_time,
                time_step,
                self.__atol,
                self.__rtol,
                self.__integrator_func,
            )
        else:
            return ensemble_stroboscopic_map(
                u,
                parameters,
                num_samples,
                sampling_time,
                self.__equations_of_motion,
                transient_time,
                time_step,
                self.__atol,
                self.__rtol,
                self.__integrator_func,
            )

    def maxima_map(
        self,
        u: Union[NDArray[np.float64], Sequence[float]],
        num_points: int,
        maxima_index: int,
        parameters: Union[None, Sequence[float], NDArray[np.float64]] = None,
        transient_time: Optional[float] = None,
    ) -> NDArray[np.float64]:
        """
        Compute the maxima map of the dynamical system for given initial conditions.

        A maxima map records the local maxima of a chosen system variable along the trajectory.
        By plotting successive maxima, one obtains a discrete return map that reveals
        oscillation amplitudes, period-doubling cascades, and other nonlinear behaviours.

        Parameters
        ----------
        u : Union[NDArray[np.float64], Sequence[float]]
            Initial conditions of the system. Must match the system's dimension.
        num_points : int
            Number of points to record in the maxima map.
        maxima_index : int
            Index of the variable whose maxima are to be recorded.
        parameters : Union[None, Sequence[float], NDArray[np.float64]], optional
            Parameters of the system, by default None. Can be a scalar, a sequence of floats, or a numpy array.
        transient_time : float, optional
            Initial time to discard before recording the map.

        Returns
        -------
        result : NDArray[np.float64]
            The maxima map points.

            - For a single initial condition (u.ndim = 1), returns a 2D array of shape
              (num_points, neq + 1), where the first column is the time and the remaining
              columns are the system coordinates at each maxima point.
            - For multiple initial conditions (u.ndim = 2), returns a 3D array of shape
              (num_ic, num_points, neq + 1).

        Raises
        ------
        ValueError
            - If the initial condition dimension does not match the system dimension.
            - If the number of parameters does not match the system.
        TypeError
            - If `num_points` or `maxima_index` are not valid numbers.
            - If `transient_time` is provided and is not a valid number.

        Examples
        --------
        >>> from pynamicalsys import ContinuousDynamicalSystem as cds
        >>> ds = cds(model="lorenz system")
        >>> u = [0.1, 0.1, 0.1]  # Initial condition
        >>> parameters = [10, 28, 8/3]
        >>> num_points = 500
        >>> maxima_index = 0
        >>> smap = ds.maxima_map(u, num_points, maxima_index, parameters=parameters)
        >>> smap.shape
        (500, 4)
        >>> u = [[0.1, 0.1, 0.1],
        ...      [0.2, 0.2, 0.2]]  # Two initial conditions
        >>> smap_ensemble = ds.stroboscopic_map(u, num_samples, sampling_time, parameters=parameters)
        >>> smap_ensemble.shape
        (2, 500, 4)
        """

        u = validate_initial_conditions(u, self.__system_dimension)
        u = u.copy()

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        else:
            parameters = validate_parameters(parameters, self.__number_of_parameters)

        validate_non_negative(num_points, "num_samples", Integral)

        validate_non_negative(maxima_index, "maxima_index", Integral)

        if transient_time is not None:
            validate_non_negative(transient_time, "transient_time", Real)

        time_step = self.__get_initial_time_step(u, parameters)

        if u.ndim == 1:
            return generate_maxima_map(
                u,
                parameters,
                num_points,
                maxima_index,
                self.__equations_of_motion,
                transient_time,
                time_step,
                self.__atol,
                self.__rtol,
                self.__integrator_func,
            )
        else:
            return ensemble_maxima_map(
                u,
                parameters,
                num_points,
                maxima_index,
                self.__equations_of_motion,
                transient_time,
                time_step,
                self.__atol,
                self.__rtol,
                self.__integrator_func,
            )

    def basin_of_attraction(
        self,
        u: Union[NDArray[np.float64], Sequence[float]],
        num_intersections: int,
        parameters: Union[None, Sequence[float], NDArray[np.float64]] = None,
        transient_time: Optional[float] = None,
        map_type: str = "SM",
        section_index: Optional[int] = None,
        section_value: Optional[float] = None,
        crossing: Optional[int] = None,
        sampling_time: Optional[float] = None,
        eps: float = 0.05,
        min_samples: int = 1,
    ) -> NDArray[np.int32]:
        """
        Compute the basin of attraction for a dynamical system for a set of initial conditions.

        Parameters
        ----------
        u : Union[NDArray[np.float64], Sequence[float]]
            Initial conditions for the dynamical system.
        num_intersections : int
            Number of intersections (or samples) to use in constructing the map (stroboscopic or Poincaré).
        parameters : Union[None, Sequence[float], NDArray[np.float64]], optional
            System parameters. If None, defaults will be used. Default is None.
        transient_time : float, optional
            Transient time to discard before analyzing the trajectories. Default is None.
        map_type : str, default "SM"
            Type of map to compute:
            - "SM" : stroboscopic map
            - "PS" : Poincaré section
        section_index : int, optional
            Index of the coordinate used for the Poincaré section (required if map_type="PS").
        section_value : float, optional
            Value of the section plane (required if map_type="PS").
        crossing : int, optional
            Crossing direction for Poincaré section:
            - -1 : downward crossings
            - 0  : all crossings
            - 1  : upward crossings
            Required if map_type="PS".
        sampling_time : float, optional
            Sampling time for stroboscopic map (required if map_type="SM").
        eps : float, default 0.05
            The maximum distance between points to be considered in the same cluster (DBSCAN parameter).
        min_samples : int, default 1
            The minimum number of points to form a cluster (DBSCAN parameter).

        Returns
        -------
        NDArray[np.int32]
            Array of integer labels indicating which attractor each initial condition belongs to.
            Label `-1` indicates points classified as noise (not part of any attractor).

        Notes
        -----
        The basin of attraction is determined by first constructing either a stroboscopic map
        or a Poincaré section from the trajectories. Then, the attractors are identified by
        clustering the trajectory centroids using the DBSCAN algorithm from scikit-learn.

        DBSCAN groups points that are close to each other in phase space, with `eps` defining
        the neighborhood radius and `min_samples` specifying the minimum number of points to
        form a cluster. Each cluster corresponds to a distinct attractor, and initial conditions
        whose trajectories end up in the same cluster are considered to belong to the same basin
        of attraction.
        """
        u = validate_initial_conditions(u, self.__system_dimension)
        u = u.copy()

        validate_non_negative(num_intersections, "num_intersections", Integral)

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        else:
            parameters = validate_parameters(parameters, self.__number_of_parameters)

        if transient_time is not None:
            validate_non_negative(transient_time, "transient_time", Real)

        if not isinstance(map_type, str):
            raise TypeError("map_type must a valid string")
        if map_type not in ["SM", "PS"]:
            raise ValueError(
                "map_type must be either SM (stroboscopic map) or PS (Poicaré section)"
            )

        if section_index is not None:
            validate_non_negative(section_index, "section_index", Integral)
            if section_index > self.__system_dimension:
                raise ValueError("section_index must be <= system_dimension")

        if section_value is not None:
            if not isinstance(section_value, Real):
                raise TypeError("section_value must be a valid real number")

        if crossing is not None:
            if not isinstance(crossing, Integral):
                raise TypeError("crossing must be an integer number")
            elif crossing not in [-1, 0, 1]:
                raise ValueError(
                    "crossing must be -1 (downward crossings), 0 (all crossings), or 1 (upward crossing)"
                )

        if sampling_time is not None:
            validate_non_negative(sampling_time, "sampling_time", Real)

        validate_non_negative(eps, "eps", Real)

        validate_non_negative(min_samples, "min_samples", Integral)

        time_step = self.__get_initial_time_step(u, parameters)

        return basin_of_attraction(
            u,
            parameters,
            num_intersections,
            self.__equations_of_motion,
            transient_time,
            time_step,
            self.__atol,
            self.__rtol,
            self.__integrator_func,
            map_type,
            section_index,
            section_value,
            crossing,
            sampling_time,
            eps,
            min_samples,
        )

    def lyapunov(
        self,
        u: Union[NDArray[np.float64], Sequence[float]],
        total_time: float,
        parameters: Union[None, Sequence[float], NDArray[np.float64]] = None,
        transient_time: Optional[float] = None,
        num_exponents: Optional[int] = None,
        return_history: bool = False,
        seed: int = 13,
        log_base: float = np.e,
        method: str = "QR",
        endpoint: bool = True,
    ) -> NDArray[np.float64]:
        """Calculate the Lyapunov exponents of a given dynamical system.

        The Lyapunov exponent is a key concept in the study of dynamical systems. It measures the average rate at which nearby trajectories in the system diverge (or converge) over time. In simple terms, it quantifies how sensitive a system is to initial conditions.

        Parameters
        ----------
        u : Union[NDArray[np.float64], Sequence[float]]
            Initial conditions of the system. Must match the system's dimension.
        total_time : float
            Total time over which to evolve the system (including transient).
        parameters : Union[None, Sequence[float], NDArray[np.float64]], optional
            Parameters of the system, by default None. Can be a scalar, a sequence of floats or a numpy array.
        transient_time : Optional[float], optional
            Transient time, i.e., the time to discard before calculating the Lyapunov exponents, by default None.
        num_exponents : Optional[int], optional
            The number of Lyapunov exponents to be calculated, by default None. If None, the method calculates the whole spectrum.
        return_history : bool, optional
            Whether to return or not the Lyapunov exponents history in time, by default False.
        seed : int, optional
            The seed to randomly generate the deviation vectors, by default 13.
        log_base : int, optional
            The base of the logarithm function, by default np.e, i.e., natural log.
        method : str, optional
            The method used to calculate the QR decomposition, by default "QR". Set to "QR_HH" to use Householder reflections.
        endpoint : bool, optional
            Whether to include the endpoint time = total_time in the calculation, by default True.

        Returns
        -------
        NDArray[np.float64]
            The Lyapunov exponents.

            - If `return_history = False`, return the Lyapunov exponents' final value.
            - If `return_history = True`, return the time series of each exponent together with the time samples.
            - If `sample_times` is provided, return the Lyapunov exponents at the specified times.

        Raises
        ------
        ValueError
            - If the Jacobian function is not provided.
            - If the initial condition is not valid, i.e., if the dimensions do not match.
            - If the number of parameters does not match.
            - If `parameters` is not a scalar, 1D list, or 1D array.
        TypeError
            - If `method` is not a string.
            - If `total_time`, `transient_time`, or `log_base` are not valid numbers.
            - If `num_exponents` is not an positive integer.
            - If `seed` is not an integer.

        Notes
        -----
        - By default, the method uses the modified Gram-Schimdt algorithm to perform the QR decomposition. If your problem requires a higher numerical stability (e.g. large-scale problem), you can set `method=QR_HH` to use Householder reflections instead.

        Examples
        --------
        >>> from pynamicalsys import ContinuousDynamicalSystem as cds
        >>> ds = cds(model="lorenz system")
        >>> u = [0.1, 0.1, 0.1]
        >>> total_time = 10000
        >>> transient_time = 5000
        >>> parameters = [16.0, 45.92, 4.0]
        >>> ds.lyapunov(u, total_time, parameters=parameters, transient_time=transient_time)
        array([ 1.49885208e+00, -1.65186396e-04, -2.24977688e+01])
        >>> ds.lyapunov(u, total_time, parameters=parameters, transient_time=transient_time, num_exponents=2)
        array([1.49873694e+00, 1.31950729e-04])
        >>> ds.lyapunov(u, total_time, parameters=parameters, transient_time=transient_time, log_base=2, method="QR_HH")
        array([ 2.16664847e+00, -6.80920729e-04, -3.24625604e+01])
        """

        if self.__jacobian is None:
            raise ValueError(
                "Jacobian function is required to compute Lyapunov exponents"
            )

        u = validate_initial_conditions(
            u, self.__system_dimension, allow_ensemble=False
        )
        u = u.copy()

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        else:
            parameters = validate_parameters(parameters, self.__number_of_parameters)

        transient_time, total_time = validate_times(transient_time, total_time)

        time_step = self.__get_initial_time_step(u, parameters)

        if num_exponents is None:
            num_exponents = self.__system_dimension
        elif num_exponents > self.__system_dimension:
            raise ValueError("num_exponents must be <= system_dimension")
        else:
            validate_non_negative(num_exponents, "num_exponents", Integral)

        if endpoint:
            total_time += time_step

        if not isinstance(method, str):
            raise TypeError("method must be a string")

        method = method.upper()
        if method == "QR":
            qr_func = qr
        elif method == "QR_HH":
            qr_func = householder_qr
        else:
            raise ValueError("method must be QR or QR_HH")

        validate_non_negative(log_base, "log_base", Real)
        if log_base == 1:
            raise ValueError("The logarithm function is not defined with base 1")

        if num_exponents == 1:
            result = maximum_lyapunov_exponent(
                u,
                parameters,
                total_time,
                self.__equations_of_motion,
                self.__jacobian,
                transient_time,
                time_step,
                self.__atol,
                self.__rtol,
                self.__integrator_func,
                return_history,
                seed,
            )
        else:
            result = lyapunov_exponents(
                u,
                parameters,
                total_time,
                self.__equations_of_motion,
                self.__jacobian,
                num_exponents,
                transient_time=transient_time,
                time_step=time_step,
                atol=self.__atol,
                rtol=self.__rtol,
                integrator=self.__integrator_func,
                return_history=return_history,
                seed=seed,
                QR=qr_func,
            )
        if return_history:
            return np.array(result) / np.log(log_base)
        else:
            return np.array(result[0]) / np.log(log_base)

    def CLV(
        self,
        u: Union[Sequence[float], NDArray[np.float64]],
        total_time: float,
        parameters: Union[
            None, float, Sequence[np.float64], NDArray[np.float64]
        ] = None,
        num_clvs: Optional[int] = None,
        transient_time: float = 0.0,
        warmup_time: float = 0.0,
        tail_time: float = 0.0,
        qr_time_step: Optional[None] = None,
        seed: int = 13,
    ):
        """
        Compute Covariant Lyapunov Vectors (CLVs) along a trajectory of this continuous-time dynamical system.

        The CLVs form a covariant (time-dependent) basis of the tangent space that transforms
        under the linearized flow exactly as the dynamics does. The i-th CLV is associated with
        the i-th Lyapunov exponent (ordered from largest to smallest) and provides the local
        expanding/contracting directions.

        This routine implements a Ginelli-style algorithm for flows:
        (i) a forward QR iteration (with periodic re-orthonormalization every ``qr_time_step``)
        to obtain the orthonormal backward Lyapunov vectors (BLVs), followed by
        (ii) a backward recursion in the triangular coefficient space to recover covariant vectors.

        Parameters
        ----------
        u : array_like, shape (system_dimension,)
            Initial condition.
        total_time : float
            Total integration time over which CLVs are returned. The output contains
            ``N + 1`` sampled time points, where ``N`` is the number of QR sampling blocks
            (roughly ``total_time / qr_time_step``), including the initial state after any transient.
        parameters : None or array_like
            Model parameters. If ``None``, this method uses the system's stored parameters.
        num_clvs : int, optional
            Number of CLVs to compute/return. Defaults to ``system_dimension``.
            Must satisfy ``1 <= num_clvs <= system_dimension``.
        transient_time : float, default=0
            Initial integration time discarded before starting the CLV computation.
            Useful to remove transient behavior and approach the attractor/typical set.
        warmup_time : int, default=0
            Number of forward QR iterations necessary to the tangent basis to converge to the
            tangent basis of the system. Increasing ``warmup_time`` improves convergence of the
            stored orthonormal frames to the backward Lyapunov vectors (BLVs).
        tail_time : int, default=0
            Number of additional forward QR steps performed *after* the main storage window
            to initialize the backward recursion robustly. Increasing ``tail_time`` improves
            convergence of the backward coefficient recursion that produces covariant vectors,
            especially in weakly hyperbolic / nearly tangent regimes. Analogously to the warmup-time,
            but backward instead of forward.
        qr_time_step : float, optional
            Time between successive QR re-orthonormalizations and storage of ``(Q, R)`` factors.
            If ``None``, defaults to the integrator's initial step size returned by
            the system's internal step-size initializer.
        seed : int, default=13
            Seed used to initialize the random upper-triangular matrix in the backward stage.
            The final CLVs should be insensitive to the seed if ``tail_time`` (and the window)
            are large enough.

        Returns
        -------
        clvs : ndarray, shape (N + 1, system_dimension, num_clvs)
            Covariant Lyapunov vectors sampled along the trajectory. At each sampled index ``k``,
            ``clvs[k, :, i]`` is the i-th CLV (column vector). Columns are normalized to unit
            Euclidean norm.
        traj : ndarray, shape (N + 1, system_dimension + 1)
            The corresponding sampled trajectory (after the transient and warm-up), with time in
            the first column and state variables in the remaining columns.

        Raises
        ------
        ValueError
            If ``total_time`` is negative.
            If ``transient_time`` is negative or not compatible with ``total_time``.
            If ``warmup_time`` is negative.
            If ``tail_time`` is negative.
            If ``num_clvs`` is not in ``[1, system_dimension]``.
        TypeError
            If ``u`` cannot be interpreted as a valid initial condition.
            If ``parameters`` cannot be interpreted as valid system parameters.

        Notes
        -----
        Step-by-step algorithm (Ginelli-style)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        Let ``d = system_dimension`` and ``p = num_clvs``. Let ``t_k`` be the sampling times
        separated by ``qr_time_step``. Over each sampling block, the system integrates both the
        trajectory and a ``d × p`` tangent basis, re-orthonormalizing the basis with a QR factorization.

        1. Transient (optional)
        Integrate from the initial condition for ``transient_time`` and discard the result.

        2. Forward QR warm-up (optional)
        Initialize an orthonormal tangent basis ``Q`` (random, orthonormalized). Then for a duration
        ``warmup_time``, repeatedly integrate the trajectory + tangent basis forward, and at each
        sampling time apply a QR factorization to ``Q``. This drives ``Q`` toward the BLVs.

        3. Forward storage window
        Over the main window of duration ``total_time``, integrate forward and at each sampling time
        compute a QR factorization of the evolved tangent basis. Store the orthonormal factors ``Q_k``
        and upper-triangular factors ``R_k`` that encode the local tangent dynamics.

        4. Tail / backward initialization
        Perform additional forward sampling blocks over duration ``tail_time`` (beyond the storage window).
        Use the resulting triangular factors to evolve an arbitrary upper-triangular matrix ``A`` toward
        its asymptotic limit for the backward recursion.

        5. Backward recursion and CLV reconstruction
        Move backward through the stored ``R_k`` factors, updating the coefficient matrix via triangular
        solves and reconstructing covariant vectors at each stored time as ``V_k = Q_k @ A_k``. Columns are
        normalized after reconstruction.

        Choosing ``warmup_time`` and ``tail_time``
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        - ``warmup_time`` controls convergence of the forward orthonormal frame to the BLVs.
        - ``tail_time`` controls convergence of the backward coefficient recursion. If too small,
        CLVs near the end of the window may retain dependence on the random initialization.

        In strongly chaotic regimes, modest values can be sufficient; in weakly chaotic or nearly
        degenerate regimes, larger values may be necessary. A practical check is to repeat the
        computation with larger ``warmup_time``/``tail_time`` and verify stability of CLV-based
        diagnostics.

        References
        ----------
        F. Ginelli et al., Characterizing dynamics with covariant Lyapunov vectors.
        Phys. Rev. Lett. 99, 130601 (2007).
        """

        u = validate_initial_conditions(
            u, self.__system_dimension, allow_ensemble=False
        )
        u = u.copy()

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        else:
            parameters = validate_parameters(parameters, self.__number_of_parameters)

        transient_time, total_time = validate_times(transient_time, total_time, Real)
        time_step = self.__get_initial_time_step(u, parameters)

        if num_clvs is None:
            num_clvs = self.__system_dimension

        if qr_time_step is None:
            qr_time_step = time_step

        return compute_clvs(
            u,
            parameters,
            total_time,
            self.__equations_of_motion,
            self.__jacobian,
            num_clvs,
            transient_time,
            warmup_time,
            tail_time,
            time_step,
            qr_time_step,
            self.__atol,
            self.__rtol,
            self.__integrator_func,
            seed,
        )

    def CLV_angles(
        self,
        u: Union[NDArray[np.float64], Sequence[float]],
        total_time: float,
        parameters: Union[None, NDArray[np.float64], Sequence[float], float] = None,
        subspaces: Optional[Sequence[Tuple[Sequence[int], Sequence[int]]]] = None,
        pairs: Optional[Sequence[Tuple[int, int]]] = None,
        transient_time: float = 0,
        warmup_time: float = 0,
        tail_time: float = 0,
        qr_time_step: int = None,
        seed: int = 13,
    ):
        """
        Compute CLV-based angle diagnostics for this continuous-time dynamical system.

        This method computes CLVs along a trajectory (see :meth:`CLV`) and returns time series of angles between
        user-specified CLV subspaces and/or CLV direction pairs. For details on how CLVs are
        computed (forward QR warm-up, storage, backward recursion, and the roles of
        ``warmup_time`` and ``tail_time``), see :meth:`covariant_lyapunov_vectors`.

        At least one of ``subspaces`` or ``pairs`` must be provided.

        Parameters
        ----------
        u : array_like, shape (system_dimension,)
            Initial condition.
        total_time : float
            Total integration time used for the angle diagnostics. Angles are evaluated at the
            same sampled times used by the internal CLV computation (spaced by ``qr_time_step``).
        parameters : None or array_like, optional
            Model parameters. If ``None``, stored system parameters are used.
        subspaces : sequence of (A, B) or None, optional
            Requested subspace-angle diagnostics. Each entry is a pair ``(A, B)``, where
            ``A`` and ``B`` are sequences of CLV indices defining two subspaces:
            ``E_A(t) = span{ v_i(t) : i in A }`` and ``E_B(t) = span{ v_j(t) : j in B }``.
            For each ``(A, B)``, the returned value is the **minimum principal angle**
            between ``E_A(t)`` and ``E_B(t)`` at each sampled time.
        pairs : sequence of (int, int) or None, optional
            Requested pairwise CLV angles. Each entry ``(i, j)`` returns the angle
            ``angle(v_i(t), v_j(t))`` at each sampled time. Indices refer to CLV ordering
            (0 = most expanding, last = most contracting).
        transient_time : float, default=0
            Initial integration time discarded before starting the diagnostic.
        warmup_time : float, default=0
            Forward QR warm-up duration passed to the CLV computation. See
            :meth:`CLV`.
        tail_time : float, default=0
            Backward-recursion convergence duration passed to the CLV computation. See
            :meth:`CLV`.
        qr_time_step : float, optional
            Time spacing between successive QR re-orthonormalizations / CLV samples. If ``None``,
            defaults to the integrator's initial step size.
        seed : int, default=13
            Seed forwarded to the CLV computation.

        Returns
        -------
        angles : ndarray, shape (N + 1, M)
            Angle time series in radians, evaluated at the ``N + 1`` sampled times used by the CLV
            computation (roughly ``N ≈ total_time / qr_time_step``).

            The columns are ordered as:
            1) angles for all requested CLV pairs in ``pairs`` (in the given order).
            2) followed by angles for all requested subspace diagnostics in ``subspaces`` (in the given order),

        traj : ndarray, shape (N + 1, system_dimension + 1)
            The sampled trajectory used for the angle computation, with time in the first column.

        Raises
        ------
        ValueError
            If ``total_time`` is negative.\\
            If ``transient_time`` is negative or not compatible with ``total_time``.\\
            If ``warmup_time`` is negative.\\
            If ``tail_time`` is negative.\\
            If both ``subspaces`` and ``pairs`` are ``None`` (or empty).\\
            If any subspace specification is invalid (empty sets, overlaps, or out-of-range indices).\\
            If any pair ``(i, j)`` is invalid (identical indices or out-of-range indices).\\
        TypeError
            If ``u`` cannot be interpreted as a valid initial condition.\\
            If ``parameters`` cannot be interpreted as valid system parameters.\\
            If ``subspaces`` or ``pairs`` are not in an acceptable format.

        Notes
        -----
        Interpretation of the angle diagnostics
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        **Subspace angles (minimum principal angle)**
            For each requested ``(A, B)``, the diagnostic computes the smallest principal angle
            between the two CLV subspaces. Near-zero values indicate near-tangencies between the
            subspaces (weak separation / weak hyperbolicity).

        **Pairwise CLV angles**
            Pairwise angles provide a direct measure of alignment between two specific covariant
            directions. They are useful for targeted checks (e.g., alignment of a neutral direction
            with an unstable one), but they do not capture the closest approach between higher-dimensional
            subspaces unless the correct pair is chosen.

        What to look for in practice
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        - **Near-zero events** in subspace angles are typically the most informative: they indicate
        near-tangencies between the chosen subspaces.
        - In systems with near-degenerate Lyapunov exponents, angles may fluctuate more because the
        associated directions/subspaces can be ill-conditioned at finite time.
        - The sampling cadence ``qr_time_step`` controls time resolution of the diagnostics. Smaller
        values resolve faster geometric changes but increase cost.

        See Also
        --------
        covariant_lyapunov_vectors : Computes CLVs and documents the Ginelli-style algorithm, including
            the roles of ``warmup_time`` and ``tail_time``.
        """

        u = validate_initial_conditions(
            u, self.__system_dimension, allow_ensemble=False
        )
        u = u.copy()

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        else:
            parameters = validate_parameters(parameters, self.__number_of_parameters)

        transient_time, total_time = validate_times(transient_time, total_time, Real)

        time_step = self.__get_initial_time_step(u, parameters)
        if qr_time_step is None:
            qr_time_step = time_step

        pairs = validate_clv_pairs(pairs, self.__system_dimension)

        subspaces = validate_clv_subspaces(subspaces, self.__system_dimension)

        return clv_angles(
            u,
            parameters,
            total_time,
            self.__equations_of_motion,
            self.__jacobian,
            transient_time,
            warmup_time,
            tail_time,
            time_step,
            qr_time_step,
            self.__atol,
            self.__rtol,
            self.__integrator_func,
            seed,
            subspaces,
            pairs,
        )

    def SALI(
        self,
        u: Union[NDArray[np.float64], Sequence[float]],
        total_time: float,
        parameters: Union[None, Sequence[float], NDArray[np.float64]] = None,
        transient_time: Optional[float] = None,
        return_history: bool = False,
        seed: int = 13,
        threshold: float = 1e-16,
        endpoint: bool = True,
    ) -> NDArray[np.float64]:
        """Calculate the smallest aligment index (SALI) for a given dynamical system.

        Parameters
        ----------
        u : Union[NDArray[np.float64], Sequence[float]]
            Initial conditions of the system. Must match the system's dimension.
        total_time : float
            Total time over which to evolve the system (including transient).
        parameters : Union[None, Sequence[float], NDArray[np.float64]], optional
            Parameters of the system, by default None. Can be a scalar, a sequence of floats or a numpy array.
        transient_time : Optional[float], optional
            Transient time, i.e., the time to discard before calculating the Lyapunov exponents, by default None.
        return_history : bool, optional
            Whether to return or not the Lyapunov exponents history in time, by default False.
        seed : int, optional
            The seed to randomly generate the deviation vectors, by default 13.
        threshold : float, optional
            The threhshold for early termination, by default 1e-16. When SALI becomes less than `threshold`, stops the execution.
        endpoint : bool, optional
            Whether to include the endpoint time = total_time in the calculation, by default True.

        Returns
        -------
        NDArray[np.float64]
            The SALI value

            - If `return_history = False`, return time and SALI, where time is the time at the end of the execution. time < total_time if SALI becomes less than `threshold` before `total_time`.
            - If `return_history = True`, return the sampled times and the SALI values.
            - If `sample_times` is provided, return the SALI at the specified times.

        Raises
        ------
        ValueError
            - If the Jacobian function is not provided.
            - If the initial condition is not valid, i.e., if the dimensions do not match.
            - If the number of parameters does not match.
            - If `parameters` is not a scalar, 1D list, or 1D array.
            - If `total_time`, `transient_time`, or `threshold` are negative.
        TypeError
            - If `total_time`, `transient_time`, or `threshold` are not valid numbers.
            - If `seed` is not an integer.

        Examples
        --------
        >>> from pynamicalsys import ContinuousDynamicalSystem as cds
        >>> ds = cds(model="lorenz system")
        >>> u = [0.1, 0.1, 0.1]
        >>> total_time = 1000
        >>> transient_time = 500
        >>> parameters = [16.0, 45.92, 4.0]
        >>> ds.SALI(u, total_time, parameters=parameters, transient_time=transient_time)
        (521.8899999999801, 7.850462293418876e-17)
        >>> # Returning the history
        >>> sali = ds.SALI(u, total_time, parameters=parameters, transient_time=transient_time, return_history=True)
        >>> sali.shape
        (2189, 2)
        """

        if self.__jacobian is None:
            raise ValueError(
                "Jacobian function is required to compute Lyapunov exponents"
            )

        u = validate_initial_conditions(
            u, self.__system_dimension, allow_ensemble=False
        )
        u = u.copy()

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        else:
            parameters = validate_parameters(parameters, self.__number_of_parameters)

        transient_time, total_time = validate_times(transient_time, total_time)

        time_step = self.__get_initial_time_step(u, parameters)

        validate_non_negative(threshold, "threshold", type_=Real)

        if endpoint:
            total_time += time_step

        result = SALI(
            u,
            parameters,
            total_time,
            self.__equations_of_motion,
            self.__jacobian,
            transient_time=transient_time,
            time_step=time_step,
            atol=self.__atol,
            rtol=self.__rtol,
            integrator=self.__integrator_func,
            return_history=return_history,
            seed=seed,
            threshold=threshold,
        )

        if return_history:
            return np.array(result)
        else:
            return np.array(result[0])

    def LDI(
        self,
        u: Union[NDArray[np.float64], Sequence[float]],
        total_time: float,
        k: int,
        parameters: Union[None, Sequence[float], NDArray[np.float64]] = None,
        transient_time: Optional[float] = None,
        return_history: bool = False,
        seed: int = 13,
        threshold: float = 1e-16,
        endpoint: bool = True,
    ) -> NDArray[np.float64]:
        """Calculate the linear dependence index (LDI) for a given dynamical system.

        Parameters
        ----------
        u : Union[NDArray[np.float64], Sequence[float]]
            Initial conditions of the system. Must match the system's dimension.
        total_time : float
            Total time over which to evolve the system (including transient).
        parameters : Union[None, Sequence[float], NDArray[np.float64]], optional
            Parameters of the system, by default None. Can be a scalar, a sequence of floats or a numpy array.
        transient_time : Optional[float], optional
            Transient time, i.e., the time to discard before calculating the Lyapunov exponents, by default None.
        return_history : bool, optional
            Whether to return or not the Lyapunov exponents history in time, by default False.
        seed : int, optional
            The seed to randomly generate the deviation vectors, by default 13.
        threshold : float, optional
            The threhshold for early termination, by default 1e-16. When SALI becomes less than `threshold`, stops the execution.
        endpoint : bool, optional
            Whether to include the endpoint time = total_time in the calculation, by default True.

        Returns
        -------
        NDArray[np.float64]
            The LDI value

            - If `return_history = False`, return time and LDI, where time is the time at the end of the execution. time < total_time if LDI becomes less than `threshold` before `total_time`.
            - If `return_history = True`, return the sampled times and the LDI values.

        Raises
        ------
        ValueError
            - If the Jacobian function is not provided.
            - If the initial condition is not valid, i.e., if the dimensions do not match.
            - If the number of parameters does not match.
            - If `parameters` is not a scalar, 1D list, or 1D array.
            - If `total_time`, `transient_time`, or `threshold` are negative.
            - If `k` < 2.
        TypeError
            - If `total_time`, `transient_time`, or `threshold` are not valid numbers.
            - If `seed` is not an integer.

        Examples
        --------
        >>> from pynamicalsys import ContinuousDynamicalSystem as cds
        >>> ds = cds(model="lorenz system")
        >>> u = [0.1, 0.1, 0.1]
        >>> total_time = 1000
        >>> transient_time = 500
        >>> parameters = [16.0, 45.92, 4.0]
        >>> ds.LDI(u, total_time, 2, parameters=parameters, transient_time=transient_time)
        array([5.23170000e+02, 6.93495605e-17])
        >>> ds.LDI(u, total_time, 3, parameters=parameters, transient_time=transient_time)
        (501.26999999999884, 9.984145370766051e-17)
        >>> # Returning the history
        >>> ldi = ds.LDI(u, total_time, 2, parameters=parameters, transient_time=transient_time)
        >>> ldi.shape
        (2181, 2)
        """

        if self.__jacobian is None:
            raise ValueError(
                "Jacobian function is required to compute Lyapunov exponents"
            )

        u = validate_initial_conditions(
            u, self.__system_dimension, allow_ensemble=False
        )
        u = u.copy()

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        else:
            parameters = validate_parameters(parameters, self.__number_of_parameters)

        transient_time, total_time = validate_times(transient_time, total_time)

        time_step = self.__get_initial_time_step(u, parameters)

        validate_non_negative(threshold, "threshold", type_=Real)

        if endpoint:
            total_time += time_step

        result = LDI(
            u,
            parameters,
            total_time,
            self.__equations_of_motion,
            self.__jacobian,
            k,
            transient_time=transient_time,
            time_step=time_step,
            atol=self.__atol,
            rtol=self.__rtol,
            integrator=self.__integrator_func,
            return_history=return_history,
            seed=seed,
            threshold=threshold,
        )

        if return_history:
            return np.array(result)
        else:
            return np.array(result[0])

    def GALI(
        self,
        u: Union[NDArray[np.float64], Sequence[float]],
        total_time: float,
        k: int,
        parameters: Union[None, Sequence[float], NDArray[np.float64]] = None,
        transient_time: Optional[float] = None,
        return_history: bool = False,
        seed: int = 13,
        threshold: float = 1e-16,
        endpoint: bool = True,
    ) -> NDArray[np.float64]:
        """Calculate the Generalized Aligment Index (GALI) for a given dynamical system.

        Parameters
        ----------
        u : Union[NDArray[np.float64], Sequence[float]]
            Initial conditions of the system. Must match the system's dimension.
        total_time : float
            Total time over which to evolve the system (including transient).
        parameters : Union[None, Sequence[float], NDArray[np.float64]], optional
            Parameters of the system, by default None. Can be a scalar, a sequence of floats or a numpy array.
        transient_time : Optional[float], optional
            Transient time, i.e., the time to discard before calculating the Lyapunov exponents, by default None.
        return_history : bool, optional
            Whether to return or not the Lyapunov exponents history in time, by default False.
        seed : int, optional
            The seed to randomly generate the deviation vectors, by default 13.
        threshold : float, optional
            The threhshold for early termination, by default 1e-16. When SALI becomes less than `threshold`, stops the execution.
        endpoint : bool, optional
            Whether to include the endpoint time = total_time in the calculation, by default True.

        Returns
        -------
        NDArray[np.float64]
            The GALI value

            - If `return_history = False`, return time and GALI, where time is the time at the end of the execution. time < total_time if GALI becomes less than `threshold` before `total_time`.
            - If `return_history = True`, return the sampled times and the GALI values.

        Raises
        ------
        ValueError
            - If the Jacobian function is not provided.
            - If the initial condition is not valid, i.e., if the dimensions do not match.
            - If the number of parameters does not match.
            - If `parameters` is not a scalar, 1D list, or 1D array.
            - If `total_time`, `transient_time`, or `threshold` are negative.
            - If `k` < 2.
        TypeError
            - If `total_time`, `transient_time`, or `threshold` are not valid numbers.
            - If `seed` is not an integer.

        Examples
        --------
        >>> from pynamicalsys import ContinuousDynamicalSystem as cds
        >>> ds = cds(model="lorenz system")
        >>> u = [0.1, 0.1, 0.1]
        >>> total_time = 1000
        >>> transient_time = 500
        >>> parameters = [16.0, 45.92, 4.0]
        >>> ds.GALI(u, total_time, 2, parameters=parameters, transient_time=transient_time)
        (521.8099999999802, 7.328757804386809e-17)
        >>> ds.GALI(u, total_time, 3, parameters=parameters, transient_time=transient_time)
        (501.26999999999884, 9.984145370766051e-17)
        >>> # Returning the history
        >>> gali = ds.GALI(u, total_time, 2, parameters=parameters, transient_time=transient_time)
        >>> gali.shape
        (2181, 2)
        """

        if self.__jacobian is None:
            raise ValueError(
                "Jacobian function is required to compute Lyapunov exponents"
            )

        u = validate_initial_conditions(
            u, self.__system_dimension, allow_ensemble=False
        )
        u = u.copy()

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        else:
            parameters = validate_parameters(parameters, self.__number_of_parameters)

        transient_time, total_time = validate_times(transient_time, total_time)

        time_step = self.__get_initial_time_step(u, parameters)

        validate_non_negative(threshold, "threshold", type_=Real)

        if endpoint:
            total_time += time_step

        result = GALI(
            u,
            parameters,
            total_time,
            self.__equations_of_motion,
            self.__jacobian,
            k,
            transient_time=transient_time,
            time_step=time_step,
            atol=self.__atol,
            rtol=self.__rtol,
            integrator=self.__integrator_func,
            return_history=return_history,
            seed=seed,
            threshold=threshold,
        )

        if return_history:
            return np.array(result)
        else:
            return np.array(result[0])

    def recurrence_time_entropy(
        self,
        u: Union[NDArray[np.float64], Sequence[float]],
        num_intersections: int,
        parameters: Union[None, Sequence[float], NDArray[np.float64]] = None,
        transient_time: Optional[float] = None,
        map_type: str = "SM",
        section_index: Optional[int] = None,
        section_value: Optional[float] = None,
        crossing: Optional[int] = None,
        sampling_time: Optional[float] = None,
        maxima_index: Optional[float] = None,
        **kwargs,
    ) -> Union[float, Tuple[float, NDArray[np.float64]]]:
        """Compute the Recurrence Time Entropy (RTE) for a dynamical system.

        Parameters
        ----------
        u: Union[NDArray[np.float64], Sequence[float]]
            Initial condition of shape(d,) where d is system dimension
        num_intersections: int
            Number of intersections to record in the Poincaré section or stroboscopic map.
        parameters: Union[None, float, Sequence[np.float64], NDArray[np.float64]], optional
            System parameters of shape(p,) passed to mapping function
        transient_time : float, optional
            Initial time to discard before recording the section.
        map_type : str
            Which map to use: stroboscopic map or Poincaré section, by default "SM"
        section_index : Optional[int]
            Index of the coordinate to define the Poincaré section (0-based). Only used when map_type="PS".
        section_value : Optional[float]
            Value of the coordinate at which the section is defined. Only used when map_type="PS".
        crossing : Optional[int]
            Specifies the type of crossing to consider:
            - 1 : positive crossing (from below to above section_value)
            - -1 : negative crossing (from above to below section_value)
            - 0 : all crossings

            Only used when map_type="PS".
        sampling_time : float
            Time interval between consecutive samples in the stroboscopic map. Only used when map_type="SM".
        maxima_index : Optional[int]
            Index of the coordinate whose maxima will be recorded. Only used when map_type="MM".
        metric: {"supremum", "euclidean", "manhattan"}, default = "supremum"
            Distance metric used for phase space reconstruction.
        std_metric: {"supremum", "euclidean", "manhattan"}, default = "supremum"
            Distance metric used for standard deviation calculation.
        lmin: int, default = 1
            Minimum line length to consider in recurrence quantification.
        threshold: float, default = 0.1
            Recurrence threshold(relative to data range).
        threshold_std: bool, default = True
            Whether to scale threshold by data standard deviation.
        return_final_state: bool, default = False
            Whether to return the final system state in results.
        return_recmat: bool, default = False
            Whether to return the recurrence matrix.
        return_p: bool, default = False
            Whether to return white vertical line length distribution.

        Returns
        -------
        Union[float, Tuple[float, NDArray[np.float64]]]
            - float: RTE value(base case)
            - Tuple: (RTE, white_line_distribution) if return_distribution = True

        Raises
        ------
        ValueError
            - If `u` is not a 1D array matching the system dimension.
            - If `parameters` is not `None` and does not match the expected number of parameters.
            - If `parameters` is `None` but the system expects parameters.
            - If `parameters` is a scalar or array-like but not 1D.
            - If `transient_time` is negative.
            - If `map_type` is not one of {"SM", "PS"}.
            - If `map_type="PS"` but any of `section_index`, `section_value`, or `crossing` is `None`.
            - If `section_index` is negative or ≥ system dimension.
            - If `crossing` is not one of {-1, 0, 1}.
            - If `map_type="SM"` but `sampling_time` is `None` or negative.
        TypeError
            - If `u` is not a scalar or array-like type.
            - If `parameters` is not a scalar or array-like type.
            - If `map_type` is not a string.
            - If `section_value` is not a real number when `map_type="PS"`.
            - If `crossing` is not an integer when `map_type="PS"`.
            - If `sampling_time` is not a real number when `map_type="SM"`.

        Notes
        -----
        - Higher RTE indicates more complex dynamics
        - Set min_recurrence_time = 2 to ignore single-point recurrences
        - Implementation follows [1]

        References
        ----------
        [1] Sales et al., Chaos 33, 033140 (2023)

        Examples
        --------
        >>>  # Basic usage
        >>> rte = system.recurrence_time_entropy(u0, params, 5000)

        >>>  # With distribution output
        >>> rte, dist = system.recurrence_time_entropy(
        ...     u0, params, 5000,
        ...     return_distribution=True,
        ...     recurrence_threshold=0.1
        ...)
        """
        u = validate_initial_conditions(
            u, self.__system_dimension, allow_ensemble=False
        )
        u = u.copy()

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        else:
            parameters = validate_parameters(parameters, self.__number_of_parameters)

        validate_non_negative(transient_time, "transient_time", Real)

        if not isinstance(map_type, str):
            raise TypeError("map_type must be a string")

        if map_type == "PS":
            if section_index is None or section_value is None or crossing is None:
                raise ValueError(
                    'When using map_type="PS", you must inform section_index, section_value, and crossing'
                )

            validate_non_negative(section_index, "section_index", Integral)
            if section_index >= self.__system_dimension:
                raise ValueError("section_index must be in [0, system_dimension)")

            if not isinstance(section_value, Real):
                raise TypeError("section_value must be a valid real number")

            if not isinstance(crossing, Integral):
                raise TypeError("crossing must be a valid integer number")
            elif crossing not in [-1, 0, 1]:
                raise ValueError("crossing must be -1, 0, or 1")

        elif map_type == "SM":

            if sampling_time is not None:
                validate_non_negative(sampling_time, "sampling_time", Real)
            else:
                raise ValueError(
                    'When using map_type="SM" you must inform sampling_time'
                )
        elif map_type == "MM":
            if maxima_index is not None:
                validate_non_negative(maxima_index, "maxima_index", Integral)
            else:
                raise ValueError(
                    'When using map_type="MM" you must inform maxima_index'
                )
        else:
            raise ValueError(
                "map_type must be SM (stroboscopic map), PS (Poincaré section), or MM (Maxima map)"
            )

        time_step = self.__get_initial_time_step(u, parameters)

        return recurrence_time_entropy(
            u,
            parameters,
            num_intersections,
            transient_time,
            self.__equations_of_motion,
            time_step,
            self.__atol,
            self.__rtol,
            self.__integrator_func,
            map_type,
            section_index,
            section_value,
            crossing,
            sampling_time,
            maxima_index,
            **kwargs,
        )

    def hurst_exponent(
        self,
        u: Union[NDArray[np.float64], Sequence[float]],
        num_intersections: int,
        parameters: Union[None, Sequence[float], NDArray[np.float64]] = None,
        transient_time: Optional[float] = None,
        wmin: int = 2,
        map_type: str = "SM",
        section_index: Optional[int] = None,
        section_value: Optional[float] = None,
        crossing: Optional[int] = None,
        sampling_time: Optional[float] = None,
        maxima_index: Optional[float] = None,
    ) -> Union[float, Tuple[float, NDArray[np.float64]]]:
        """
        Estimate the Hurst exponent for a system trajectory using the rescaled range (R/S) method.

        Parameters
        ----------
        u : NDArray[np.float64]
            Initial condition vector of shape (n,).
        parameters : Union[None, float, Sequence[np.float64], NDArray[np.float64]], optional
            Parameters passed to the mapping function.
        total_time : int
            Total number of iterations used to generate the trajectory.
        transient_time : Optional[int], optional
            Number of initial iterations to discard as transient. If `None`, no transient is removed. Default is `None`.
        wmin : int, optional
            Minimum window size for the rescaled range calculation. Default is 2.
        map_type : str
            Which map to use: stroboscopic map or Poincaré section, by default "SM"
        section_index : Optional[int]
            Index of the coordinate to define the Poincaré section (0-based). Only used when map_type="PS".
        section_value : Optional[float]
            Value of the coordinate at which the section is defined. Only used when map_type="PS".
        crossing : Optional[int]
            Specifies the type of crossing to consider:
            - 1 : positive crossing (from below to above section_value)
            - -1 : negative crossing (from above to below section_value)
            - 0 : all crossings

            Only used when map_type="PS".
        sampling_time : float
            Time interval between consecutive samples in the stroboscopic map. Only used when map_type="SM".
        maxima_index : Optional[int]
            Index of the coordinate whose maxima will be recorded. Only used when map_type="MM".

        Returns
        -------
        NDArray[np.float64]
            Estimated Hurst exponents for each dimension of the input vector `u`, of shape (n,).

        Raises
        ------
        TypeError
            - If `map_type` is not a string.
            - If `section_value` is not a real number when `map_type="PS"`.
            - If `crossing` is not an integer when `map_type="PS"`.
            - If `sampling_time` is not a real number when `map_type="SM"`.
            - If `maxima_index` is not an integer when `map_type="MM"`.
        ValueError
            - If `map_type` is not one of {"SM", "PS", "MM"}.
            - If `map_type="PS"` and any of `section_index`, `section_value`, or `crossing` is `None`.
            - If `section_index` is negative or ≥ system dimension when `map_type="PS"`.
            - If `crossing` is not in {-1, 0, 1} when `map_type="PS"`.
            - If `map_type="SM"` and `sampling_time` is `None` or negative.
            - If `map_type="MM"` and `maxima_index` is `None` or negative.
            - If `transient_time` is negative.
            - If `wmin` is less than 2 or greater than or equal to `num_intersections // 2`.

        Notes
        -----
        The Hurst exponent is a measure of the long-term memory of a time series:

        - H = 0.5 indicates a random walk (no memory).
        - H > 0.5 indicates persistent behavior (positive autocorrelation).
        - H < 0.5 indicates anti-persistent behavior (negative autocorrelation).

        This implementation computes the rescaled range (R/S) for various window sizes and
        performs a linear regression in log-log space to estimate the exponent.

        The function supports multivariate time series, estimating one Hurst exponent per dimension.
        """
        u = validate_initial_conditions(
            u, self.__system_dimension, allow_ensemble=False
        )
        u = u.copy()

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        else:
            parameters = validate_parameters(parameters, self.__number_of_parameters)

        validate_non_negative(transient_time, "transient_time", Real)

        if not isinstance(map_type, str):
            raise TypeError("map_type must be a string")

        if map_type == "PS":
            if section_index is None or section_value is None or crossing is None:
                raise ValueError(
                    'When using map_type="PS", you must inform section_index, section_value, and crossing'
                )

            validate_non_negative(section_index, "section_index", Integral)
            if section_index >= self.__system_dimension:
                raise ValueError("section_index must be in [0, system_dimension)")

            if not isinstance(section_value, Real):
                raise TypeError("section_value must be a valid real number")

            if not isinstance(crossing, Integral):
                raise TypeError("crossing must be a valid integer number")
            elif crossing not in [-1, 0, 1]:
                raise ValueError("crossing must be -1, 0, or 1")

        elif map_type == "SM":

            if sampling_time is not None:
                validate_non_negative(sampling_time, "sampling_time", Real)
            else:
                raise ValueError(
                    'When using map_type="SM" you must inform sampling_time'
                )
        elif map_type == "MM":
            if maxima_index is not None:
                validate_non_negative(maxima_index, "maxima_index", Integral)
            else:
                raise ValueError(
                    'When using map_type="MM" you must inform maxima_index'
                )
        else:
            raise ValueError(
                "map_type must be SM (stroboscopic map), PS (Poincaré section), or MM (Maxima map)"
            )

        if wmin < 2 or wmin >= num_intersections // 2:
            raise ValueError(
                f"`wmin` must be an integer >= 2 and <= total_time / 2. Got {wmin}."
            )

        time_step = self.__get_initial_time_step(u, parameters)

        return hurst_exponent_wrapped(
            u,
            parameters,
            num_intersections,
            self.__equations_of_motion,
            time_step,
            self.__atol,
            self.__rtol,
            self.__integrator_func,
            map_type,
            section_index,
            section_value,
            crossing,
            sampling_time,
            maxima_index,
            wmin,
            transient_time,
        )
