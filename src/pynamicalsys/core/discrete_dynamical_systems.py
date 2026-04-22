# discrete_dynamical_systems.py

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
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, Literal

import numpy as np
from numba.core.errors import NumbaExperimentalFeatureWarning
from numpy.typing import NDArray

from pynamicalsys.common.recurrence_quantification_analysis import (
    RTEConfig,
    build_recurrence_matrix,
    calculate_threshold,
)

from pynamicalsys.common.utils import finite_difference_jacobian, qr, householder_qr

from pynamicalsys.common.types import int_t, numeric_like_t, numeric_t, observable_t


from pynamicalsys.discrete_time.trajectory import (
    iterate_mapping,
    generate_trajectory,
    ensemble_trajectories,
)
from pynamicalsys.discrete_time.bifurcation import bifurcation_diagram
from pynamicalsys.discrete_time.periodic_orbits import (
    period_counter,
    is_periodic,
    find_periodic_orbit,
    find_periodic_orbit_symmetry_line,
)
from pynamicalsys.discrete_time.stability import (
    eigenvalues_and_eigenvectors,
    classify_stability,
)
from pynamicalsys.discrete_time.manifolds import calculate_manifolds
from pynamicalsys.discrete_time.rotation import rotation_number
from pynamicalsys.discrete_time.escape import (
    escape_basin_and_time_entering,
    escape_time_exiting,
    survival_probability_core,
)
from pynamicalsys.discrete_time.averages import ensemble_time_average
from pynamicalsys.discrete_time.lyapunov import (
    lyapunov_1D,
    lyapunov_er,
    maximum_lyapunov_er,
    lyapunov_qr,
    finite_time_lyapunov,
)
from pynamicalsys.discrete_time.sali import sali
from pynamicalsys.discrete_time.ldi import ldi_k
from pynamicalsys.discrete_time.gali import gali_k
from pynamicalsys.discrete_time.clv import compute_clvs, clv_angles
from pynamicalsys.discrete_time.birkhoff import dig
from pynamicalsys.discrete_time.hurst import (
    hurst_exponent_wrapped,
    finite_time_hurst_exponent,
)
from pynamicalsys.discrete_time.rte import RTE, rte_return_t, finite_time_RTE


from pynamicalsys.discrete_time.models import (
    extended_standard_nontwist_map,
    extended_standard_nontwist_map_backwards,
    extended_standard_nontwist_map_jacobian,
    henon_map,
    henon_map_jacobian,
    leonel_map,
    leonel_map_backwards,
    leonel_map_jacobian,
    logistic_map,
    logistic_map_jacobian,
    lozi_map,
    lozi_map_jacobian,
    rulkov_map,
    rulkov_map_jacobian,
    standard_map,
    standard_map_backwards,
    standard_map_jacobian,
    standard_nontwist_map,
    standard_nontwist_map_backwards,
    standard_nontwist_map_jacobian,
    symplectic_map_4D,
    symplectic_map_4D_backwards,
    symplectic_map_4D_jacobian,
    unbounded_standard_map,
)

from pynamicalsys.discrete_time.transport import (
    average_vs_time,
    cumulative_average_vs_time,
    diffusion_coefficient,
    mean_squared_displacement,
    recurrence_times,
    root_mean_squared,
)

from pynamicalsys.discrete_time.validators import (
    validate_finite_time,
    validate_and_convert_param_range,
    validate_sample_times,
)

from pynamicalsys.common.validators import (
    validate_initial_conditions,
    validate_parameters,
    validate_non_negative,
    validate_positive,
    validate_transient_time,
    validate_axis,
    validate_clv_pairs,
    validate_clv_subspaces,
)


class DiscreteDynamicalSystem:
    """Class representing a discrete dynamical system with various models and methods for analysis.

    This class allows users to work with predefined dynamical models or custom mappings,
    compute trajectories, bifurcation diagrams, periods, and perform various dynamical analyses.
    It supports both single initial conditions and ensembles of initial conditions, providing
    methods for generating trajectories, computing bifurcation diagrams, and analyzing stability.

    Parameters
    ----------
    model : str, optional
        Name of the predefined model to use (e.g., "henon map"). If provided, overrides custom mappings.
    mapping : callable, optional
        Custom mapping function with signature f(u, parameters) -> array_like.
        If provided, model must be None.
    jacobian : callable, optional
        Custom Jacobian function with signature J(u, parameters, *args) -> array_like.
        If provided, must be compatible with the mapping function.
    backwards_mapping : callable, optional
        Custom inverse mapping function with signature f_inv(u, parameters) -> array_like.
        If provided, must be compatible with the mapping function.
    system_dimension : int, optional
        Dimension of the system (number of variables in the mapping).
        Required if using custom mappings without a predefined model.

    Raises
    ------
    ValueError
        - If neither model nor mapping is provided, or if provided model name is not implemented.
        - If mapping is provided without jacobian for models requiring it.

    TypeError
        - If provided mapping or jacobian is not callable.

    Notes
    -----
    - When providing custom functions, either provide both mapping and jacobian,
        or just mapping (in which case finite differences will be used for Jacobian)
    - When providing custom functions, the mapping function signature should be f(u, parameters) -> NDArray[np.float64]
    - The class supports various predefined models such as the standard map, Hénon map, logistic map, and others.
    - The available models can be queried using the `available_models` class method.

    Examples
    --------
    >>> # Using predefined model
    >>> system = DiscreteDynamicalSystem(model="henon map")
    >>> # Using custom mappings
    >>> def my_map(u, parameters):
    ...     return np.array([u[0] + parameters[0] * u[1], u[1] - parameters[1] * u[0]])
    >>> def my_jacobian(u, parameters):
    ...     return np.array([[1, parameters[0]], [-parameters[1], 1]])
    >>> system = DiscreteDynamicalSystem(
        mapping=my_map,
        jacobian=my_jacobian,
        system_dimension=2,
        number_of_parameters=2
    )
    """

    # Class-level constant defining all available models
    __AVAILABLE_MODELS: Dict[str, Dict[str, Any]] = {
        "standard map": {
            "description": "Standard Chirikov-Taylor map (area-preserving 2D)",
            "has_jacobian": True,
            "has_backwards_map": True,
            "mapping": standard_map,
            "jacobian": standard_map_jacobian,
            "backwards_mapping": standard_map_backwards,
            "dimension": 2,
            "number_of_parameters": 1,
            "parameters": ["k"],
        },
        "unbounded standard map": {
            "description": "Standard Chirikov-Taylor map withou boundaries on the y varibles. Useful to study diffusion",
            "has_jacobian": False,
            "has_backwards_map": False,
            "mapping": unbounded_standard_map,
            "jacobian": None,
            "backwards_mapping": None,
            "dimension": 2,
            "number_of_parameters": 1,
            "parameters": ["k"],
        },
        "henon map": {
            "description": "Hénon quadratic map",
            "has_jacobian": True,
            "has_backwards_map": False,
            "mapping": henon_map,
            "jacobian": henon_map_jacobian,
            "backwards_mapping": None,
            "dimension": 2,
            "number_of_parameters": 2,
            "parameters": ["a", "b"],
        },
        "lozi map": {
            "description": "Lozi map",
            "has_jacobian": True,
            "has_backwards_map": False,
            "mapping": lozi_map,
            "jacobian": lozi_map_jacobian,
            "backwards_mapping": None,
            "dimension": 2,
            "number_of_parameters": 2,
            "parameters": ["a", "b"],
        },
        "rulkov map": {
            "description": "Rulkov map",
            "has_jacobian": True,
            "has_backwards_map": False,
            "mapping": rulkov_map,
            "jacobian": rulkov_map_jacobian,
            "backwards_mapping": None,
            "dimension": 2,
            "number_of_parameters": 3,
            "parameters": ["alpha", "sigma", "mu"],
        },
        "logistic map": {
            "description": "Logistic map (1D nonlinear system)",
            "has_jacobian": True,
            "has_backwards_map": False,
            "mapping": logistic_map,
            "jacobian": logistic_map_jacobian,
            "backwards_mapping": None,
            "dimension": 1,
            "number_of_parameters": 1,
            "parameters": ["r"],
        },
        "standard nontwist map": {
            "description": "Standard nontwist map (area-preserving but violates twist condition)",
            "has_jacobian": True,
            "has_backwards_map": True,
            "mapping": standard_nontwist_map,
            "jacobian": standard_nontwist_map_jacobian,
            "backwards_mapping": standard_nontwist_map_backwards,
            "dimension": 2,
            "number_of_parameters": 2,
            "parameters": ["a", "b"],
        },
        "extended standard nontwist map": {
            "description": "Extended version of standard nontwist map",
            "has_jacobian": True,
            "has_backwards_map": True,
            "mapping": extended_standard_nontwist_map,
            "jacobian": extended_standard_nontwist_map_jacobian,
            "backwards_mapping": extended_standard_nontwist_map_backwards,
            "dimension": 2,
            "number_of_parameters": 4,
            "parameters": ["a", "b", "c", "m"],
        },
        "leonel map": {
            "description": "Leonel's map model",
            "has_jacobian": True,
            "has_backwards_map": True,
            "mapping": leonel_map,
            "jacobian": leonel_map_jacobian,
            "backwards_mapping": leonel_map_backwards,
            "dimension": 2,
            "number_of_parameters": 2,
            "parameters": ["eps", "gamma"],
        },
        "4d symplectic map": {
            "description": "4D symplectic map: two coupled standard maps",
            "has_jacobian": True,
            "has_backwards_map": True,
            "mapping": symplectic_map_4D,
            "jacobian": symplectic_map_4D_jacobian,
            "backwards_mapping": symplectic_map_4D_backwards,
            "dimension": 4,
            "number_of_parameters": 3,
            "parameters": ["eps1", "eps2", "xi"],
        },
    }

    def __init__(
        self,
        model: Optional[str] = None,
        mapping: Optional[Callable] = None,
        jacobian: Optional[Callable] = None,
        backwards_mapping: Optional[Callable] = None,
        system_dimension: Optional[int_t] = None,
        parameters: Optional[Sequence] = None,
        number_of_parameters: Optional[int_t] = None,
    ) -> None:
        """Initialize the discrete dynamical system with either a predefined model or custom mappings.

        Parameters
        ----------
        model : str, optional
            Name of the predefined model to use.
        mapping : callable, optional
            Custom mapping function with signature f(u, parameters) -> array_like
        jacobian : callable, optional
            Custom Jacobian function with signature J(u, parameters, *args) -> array_like
        backwards_mapping : callable, optional
            Custom inverse mapping function with signature f_inv(u, parameters) -> array_like
        system_dimension : int, optional
            Dimension of the system (number of variables in the mapping).
        parameters : sequence, optional
            The parameters of the system. If provided, automatically defines the number of parameters.
        number_of_parameters : int, optional
            Number of parameters of the system. Used only when parameters is not provided.

        Raises
        ------
        ValueError
            - If neither model nor mapping is provided.
            - If both model or mapping are provided.
            - If provided model name is not implemented.

        Notes
        -----
        - When providing custom functions, either provide both mapping and jacobian,
        or just mapping (in which case finite differences will be used for Jacobian)
        - When providing custom functions, the mapping function signature should be f(u, parameters) -> NDArray[np.float64]

        Examples
        --------
        >>> # Using predefined model
        >>> system = DynamicalSystem(model="henon_map")
        >>> # Using custom mappings
        >>> system = DynamicalSystem(mapping=my_map, jacobian=my_jacobian, system_dimension=dim)
        """

        self.__mapping: Callable
        self.__jacobian: Callable
        self.__backwards_mapping: Optional[Callable]
        self.__system_dimension: int
        self.__number_of_parameters: int
        self.__parameters: NDArray[np.float64] | None
        self.__model: str | None

        warnings.filterwarnings("ignore", category=NumbaExperimentalFeatureWarning)

        if model is not None and mapping is not None:
            raise ValueError("Cannot specify both model and custom mapping")

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
            self.__mapping = model_info["mapping"]
            self.__jacobian = (
                model_info["jacobian"]
                if model_info["jacobian"] is not None
                else finite_difference_jacobian
            )
            self.__backwards_mapping = model_info["backwards_mapping"]
            self.__system_dimension = model_info["dimension"]
            self.__parameters = None
            self.__number_of_parameters = int(model_info["number_of_parameters"])

            if jacobian is not None:  # Allow override of default Jacobian
                self.__jacobian = jacobian

            if backwards_mapping is not None:  # Allow override of default backwards map
                self.__backwards_mapping = backwards_mapping

        elif (
            mapping is not None
            and system_dimension is not None
            and (parameters is not None or number_of_parameters is not None)
        ):
            self.__mapping = mapping
            self.__jacobian = (
                jacobian if jacobian is not None else finite_difference_jacobian
            )
            self.__backwards_mapping = backwards_mapping

            validate_non_negative(system_dimension, "system_dimension", Integral)
            if number_of_parameters is not None:
                validate_non_negative(
                    number_of_parameters, "number_of_parameters", Integral
                )

            self.__system_dimension = int(system_dimension)
            if parameters is not None:
                self.__number_of_parameters = int(np.atleast_1d(parameters).size)
                self.__parameters = validate_parameters(
                    parameters, self.__number_of_parameters
                )
            elif number_of_parameters is not None:
                validate_non_negative(
                    number_of_parameters, "number_of_parameters", Integral
                )
                self.__parameters = None
                self.__number_of_parameters = int(number_of_parameters)
            else:
                raise ValueError(
                    "Must provide either 'parameters' or 'number_of_parameters'."
                )

            # Validate custom functions
            if not callable(self.__mapping):
                raise TypeError("Custom mapping must be callable")

            if self.__jacobian is not None and not callable(self.__jacobian):
                raise TypeError("Custom Jacobian must be callable or None")

            if self.__backwards_mapping is not None and not callable(
                self.__backwards_mapping
            ):
                raise TypeError("Custom backwards mapping must be callable or None")

        else:
            raise ValueError(
                "Must specify either a model name or custom mapping function with its dimension and parameters or number of paramters."
            )

    @classmethod
    def available_models(cls) -> List[str]:
        """Return a list of available models."""
        return list(cls.__AVAILABLE_MODELS.keys())

    @property
    def info(self) -> Dict[str, Any]:
        """Return a dictionary with information about the current model."""

        if self.__model is None:
            raise ValueError(
                "The 'info' property is only available when a model is provided."
            )

        model = self.__model.lower()

        return self.__AVAILABLE_MODELS[model]

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

    def step(
        self,
        u: Union[NDArray[np.float64], Sequence[float], float],
        parameters: Union[None, float, Sequence[float], NDArray[np.float64]] = None,
    ) -> NDArray[np.float64]:
        """Perform one step in the mapping evolution

        Parameters
        ----------
        u : Union[NDArray[np.float64], Sequence[float], float]
            Initial condition(s):
            - Single IC: 1D array of shape (d,) where d is the system dimension
            - Ensemble: 2D array of shape (n, d) for n initial conditions
            - Also accepts sequence types that will be converted to numpy arrays
            - Scalar
        parameters : Union[NDArray[np.float64], Sequence[float], float], optional
            Parameters of the dynamical system, shape (p,) where p is the number of parameters

        Returns
        -------
        NDArray[np.float64]
            The next step of the given initial condition with the same shape as `u`.

        Raises
        ------
        ValueError
            - If `u` is not a scalar, 1D, or 2D array, or if its shape does not match the expected system dimension.
            - If `u` is a 1D array but its length does not match the system dimension, or if `u` is a 2D array but does not match the expected shape for an ensemble.
            - If `parameters` is not None and does not match the expected number of parameters.
            - If `parameters` is None but the system expects parameters.
            - If `parameters` is a scalar or array-like but not 1D.

        TypeError
            - If `u` is not a scalar or array-like type.
            - If `parameters` is not a scalar or array-like type.

        Examples
        --------
        >>> from pynamicalsys import DiscreteDynamicalSystem as dds
        >>> ds = dds(model="standard map")
        >>> # Single initial condition
        >>> u = [0.2, 0.5]
        >>> ds.step(u, parameters=1.5)
        [[0.92704802 0.72704802]]
        >>> # Multiple initial conditions
        >>> u = np.array([[0.2, 0.5], [0.2, 0.3], [0.2, 0.6]])
        >>> ds.step(u, paramters=1.5)
        array([[0.92704802, 0.72704802],
               [0.72704802, 0.52704802],
               [0.02704802, 0.82704802]])
        """
        u = validate_initial_conditions(u, self.__system_dimension, allow_ensemble=True)

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        else:
            parameters = validate_parameters(parameters, self.__number_of_parameters)

        if u.ndim == 1:
            u_next = self.__mapping(u, parameters)
        else:
            u_next = np.zeros_like(u)
            for i in range(u_next.shape[0]):
                u_next[i] = self.__mapping(u[i], parameters)

        return u_next

    def trajectory(
        self,
        u: numeric_like_t,
        total_time: int_t,
        parameters: numeric_like_t | None = None,
        transient_time: int_t | None = None,
    ) -> NDArray[np.float64]:
        """
        Generate a trajectory from a single initial condition or from an ensemble of
        initial conditions.

        If `u` is one-dimensional, the trajectory of a single initial condition is
        returned. If `u` is two-dimensional, trajectories are computed for all
        initial conditions in the ensemble and returned in concatenated form.

        Parameters
        ----------
        u : numeric_like_t
            Initial condition or ensemble of initial conditions.

            - Single initial condition: shape `(system_dimension,)`
            - Ensemble of initial conditions: shape
              `(num_initial_conditions, system_dimension)`
        total_time : int_t
            Total number of iterations used to generate the trajectory.
        parameters : numeric_like_t | None, optional
            System parameters passed to the mapping function. If `None`, the stored
            system parameters are used.
        transient_time : int_t | None, optional
            Number of initial iterations discarded before storing the trajectory.
            If provided, it must satisfy `0 <= transient_time < total_time`.

        Returns
        -------
        NDArray[np.float64]
            Trajectory array.

            - Single initial condition:
              shape `(sample_size, system_dimension)`
            - Ensemble of initial conditions:
              shape `(num_initial_conditions * sample_size, system_dimension)`

            where `sample_size = total_time - transient_time` if a transient is
            used, and `sample_size = total_time` otherwise.

            For one-dimensional systems with a single initial condition, the output
            is returned as a one-dimensional array of shape `(sample_size,)`.

        Raises
        ------
        ValueError
            - If `u` is incompatible with the system dimension.
            - If `parameters` does not match the expected number of parameters.
            - If `total_time` is not positive.
            - If `transient_time` is invalid.
        TypeError
            - If `u` is not a scalar or array-like numeric object.
            - If `parameters` is not a scalar or array-like numeric object.
            - If `total_time` is not an integer.
            - If `transient_time` is not an integer.

        Notes
        -----
        For ensembles, trajectories are concatenated along the first axis. To recover
        the individual trajectories, reshape the output as
        `(num_initial_conditions, sample_size, system_dimension)`.

        Examples
        --------
        >>> u0 = np.array([0.1, 0.2])
        >>> ts = system.trajectory(u0, 5000, parameters=[0.5, 1.0])

        >>> ics = np.random.rand(100, 2)
        >>> ts = system.trajectory(ics, 10000, parameters=[1.0, 0.1], transient_time=1000)
        >>> ts = ts.reshape(100, 9000, 2)
        """
        u_arr: NDArray[np.float64] = validate_initial_conditions(
            u,
            self.__system_dimension,
            allow_ensemble=True,
        )

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        parameters_arr: NDArray[np.float64] = validate_parameters(
            parameters,
            self.__number_of_parameters,
        )

        validate_positive(total_time, "total_time", Integral)
        validate_transient_time(transient_time, total_time, type_=Integral)

        if u_arr.ndim == 1:
            trajectory = generate_trajectory(
                u=u_arr,
                parameters=parameters_arr,
                total_time=total_time,
                mapping=self.__mapping,
                transient_time=transient_time,
            )
            if self.__system_dimension == 1:
                return trajectory[:, 0]
            return trajectory

        return ensemble_trajectories(
            u=u_arr,
            parameters=parameters_arr,
            total_time=total_time,
            mapping=self.__mapping,
            transient_time=transient_time,
        )

    def bifurcation_diagram(
        self,
        u: numeric_like_t,
        param_index: int,
        param_range: NDArray[np.float64] | tuple[float, float, int],
        total_time: int_t,
        parameters: numeric_like_t | None = None,
        transient_time: int_t = 0,
        continuation: bool = False,
        return_last_state: bool = False,
        observable_index: int_t = 0,
    ) -> (
        tuple[NDArray[np.float64], NDArray[np.float64]]
        | tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
    ):
        """
        Compute a bifurcation diagram by varying one system parameter and recording
        a scalar observable along the resulting trajectories.

        Parameters
        ----------
        u : numeric_like_t
            Initial condition of shape `(system_dimension,)`.
        param_index : int_t
            Index of the parameter to be varied.
        param_range : NDArray[np.float64] | tuple[float, float, int]
            Parameter values used in the sweep. It can be either:
            - a 1D array of parameter values, or
            - a tuple `(start, stop, num_points)` passed to `numpy.linspace`.
        total_time : int_t
            Number of iterations computed for each parameter value.
        parameters : numeric_like_t | None, optional
            Base parameter array. The entry at `param_index` is overwritten during
            the sweep.
        transient_time : int_t, optional
            Number of initial iterations discarded for each parameter value.
            Default is `0`.
        continuation : bool, optional
            If True, use the final state from the previous parameter value as the
            initial condition for the next one.
        return_last_state : bool, optional
            If True, also return the final state obtained at the last parameter value.
        observable_index : int_t, optional
            Index of the state coordinate used as the observable in the bifurcation
            diagram. Default is `0`.

        Returns
        -------
        tuple
            If `return_last_state` is False:
                `(param_values, results)`
            If `return_last_state` is True:
                `(param_values, results, last_state)`

            Here:
            - `param_values` has shape `(num_points,)`
            - `results` has shape `(num_points, sample_size)`
            - `last_state` has shape `(system_dimension,)`

        Raises
        ------
        ValueError
            - If `u` is not compatible with the system dimension.
            - If `parameters` does not match the expected number of parameters.
            - If `param_index` is out of bounds.
            - If `observable_index` is out of bounds.
            - If `total_time` is negative.
            - If `transient_time` is invalid.
        TypeError
            - If `u` is not a scalar or array-like numeric object.
            - If `parameters` is not a scalar or array-like numeric object.
            - If `param_index`, `observable_index`, `total_time`, or `transient_time`
              are not integers.
            - If `continuation` or `return_last_state` are not booleans.

        Notes
        -----
        The varied parameter is always stored in the full parameter array passed to
        the low-level routine. If the system has stored parameters and `parameters`
        is not provided, the stored array is used as the base parameter vector.
        """
        u_arr: NDArray[np.float64] = validate_initial_conditions(
            u, self.__system_dimension, allow_ensemble=False
        )

        validate_non_negative(param_index, "param_index", Integral)
        if param_index >= self.__number_of_parameters:
            raise ValueError(
                f"param_index {param_index} out of bounds for system with "
                f"{self.__number_of_parameters} parameters"
            )

        param_values: NDArray[np.float64] = validate_and_convert_param_range(
            param_range
        )

        validate_non_negative(total_time, "total_time", Integral)
        validate_transient_time(transient_time, total_time, Integral)

        validate_non_negative(observable_index, "observable_index", Integral)
        if observable_index >= self.__system_dimension:
            raise ValueError(
                f"observable_index {observable_index} out of bounds for system dimension "
                f"{self.__system_dimension}"
            )

        if not isinstance(continuation, bool):
            raise TypeError("continuation must be a boolean")

        if not isinstance(return_last_state, bool):
            raise TypeError("return_last_state must be a boolean")

        if self.__parameters is not None:
            if parameters is None:
                parameters = self.__parameters
            else:
                parameters = validate_parameters(
                    parameters, self.__number_of_parameters - 1
                )
        else:
            if parameters is not None:
                parameters = validate_parameters(
                    parameters, self.__number_of_parameters - 1
                )
                parameters = np.insert(parameters, param_index, 0)
            else:
                parameters = np.array([0], dtype=np.float64)

        def observable_fn(x: NDArray[np.float64]) -> numeric_t:
            return x[observable_index]

        return bifurcation_diagram(
            u=u_arr,
            parameters=parameters,
            param_index=param_index,
            param_range=param_values,
            total_time=total_time,
            mapping=self.__mapping,
            transient_time=transient_time,
            continuation=continuation,
            return_last_state=return_last_state,
            observable_fn=observable_fn,
        )

    def period(
        self,
        u: numeric_like_t,
        max_time: int_t = 10000,
        parameters: numeric_like_t | None = None,
        transient_time: int_t | None = None,
        tolerance: numeric_t = 1e-10,
        min_period: int = 1,
        max_period: int = 1000,
        stability_checks: int = 3,
    ) -> int:
        """
        Estimate the period of an orbit from state recurrences.

        The method searches for repeated returns of the trajectory to its initial
        post-transient state within a prescribed tolerance. A candidate period is
        accepted only after repeated consistent detections.

        Parameters
        ----------
        u : numeric_like_t
            Initial condition of shape `(system_dimension,)`.
        max_time : int_t, optional
            Maximum number of iterations used in the search.
        parameters : numeric_like_t | None, optional
            System parameters passed to the mapping function. If None, the stored
            system parameters are used.
        transient_time : int_t | None, optional
            Number of initial iterations discarded before the search.
        tolerance : numeric_t, optional
            Absolute tolerance used to detect recurrences.
        min_period : int, optional
            Minimum admissible period.
        max_period : int, optional
            Maximum admissible period.
        stability_checks : int, optional
            Number of identical consecutive detections required before accepting a
            period.

        Returns
        -------
        int
            Detected period.

            - A positive integer indicates a periodic orbit.
            - `1` indicates a fixed point.
            - `-1` indicates that no valid period was detected within the search
              window.

        Raises
        ------
        ValueError
            - If `u` is not compatible with the system dimension.
            - If `parameters` does not match the expected number of parameters.
            - If `max_time` is negative.
            - If `transient_time` is invalid.
            - If `tolerance <= 0`.
            - If `min_period < 1`.
            - If `max_period < 1`.
            - If `max_period < min_period`.
            - If `stability_checks < 1`.
        TypeError
            - If `u` is not a scalar or array-like numeric object.
            - If `parameters` is not a scalar or array-like numeric object.
            - If `max_time` is not an integer.
            - If `transient_time` is not an integer.
            - If `min_period`, `max_period`, or `stability_checks` are not integers.

        Notes
        -----
        For reliable detection, `max_time` should be significantly larger than the
        expected period.
        """
        u_arr: NDArray[np.float64] = validate_initial_conditions(
            u, self.__system_dimension, allow_ensemble=False
        )

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        parameters_arr: NDArray[np.float64] = validate_parameters(
            parameters, self.__number_of_parameters
        )

        validate_non_negative(max_time, "max_time", Integral)
        validate_transient_time(transient_time, max_time, Integral)

        validate_positive(min_period, "min_period", Integral)
        validate_positive(max_period, "max_period", Integral)
        if max_period < min_period:
            raise ValueError("max_period must be greater than or equal to min_period")

        validate_positive(stability_checks, "stability_checks", Integral)
        validate_positive(tolerance, "tolerance", Real)

        return period_counter(
            u=u_arr,
            parameters=parameters_arr,
            mapping=self.__mapping,
            total_time=max_time,
            transient_time=transient_time,
            tolerance=tolerance,
            min_period=min_period,
            max_period=max_period,
            stability_checks=stability_checks,
        )

    def is_periodic(
        self,
        u: numeric_like_t,
        period: int,
        parameters: numeric_like_t | None = None,
        tolerance: numeric_t = 1e-10,
        transient_time: int_t | None = None,
    ) -> bool:
        """
        Check whether an initial condition belongs to a periodic orbit of a given period.

        Parameters
        ----------
        u : numeric_like_t
            Initial condition of shape `(system_dimension,)`.
        period : int
            Period to test.
        parameters : numeric_like_t | None, optional
            System parameters passed to the mapping function. If None, the stored
            system parameters are used.
        tolerance : numeric_t, optional
            Absolute tolerance used in the periodicity check.
        transient_time : int_t | None, optional
            Number of initial iterations discarded before testing periodicity.

        Returns
        -------
        bool
            True if the orbit returns to the same state after `period` iterations,
            within the specified tolerance. False otherwise.
        """
        u_arr: NDArray[np.float64] = validate_initial_conditions(
            u, self.__system_dimension, allow_ensemble=False
        )

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        parameters_arr: NDArray[np.float64] = validate_parameters(
            parameters, self.__number_of_parameters
        )

        validate_positive(period, "period", Integral)
        validate_positive(tolerance, "tolerance", Real)

        if transient_time is not None:
            validate_non_negative(transient_time, "transient_time", Integral)

        return is_periodic(
            u=u_arr,
            parameters=parameters_arr,
            mapping=self.__mapping,
            period=period,
            tolerance=tolerance,
            transient_time=transient_time,
        )

    def find_periodic_orbit(
        self,
        grid_points: numeric_like_t,
        period: int,
        parameters: numeric_like_t | None = None,
        tolerance: numeric_t = 1e-5,
        max_iter: int_t = 1000,
        convergence_threshold: numeric_t = 1e-15,
        tolerance_decay_factor: numeric_t = 0.5,
        verbose: bool = False,
        symmetry_line: Callable[..., NDArray[np.float64]] | None = None,
        axis: int | None = None,
        transient_time: int_t | None = None,
    ) -> NDArray[np.float64]:
        """
        Find a periodic orbit through iterative grid refinement.

        Parameters
        ----------
        grid_points : numeric_like_t
            Initial search set.

            - If `symmetry_line is None`, it must be a 3D array of shape
              `(grid_size_x, grid_size_y, 2)`.
            - If `symmetry_line is not None`, it must be a 1D array containing the
              coordinates sampled along the chosen symmetry parametrization.
        period : int
            Period of the orbit to search for.
        parameters : numeric_like_t | None, optional
            System parameters passed to the mapping function. If None, the stored
            system parameters are used.
        tolerance : numeric_t, optional
            Initial periodicity tolerance.
        max_iter : int_t, optional
            Maximum number of refinement iterations.
        convergence_threshold : numeric_t, optional
            Convergence threshold for orbit displacement and search-box size.
        tolerance_decay_factor : numeric_t, optional
            Multiplicative factor used to reduce the tolerance after each iteration.
            Must satisfy `0 < tolerance_decay_factor < 1`.
        verbose : bool, optional
            If True, print iteration diagnostics.
        symmetry_line : Callable[..., NDArray[np.float64]] | None, optional
            Symmetry-line or symmetry-curve function. If provided, the search is
            restricted to that line or curve.
        axis : int | None, optional
            Axis convention for the symmetry-line search. Must be 0 or 1 when
            `symmetry_line` is provided.
        transient_time : int_t | None, optional
            Number of initial iterations discarded before testing periodicity.

        Returns
        -------
        NDArray[np.float64]
            Approximation of the periodic orbit.

        Raises
        ------
        ValueError
            - If the system dimension is not 2.
            - If `grid_points` has invalid shape.
            - If `period < 1`.
            - If `tolerance <= 0`.
            - If `max_iter < 1`.
            - If `convergence_threshold <= 0`.
            - If `tolerance_decay_factor` is not in `(0, 1)`.
            - If `symmetry_line` is provided and `axis` is missing.
            - If `axis` is not 0 or 1.
            - If `transient_time` is negative.
        TypeError
            - If `grid_points` cannot be interpreted as an array.
            - If `parameters` is not a scalar or array-like numeric object.
            - If `symmetry_line` is not callable.
            - If `period`, `max_iter`, or `axis` are not integers.
        """
        if self.__system_dimension != 2:
            raise ValueError("find_periodic_orbit is only implemented for 2D systems")

        if symmetry_line is not None and not callable(symmetry_line):
            raise TypeError("symmetry_line must be callable")

        grid_points_arr = np.asarray(grid_points, dtype=np.float64)

        if symmetry_line is None:
            if grid_points_arr.ndim != 3 or grid_points_arr.shape[2] != 2:
                raise ValueError(
                    "grid_points must have shape (grid_size_x, grid_size_y, 2) when symmetry_line is None"
                )
        else:
            if grid_points_arr.ndim != 1:
                raise ValueError(
                    "grid_points must be a 1D array when symmetry_line is provided"
                )

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        parameters_arr: NDArray[np.float64] = validate_parameters(
            parameters, self.__number_of_parameters
        )

        validate_positive(period, "period", Integral)
        validate_positive(tolerance, "tolerance", Real)
        validate_positive(max_iter, "max_iter", Integral)
        validate_positive(convergence_threshold, "convergence_threshold", Real)

        validate_positive(tolerance_decay_factor, "tolerance_decay_factor", Real)
        if tolerance_decay_factor >= 1:
            raise ValueError("tolerance_decay_factor must satisfy 0 < value < 1")

        if transient_time is not None:
            validate_non_negative(transient_time, "transient_time", Integral)

        if symmetry_line is not None:
            if axis is None:
                raise ValueError(
                    "axis must be provided when symmetry_line is specified"
                )
            if not isinstance(axis, Integral):
                raise TypeError("axis must be an integer")
            if axis not in (0, 1):
                raise ValueError("axis must be 0 or 1")

            axis_int = int(axis)

            return find_periodic_orbit_symmetry_line(
                points=grid_points_arr,
                parameters=parameters_arr,
                mapping=self.__mapping,
                period=period,
                func=symmetry_line,
                axis=axis_int,
                tolerance=tolerance,
                max_iter=max_iter,
                convergence_threshold=convergence_threshold,
                tolerance_decay_factor=tolerance_decay_factor,
                verbose=verbose,
                transient_time=transient_time,
            )

        return find_periodic_orbit(
            grid_points=grid_points_arr,
            parameters=parameters_arr,
            mapping=self.__mapping,
            period=period,
            tolerance=tolerance,
            max_iter=max_iter,
            convergence_threshold=convergence_threshold,
            tolerance_decay_factor=tolerance_decay_factor,
            verbose=verbose,
            transient_time=transient_time,
        )

    def eigenvalues_and_eigenvectors(
        self,
        u: numeric_like_t,
        period: int,
        parameters: numeric_like_t | None = None,
        normalize: bool = True,
        sort_by_magnitude: bool = True,
    ) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
        """
        Compute the eigenvalues and eigenvectors of the monodromy matrix of a
        periodic orbit of this discrete-time system.

        Parameters
        ----------
        u : numeric_like_t
            Initial condition of shape `(system_dimension,)`.
        period : int_t
            Period of the orbit.
        parameters : numeric_like_t | None, optional
            System parameters. If None, the stored system parameters are used.
        normalize : bool, optional
            If True, normalize the returned eigenvectors to unit Euclidean norm.
        sort_by_magnitude : bool, optional
            If True, sort the eigenpairs by decreasing eigenvalue magnitude.

        Returns
        -------
        tuple[NDArray[np.complex128], NDArray[np.complex128]]
            A tuple `(eigenvalues, eigenvectors)` where

            - `eigenvalues` has shape `(system_dimension,)`
            - `eigenvectors` has shape `(system_dimension, system_dimension)`

            Each column of `eigenvectors` is an eigenvector associated with the
            eigenvalue in the same position.

        Raises
        ------
        ValueError
            - If `u` is incompatible with the system dimension.
            - If `parameters` does not match the expected number of parameters.
            - If `period` is not positive.
            - If the system does not provide a Jacobian.
        TypeError
            - If `u` is not a scalar or array-like numeric object.
            - If `parameters` is not a scalar or array-like numeric object.
            - If `period` is not an integer.
            - If `normalize` is not a boolean.
            - If `sort_by_magnitude` is not a boolean.
        """
        if self.__jacobian is None:
            raise ValueError(
                "eigenvalues_and_eigenvectors requires a Jacobian function."
            )

        u_arr: NDArray[np.float64] = validate_initial_conditions(
            u,
            self.__system_dimension,
            allow_ensemble=False,
        )

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        parameters_arr: NDArray[np.float64] = validate_parameters(
            parameters,
            self.__number_of_parameters,
        )

        validate_positive(period, "period", Integral)

        if not isinstance(normalize, bool):
            raise TypeError("normalize must be a boolean")

        if not isinstance(sort_by_magnitude, bool):
            raise TypeError("sort_by_magnitude must be a boolean")

        return eigenvalues_and_eigenvectors(
            u=u_arr,
            parameters=parameters_arr,
            mapping=self.__mapping,
            jacobian=self.__jacobian,
            period=period,
            normalize=normalize,
            sort_by_magnitude=sort_by_magnitude,
        )

    def classify_stability(
        self,
        u: numeric_like_t,
        period: int,
        parameters: numeric_like_t | None = None,
        threshold: numeric_t = 1.0,
        tol: numeric_t = 1e-8,
    ) -> dict[str, str | NDArray[np.complex128]]:
        """
        Classify the local linear stability of a 2D periodic orbit of this
        discrete-time system.

        Parameters
        ----------
        u : numeric_like_t
            Initial condition of shape `(2,)`.
        period : int_t
            Period of the orbit.
        parameters : numeric_like_t | None, optional
            System parameters. If None, the stored system parameters are used.
        threshold : numeric_t, optional
            Reference radius used to separate contracting and expanding
            multipliers. For standard discrete-time stability analysis this should
            usually remain equal to `1.0`.
        tol : numeric_t, optional
            Numerical tolerance used when deciding whether a multiplier lies on
            the threshold.

        Returns
        -------
        dict[str, str | NDArray[np.complex128]]
            Dictionary with keys

            - `"classification"` : stability label
            - `"eigenvalues"` : Floquet multipliers
            - `"eigenvectors"` : corresponding eigenvectors

        Raises
        ------
        ValueError
            - If the system dimension is not 2.
            - If `u` is incompatible with the system dimension.
            - If `parameters` does not match the expected number of parameters.
            - If `period` is not positive.
            - If `threshold` is negative.
            - If `tol` is negative.
            - If the system does not provide a Jacobian.
        TypeError
            - If `u` is not a scalar or array-like numeric object.
            - If `parameters` is not a scalar or array-like numeric object.
            - If `period` is not an integer.
            - If `threshold` is not numeric.
            - If `tol` is not numeric.
        """
        if self.__system_dimension != 2:
            raise ValueError("classify_stability is only implemented for 2D systems")

        if self.__jacobian is None:
            raise ValueError("classify_stability requires a Jacobian function.")

        u_arr: NDArray[np.float64] = validate_initial_conditions(
            u,
            self.__system_dimension,
            allow_ensemble=False,
        )

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        parameters_arr: NDArray[np.float64] = validate_parameters(
            parameters,
            self.__number_of_parameters,
        )

        validate_positive(period, "period", Integral)
        validate_non_negative(threshold, "threshold", Real)
        validate_non_negative(tol, "tol", Real)

        return classify_stability(
            u=u_arr,
            parameters=parameters_arr,
            mapping=self.__mapping,
            jacobian=self.__jacobian,
            period=period,
            threshold=threshold,
            tol=tol,
        )

    def manifold(
        self,
        u: numeric_like_t,
        period: int,
        parameters: numeric_like_t | None = None,
        delta: numeric_t = 1e-4,
        n_points: int | tuple[int, int] = 100,
        iter_time: int | tuple[int, int] = 100,
        stability: Literal["stable", "unstable"] = "unstable",
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Compute the two branches of the stable or unstable manifold of a 2D saddle
        periodic orbit.

        Parameters
        ----------
        u : numeric_like_t
            Initial condition on the periodic orbit, with shape `(2,)`.
        period : int
            Period of the orbit.
        parameters : numeric_like_t | None, optional
            System parameters. If None, the stored system parameters are used.
        delta : numeric_t, optional
            Initial displacement magnitude used to seed the manifold branches.
        n_points : int | tuple[int, int], optional
            Number of seed points used on each manifold branch. If an integer is
            given, the same value is used for both branches.
        iter_time : int_t | tuple[int_t, int_t], optional
            Number of iterations used to evolve each branch. If an integer is given,
            the same value is used for both branches.
        stability : {"stable", "unstable"}, optional
            Which invariant manifold to compute.

        Returns
        -------
        tuple[NDArray[np.float64], NDArray[np.float64]]
            Two arrays containing the `+` and `-` manifold branches.

        Raises
        ------
        ValueError
            - If the system is not 2-dimensional.
            - If `u` is not compatible with the system dimension.
            - If `parameters` does not match the expected number of parameters.
            - If `period` is not positive.
            - If `delta` is not positive.
            - If `n_points` is not a positive integer or a tuple of two positive integers.
            - If `iter_time` is not a positive integer or a tuple of two positive integers.
            - If `stability` is invalid.
        RuntimeError
            - If `stability="stable"` but no backward mapping is available.
            - If no Jacobian is available.

        Notes
        -----
        This method is only implemented for 2D systems and requires the selected
        periodic orbit to be a saddle.
        """
        if self.__system_dimension != 2:
            raise ValueError("manifold is only implemented for 2D systems")

        if self.__jacobian is None:
            raise RuntimeError("Jacobian function must be provided")

        if stability not in ("stable", "unstable"):
            raise ValueError("stability must be either 'stable' or 'unstable'")

        if stability == "stable" and self.__backwards_mapping is None:
            raise RuntimeError(
                "backward mapping function must be provided to compute the stable manifold"
            )

        u_arr: NDArray[np.float64] = validate_initial_conditions(
            u, self.__system_dimension, allow_ensemble=False
        )

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        parameters_arr: NDArray[np.float64] = validate_parameters(
            parameters, self.__number_of_parameters
        )

        validate_positive(period, "period", Integral)
        validate_positive(delta, "delta", Real)

        if isinstance(n_points, Integral):
            n_points_tuple = (int(n_points), int(n_points))
        elif isinstance(n_points, tuple):
            if len(n_points) != 2:
                raise ValueError("n_points must be an int or a tuple of length 2")
            n_points_tuple = (int(n_points[0]), int(n_points[1]))
        else:
            raise TypeError("n_points must be an int or a tuple of two ints")

        if n_points_tuple[0] < 1 or n_points_tuple[1] < 1:
            raise ValueError("all n_points values must be positive")

        if isinstance(iter_time, Integral):
            iter_time_tuple = (int(iter_time), int(iter_time))
        elif isinstance(iter_time, tuple) or isinstance(iter_time, list):
            if len(iter_time) != 2:
                raise ValueError("iter_time must be an int or a tuple of length 2")
            iter_time_tuple = (int(iter_time[0]), int(iter_time[1]))
        else:
            raise TypeError("iter_time must be an int or a tuple of two ints")

        if iter_time_tuple[0] < 1 or iter_time_tuple[1] < 1:
            raise ValueError("all iter_time values must be positive")

        backward_mapping = self.__backwards_mapping
        if backward_mapping is None:
            backward_mapping = self.__mapping

        return calculate_manifolds(
            u=u_arr,
            parameters=parameters_arr,
            forward_mapping=self.__mapping,
            backward_mapping=backward_mapping,
            jacobian=self.__jacobian,
            period=period,
            delta=delta,
            n_points=n_points_tuple,
            iter_time=iter_time_tuple,
            stability=stability,
        )

    def rotation_number(
        self,
        u: numeric_like_t,
        total_time: int_t,
        parameters: numeric_like_t | None = None,
        mod: numeric_t = 1.0,
    ) -> float:
        """
        Compute the rotation number of a trajectory.

        The rotation number is estimated as the time average of the wrapped
        increment of the first coordinate modulo ``mod``.

        Parameters
        ----------
        u : numeric_like_t
            Initial condition of shape ``(system_dimension,)``.
        total_time : int_t
            Number of iterations used in the average.
        parameters : numeric_like_t | None, optional
            System parameters passed to the mapping function. If ``None``, the
            stored system parameters are used.
        mod : numeric_t, optional
            Period used to wrap the increment of the first coordinate.
            Must be positive. The default is ``1.0``.

        Returns
        -------
        float
            Estimated rotation number.

        Raises
        ------
        ValueError
            - If the system dimension is less than 1.
            - If ``u`` is not compatible with the system dimension.
            - If ``parameters`` does not match the expected number of parameters.
            - If ``total_time`` is not positive.
            - If ``mod`` is not positive.
        TypeError
            - If ``u`` is not a scalar or array-like numeric object.
            - If ``parameters`` is not a scalar or array-like numeric object.
            - If ``total_time`` is not an integer.
            - If ``mod`` is not a real number.

        Notes
        -----
        This wrapper validates the inputs and forwards the computation to the
        low-level ``rotation_number`` routine.
        """
        if self.__system_dimension < 1:
            raise ValueError(
                f"rotation_number requires system_dimension >= 1, got {self.__system_dimension}."
            )

        u_arr: NDArray[np.float64] = validate_initial_conditions(
            u, self.__system_dimension, allow_ensemble=False
        )

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        parameters_arr: NDArray[np.float64] = validate_parameters(
            parameters, self.__number_of_parameters
        )

        validate_positive(total_time, "total_time", Integral)
        validate_positive(mod, "mod", Real)

        return rotation_number(
            u=u_arr,
            parameters=parameters_arr,
            total_time=total_time,
            mapping=self.__mapping,
            mod=mod,
        )

    def escape_analysis(
        self,
        u: numeric_like_t,
        max_time: int_t,
        exits: NDArray[np.float64]
        | Sequence[Sequence[float]]
        | Sequence[NDArray[np.float64]],
        parameters: numeric_like_t | None = None,
        escape: str = "entering",
        hole_size: numeric_t | None = None,
    ) -> Tuple[int, int_t]:
        """
        Compute the escape index and escape time for a single trajectory.

        Parameters
        ----------
        u : numeric_like_t
            Initial condition of shape `(system_dimension,)`.
        max_time : int_t
            Maximum number of iterations.
        exits : NDArray[np.float64] | Sequence[Sequence[float]] | Sequence[NDArray[np.float64]]
            Exit specification.

            If `escape == "entering"`:
                Exit centers, with shape `(n_exits, system_dimension)` or `(system_dimension,)`
                for a single exit center. Hyperrectangular holes are built around each center
                using `hole_size`.

            If `escape == "exiting"`:
                A bounded region of shape `(system_dimension, 2)`, where each row is
                `[lower, upper]` for one coordinate.
        parameters : numeric_like_t | None, optional
            System parameters. If None, stored system parameters are used.
        escape : str, optional
            Escape mode. Must be either `"entering"` or `"exiting"`.
        hole_size : numeric_t | None, optional
            Side length of the hyperrectangular holes used when `escape == "entering"`.

        Returns
        -------
        Tuple[int, int]
            If `escape == "entering"`:
                `(exit_index, escape_time)`

            If `escape == "exiting"`:
                `(face_index, escape_time)`

        Raises
        ------
        ValueError
            - If `u` is incompatible with the system dimension.
            - If `parameters` does not match the expected number of parameters.
            - If `max_time <= 0`.
            - If `escape` is not `"entering"` or `"exiting"`.
            - If `hole_size` is missing or not positive when `escape == "entering"`.
            - If `exits` has an invalid shape.
        TypeError
            - If `u` is not a scalar or array-like numeric object.
            - If `parameters` is not a scalar or array-like numeric object.
            - If `max_time` is not an integer.
        """
        u_arr: NDArray[np.float64] = validate_initial_conditions(
            u, self.__system_dimension, allow_ensemble=False
        )

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        parameters_arr: NDArray[np.float64] = validate_parameters(
            parameters, self.__number_of_parameters
        )

        validate_positive(max_time, "max_time", Integral)

        if escape not in ("entering", "exiting"):
            raise ValueError("escape must be either 'entering' or 'exiting'")

        exits_arr = np.asarray(exits, dtype=np.float64)

        if escape == "entering":
            if hole_size is None:
                raise ValueError("hole_size must be provided when escape='entering'")
            validate_positive(hole_size, "hole_size", Real)

            if exits_arr.ndim == 1:
                if exits_arr.shape[0] != self.__system_dimension:
                    raise ValueError(
                        f"single exit center must have length {self.__system_dimension}"
                    )
                exits_arr = exits_arr.reshape(1, self.__system_dimension)
            elif exits_arr.ndim == 2:
                if exits_arr.shape[1] != self.__system_dimension:
                    raise ValueError(
                        f"exit centers must have shape (n_exits, {self.__system_dimension})"
                    )
            else:
                raise ValueError(
                    "for escape='entering', exits must have shape "
                    "(system_dimension,) or (n_exits, system_dimension)"
                )

            lower = exits_arr - hole_size / 2.0
            upper = exits_arr + hole_size / 2.0
            holes = np.empty(
                (exits_arr.shape[0], self.__system_dimension, 2), dtype=np.float64
            )
            holes[:, :, 0] = lower
            holes[:, :, 1] = upper

            return escape_basin_and_time_entering(
                u=u_arr,
                parameters=parameters_arr,
                mapping=self.__mapping,
                max_time=max_time,
                exits=holes,
            )

        if exits_arr.ndim != 2 or exits_arr.shape != (self.__system_dimension, 2):
            raise ValueError(
                f"for escape='exiting', exits must have shape ({self.__system_dimension}, 2)"
            )

        if np.any(exits_arr[:, 0] > exits_arr[:, 1]):
            raise ValueError("each region limit must satisfy lower <= upper")

        return escape_time_exiting(
            u=u_arr,
            parameters=parameters_arr,
            mapping=self.__mapping,
            max_time=max_time,
            region_limits=exits_arr,
        )

    def survival_probability(
        self,
        escape_times: NDArray[np.integer] | Sequence[int],
        max_time: int_t,
        min_time: int_t = 1,
        time_step: int_t = 1,
    ) -> Tuple[NDArray[np.int64], NDArray[np.float64]]:
        """
        Compute the survival probability from an array of escape times.

        Parameters
        ----------
        escape_times : NDArray[np.integer] | Sequence[int]
            Escape times for an ensemble of trajectories.
        max_time : int_t
            Maximum evaluation time.
        min_time : int_t, optional
            Minimum evaluation time. Default is `1`.
        time_step : int_t, optional
            Step between consecutive evaluation times. Default is `1`.

        Returns
        -------
        Tuple[NDArray[np.int64], NDArray[np.float64]]
            Tuple `(t_values, survival_probs)`, where:

            - `t_values` contains the evaluation times,
            - `survival_probs` contains the corresponding survival probabilities.

        Raises
        ------
        ValueError
            - If `escape_times` is not one-dimensional.
            - If any escape time is smaller than `1`.
            - If `max_time <= min_time`.
            - If `time_step <= 0`.
        TypeError
            - If `escape_times` cannot be converted to an integer NumPy array.
            - If `max_time`, `min_time`, or `time_step` are not integers.
        """
        validate_positive(max_time, "max_time", Integral)
        validate_positive(min_time, "min_time", Integral)
        validate_positive(time_step, "time_step", Integral)

        try:
            escape_arr = np.asarray(escape_times, dtype=np.int64)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "escape_times must be convertible to a 1D integer array"
            ) from exc

        if escape_arr.ndim != 1:
            raise ValueError("escape_times must be a 1D array")

        if np.any(escape_arr < 1):
            raise ValueError("all escape_times must be >= 1")

        return survival_probability_core(
            escape_times=escape_arr,
            max_time=max_time,
            min_time=min_time,
            time_step=time_step,
        )

    def diffusion_coefficient(
        self,
        u: numeric_like_t,
        total_time: int_t,
        parameters: numeric_like_t | None = None,
        axis: int = 1,
    ) -> np.float64:
        """
        Compute the diffusion coefficient from an ensemble of trajectories.

        Parameters
        ----------
        u : numeric_like_t
            Ensemble of initial conditions with shape `(num_ic, system_dimension)`.
        total_time : int_t
            Number of iterations used in the transport estimate.
        parameters : numeric_like_t | None, optional
            System parameters passed to the mapping function. If None, the stored
            system parameters are used.
        axis : int, optional
            Coordinate index used in the displacement calculation.

        Returns
        -------
        np.float64
            Estimated diffusion coefficient.

        Raises
        ------
        ValueError
            If `u` is not a 2D array of valid initial conditions.
            If `parameters` does not match the expected number of parameters.
            If `total_time` is negative.
            If `axis` is not a valid coordinate index.
        TypeError
            If `u`, `parameters`, `total_time`, or `axis` have invalid types.
        """
        u_arr: NDArray[np.float64] = validate_initial_conditions(
            u, self.__system_dimension, allow_ensemble=True
        )

        if u_arr.ndim != 2:
            raise ValueError(
                f"u must be a 2D array of shape (num_ic, {self.__system_dimension})"
            )

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        parameters_arr: NDArray[np.float64] = validate_parameters(
            parameters, self.__number_of_parameters
        )

        validate_non_negative(total_time, "total_time", Integral)
        validate_axis(axis, self.__system_dimension)

        return diffusion_coefficient(
            u0=u_arr,
            parameters=parameters_arr,
            total_time=total_time,
            mapping=self.__mapping,
            axis=axis,
        )

    def average_in_time(
        self,
        u: numeric_like_t,
        total_time: int_t,
        parameters: numeric_like_t | None = None,
        sample_times: NDArray[np.integer] | Sequence[int] | None = None,
        axis: int = 1,
    ) -> NDArray[np.float64]:
        """
        Compute the ensemble average of one coordinate as a function of time.

        Parameters
        ----------
        u : numeric_like_t
            Ensemble of initial conditions with shape `(num_ic, system_dimension)`.
        total_time : int_t
            Total number of iterations.
        parameters : numeric_like_t | None, optional
            System parameters passed to the mapping function. If None, the stored
            system parameters are used.
        sample_times : NDArray[np.int32] | Sequence[int] | None, optional
            Sampling times at which the average is recorded. If None, all times from
            `1` to `total_time` are used.
        axis : int, optional
            Coordinate index whose ensemble average is computed.

        Returns
        -------
        NDArray[np.float64]
            Ensemble-average time series.
        """
        u_arr: NDArray[np.float64] = validate_initial_conditions(
            u, self.__system_dimension, allow_ensemble=True
        )

        if u_arr.ndim != 2:
            raise ValueError(
                f"u must be a 2D array of shape (num_ic, {self.__system_dimension})"
            )

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        parameters_arr: NDArray[np.float64] = validate_parameters(
            parameters, self.__number_of_parameters
        )

        validate_non_negative(total_time, "total_time", Integral)
        validate_axis(axis, self.__system_dimension)

        sample_times_arr: NDArray[np.int64] | None = None
        if sample_times is not None:
            sample_times_arr = validate_sample_times(sample_times, total_time)
        else:
            sample_times_arr = np.arange(1, total_time + 1, dtype=np.int64)

        return average_vs_time(
            u=u_arr,
            parameters=parameters_arr,
            total_time=total_time,
            mapping=self.__mapping,
            sample_times=sample_times_arr,
            axis=axis,
        )

    def cumulative_average(
        self,
        u: numeric_like_t,
        total_time: int_t,
        parameters: numeric_like_t | None = None,
        sample_times: NDArray[np.int32] | Sequence[int] | None = None,
        axis: int = 1,
    ) -> NDArray[np.float64]:
        """
        Compute the cumulative ensemble average of one coordinate as a function of time.

        Parameters
        ----------
        u : numeric_like_t
            Ensemble of initial conditions with shape `(num_ic, system_dimension)`.
        total_time : int_t
            Total number of iterations.
        parameters : numeric_like_t | None, optional
            System parameters passed to the mapping function. If None, the stored
            system parameters are used.
        sample_times : NDArray[np.int32] | Sequence[int] | None, optional
            Sampling times at which the cumulative average is recorded. If None, all
            times from `1` to `total_time` are used.
        axis : int, optional
            Coordinate index whose cumulative ensemble average is computed.

        Returns
        -------
        NDArray[np.float64]
            Cumulative ensemble-average time series.
        """
        u_arr: NDArray[np.float64] = validate_initial_conditions(
            u, self.__system_dimension, allow_ensemble=True
        )

        if u_arr.ndim != 2:
            raise ValueError(
                f"u must be a 2D array of shape (num_ic, {self.__system_dimension})"
            )

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        parameters_arr: NDArray[np.float64] = validate_parameters(
            parameters, self.__number_of_parameters
        )

        validate_non_negative(total_time, "total_time", Integral)
        validate_axis(axis, self.__system_dimension)

        sample_times_arr: NDArray[np.int64] | None = None
        if sample_times is not None:
            sample_times_arr = validate_sample_times(sample_times, total_time)
        else:
            sample_times_arr = np.arange(1, total_time + 1, dtype=np.int64)

        return cumulative_average_vs_time(
            u=u_arr,
            parameters=parameters_arr,
            total_time=total_time,
            mapping=self.__mapping,
            sample_times=sample_times_arr,
            axis=axis,
        )

    def root_mean_squared(
        self,
        u: numeric_like_t,
        total_time: int_t,
        parameters: numeric_like_t | None = None,
        sample_times: NDArray[np.int32] | Sequence[int] | None = None,
        axis: int = 1,
    ) -> NDArray[np.float64]:
        """
        Compute the root-mean-squared value of one coordinate as a function of time.

        Parameters
        ----------
        u : numeric_like_t
            Ensemble of initial conditions with shape `(num_ic, system_dimension)`.
        total_time : int_t
            Total number of iterations.
        parameters : numeric_like_t | None, optional
            System parameters passed to the mapping function. If None, the stored
            system parameters are used.
        sample_times : NDArray[np.int32] | Sequence[int] | None, optional
            Sampling times at which the RMS value is recorded. If None, all times
            from `1` to `total_time` are used.
        axis : int, optional
            Coordinate index whose RMS value is computed.

        Returns
        -------
        NDArray[np.float64]
            RMS time series.
        """
        u_arr: NDArray[np.float64] = validate_initial_conditions(
            u, self.__system_dimension, allow_ensemble=True
        )

        if u_arr.ndim != 2:
            raise ValueError(
                f"u must be a 2D array of shape (num_ic, {self.__system_dimension})"
            )

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        parameters_arr: NDArray[np.float64] = validate_parameters(
            parameters, self.__number_of_parameters
        )

        validate_non_negative(total_time, "total_time", Integral)
        validate_axis(axis, self.__system_dimension)

        sample_times_arr: NDArray[np.int64] | None = None
        if sample_times is not None:
            sample_times_arr = validate_sample_times(sample_times, total_time)
        else:
            sample_times_arr = np.arange(1, total_time + 1, dtype=np.int64)

        return root_mean_squared(
            u=u_arr,
            parameters=parameters_arr,
            total_time=total_time,
            mapping=self.__mapping,
            sample_times=sample_times_arr,
            axis=axis,
        )

    def mean_squared_displacement(
        self,
        u: numeric_like_t,
        total_time: int_t,
        parameters: numeric_like_t | None = None,
        sample_times: NDArray[np.int32] | Sequence[int] | None = None,
        axis: int = 1,
    ) -> NDArray[np.float64]:
        """
        Compute the mean squared displacement of one coordinate as a function of time.

        Parameters
        ----------
        u : numeric_like_t
            Ensemble of initial conditions with shape `(num_ic, system_dimension)`.
        total_time : int_t
            Total number of iterations.
        parameters : numeric_like_t | None, optional
            System parameters passed to the mapping function. If None, the stored
            system parameters are used.
        sample_times : NDArray[np.int32] | Sequence[int] | None, optional
            Sampling times at which the MSD is recorded. If None, all times from
            `1` to `total_time` are used.
        axis : int, optional
            Coordinate index used in the displacement calculation.

        Returns
        -------
        NDArray[np.float64]
            Mean-squared-displacement time series.
        """
        u_arr: NDArray[np.float64] = validate_initial_conditions(
            u, self.__system_dimension, allow_ensemble=True
        )

        if u_arr.ndim != 2:
            raise ValueError(
                f"u must be a 2D array of shape (num_ic, {self.__system_dimension})"
            )

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        parameters_arr: NDArray[np.float64] = validate_parameters(
            parameters, self.__number_of_parameters
        )

        validate_non_negative(total_time, "total_time", Integral)
        validate_axis(axis, self.__system_dimension)

        sample_times_arr: NDArray[np.int64] | None = None
        if sample_times is not None:
            sample_times_arr = validate_sample_times(sample_times, total_time)
        else:
            sample_times_arr = np.arange(1, total_time + 1, dtype=np.int64)

        return mean_squared_displacement(
            u0=u_arr,
            parameters=parameters_arr,
            total_time=total_time,
            mapping=self.__mapping,
            sample_times=sample_times_arr,
            axis=axis,
        )

    def ensemble_time_average(
        self,
        u: numeric_like_t,
        total_time: int_t,
        parameters: numeric_like_t | None = None,
        axis: int = 1,
    ) -> NDArray[np.float64]:
        """
        Compute the centered time average for each trajectory in an ensemble.

        Parameters
        ----------
        u : numeric_like_t
            Ensemble of initial conditions with shape `(num_ic, system_dimension)`.
        total_time : int_t
            Total number of iterations.
        parameters : numeric_like_t | None, optional
            System parameters passed to the mapping function. If None, the stored
            system parameters are used.
        axis : int, optional
            Coordinate index used in the time average.

        Returns
        -------
        NDArray[np.float64]
            One centered time-average value for each initial condition.
        """
        u_arr: NDArray[np.float64] = validate_initial_conditions(
            u, self.__system_dimension, allow_ensemble=True
        )

        if u_arr.ndim != 2:
            raise ValueError(
                f"u must be a 2D array of shape (num_ic, {self.__system_dimension})"
            )

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        parameters_arr: NDArray[np.float64] = validate_parameters(
            parameters, self.__number_of_parameters
        )

        validate_non_negative(total_time, "total_time", Integral)
        validate_axis(axis, self.__system_dimension)

        return ensemble_time_average(
            u=u_arr,
            parameters=parameters_arr,
            mapping=self.__mapping,
            total_time=total_time,
            axis=axis,
        )

    def recurrence_times(
        self,
        u: numeric_like_t,
        total_time: int_t,
        parameters: numeric_like_t | None = None,
        eps: numeric_t = 1e-2,
        transient_time: int_t | None = None,
    ) -> NDArray[np.float64]:
        """
        Compute recurrence times to an `eps`-neighborhood of the reference point.

        Parameters
        ----------
        u : numeric_like_t
            Initial condition of shape `(system_dimension,)`.
        total_time : int_t
            Total number of iterations used to detect recurrences.
        parameters : numeric_like_t | None, optional
            System parameters passed to the mapping function. If None, the stored
            system parameters are used.
        eps : numeric_t, optional
            Side length of the recurrence neighborhood.
        transient_time : int_t | None, optional
            Number of initial iterations discarded before defining the recurrence box.

        Returns
        -------
        NDArray[np.float64]
            Array containing the recurrence times between successive returns to the
            `eps`-neighborhood.
        """
        u_arr: NDArray[np.float64] = validate_initial_conditions(
            u, self.__system_dimension, allow_ensemble=False
        )

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        parameters_arr: NDArray[np.float64] = validate_parameters(
            parameters, self.__number_of_parameters
        )

        validate_non_negative(total_time, "total_time", Integral)
        validate_transient_time(transient_time, total_time, Integral)
        validate_non_negative(eps, "eps", Real)

        return recurrence_times(
            u=u_arr,
            parameters=parameters_arr,
            total_time=total_time,
            mapping=self.__mapping,
            eps=eps,
            transient_time=transient_time,
        )

    def dig(
        self,
        u: numeric_like_t,
        total_time: int_t,
        parameters: numeric_like_t | None = None,
        func: observable_t = lambda x: np.cos(2 * np.pi * x[:, 0]),
        transient_time: int_t | None = None,
    ) -> float:
        """
        Compute the number of correct decimal digits in the convergence of a
        weighted Birkhoff average.

        This diagnostic compares weighted Birkhoff averages computed over two
        consecutive halves of the trajectory. Larger values indicate better
        convergence of the observable average and are typically associated with
        more regular dynamics.

        Parameters
        ----------
        u : numeric_like_t
            Initial condition of shape `(system_dimension,)`.
        total_time : int_t
            Total number of iterations used in the computation.
            If an odd value is provided, it is increased by one internally so that
            the trajectory can be split into two equal halves.
        parameters : numeric_like_t | None, optional
            System parameters passed to the mapping function. If None, the stored
            system parameters are used.
        func : observable_t, optional
            Observable function applied to the generated trajectory.
            It must accept a 2D array of shape `(n, system_dimension)` and return
            either:
            - a 1D array of shape `(n,)`, or
            - a scalar value.
            The default observable is `cos(2π x_0)`.
        transient_time : int_t | None, optional
            Number of initial iterations discarded before the computation.

        Returns
        -------
        float
            Weighted-Birkhoff convergence indicator defined as
            `-log10(|WB_0 - WB_1|)`, where `WB_0` and `WB_1` are the weighted
            Birkhoff averages over the first and second halves of the trajectory.

        Raises
        ------
        ValueError
            - If `u` is not compatible with the system dimension.
            - If `parameters` does not match the expected number of parameters.
            - If `total_time` is negative.
            - If `transient_time` is invalid.
            - If `func` does not return either a scalar or a 1D array.
        TypeError
            - If `u` is not a scalar or array-like numeric object.
            - If `parameters` is not a scalar or array-like numeric object.
            - If `total_time` is not an integer.
            - If `transient_time` is not an integer.
            - If `func` is not callable.

        Notes
        -----
        The computation uses a weighted Birkhoff average on two consecutive
        trajectory segments of equal length. The observable is evaluated on the
        generated trajectory after the transient, if any.

        Examples
        --------
        >>> x_obs = lambda X: np.cos(X[:, 0])
        >>> value = system.dig(u0, 1000, parameters=params, func=x_obs)

        >>> value = system.dig(
        ...     u0,
        ...     1000,
        ...     parameters=params,
        ...     func=lambda X: np.sin(X[:, 0] + X[:, 1]),
        ... )

        >>> value = system.dig(
        ...     u0,
        ...     2000,
        ...     parameters=params,
        ...     func=x_obs,
        ...     transient_time=500,
        ... )
        """
        u_arr: NDArray[np.float64] = validate_initial_conditions(
            u, self.__system_dimension, allow_ensemble=False
        )

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        parameters_arr: NDArray[np.float64] = validate_parameters(
            parameters, self.__number_of_parameters
        )

        validate_non_negative(total_time, "total_time", Integral)
        validate_transient_time(transient_time, total_time, Integral)

        if total_time % 2 != 0:
            total_time += 1

        if not callable(func):
            raise TypeError("func must be callable")

        test_input = np.array([u_arr], dtype=np.float64)
        test_output = func(test_input)

        if not isinstance(test_output, np.ndarray):
            raise ValueError("func must return a NumPy array")

        if test_output.ndim != 1:
            raise ValueError("func must return a 1D array")

        if test_output.shape[0] != test_input.shape[0]:
            raise ValueError(
                "func must return a 1D array with one value for each input state"
            )
        return dig(
            u=u_arr,
            parameters=parameters_arr,
            total_time=total_time,
            mapping=self.__mapping,
            func=func,
            transient_time=transient_time,
        )

    def lyapunov(
        self,
        u: numeric_like_t,
        total_time: int_t,
        parameters: Optional[numeric_like_t] = None,
        method: str = "QR",
        return_history: bool = False,
        sample_times: NDArray[np.integer] | Sequence[int_t] | None = None,
        transient_time: Optional[int_t] = None,
        num_exponents: Optional[int] = None,
        log_base: numeric_t = np.e,
        return_last_state: bool = False,
    ):
        """Compute Lyapunov exponents using specified numerical method.

        Parameters
        ----------
        u : Union[NDArray[np.float64], Sequence[float]]
            Initial condition(s) of shape (d,) or (n, d) where d is system dimension
        total_time : int
            Total iterations to compute (default 10000, must be ≥ 1)
        parameters : Union[None, float, Sequence[np.float64], NDArray[np.float64]], optional
            System parameters of shape (p,) passed to mapping function
        method : str, optional
            Computation method:
            - "QR": QR decomposition
            - "QR_HH": Householder QR (more stable)
        return_history : bool, optional
            If True, returns convergence history (default False)
        sample_times : Optional[Union[NDArray[np.float64], Sequence[int]]], optional
            Specific times to sample when return_history=True
        transient_time : Optional[int], optional
            Initial iterations to discard
        num_exponents : Optional[int], optional
            Number of Lyapunov exponents to compute, by default None. If None, compute the whole spectrum.
        log_base : float, optional (default np.e)
            Logarithm base for exponents (e.g. e, 2, or 10)

        Returns
        -------
        Union[Tuple[NDArray[np.float64], NDArray[np.float64]],
              Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]]

            - If return_history=False: exponents
            - If return_history=True: history

        Raises
        ------
        ValueError
            - If `u` is not a scalar, or 1D array, or if its shape does not match the expected system dimension.
            - If `parameters` is not None and does not match the expected number of parameters.
            - If `parameters` is None but the system expects parameters.
            - If `parameters` is a scalar or array-like but not 1D.
            - If `total_time` is negative.
            - If `trasient_time` is negative.
            - If `transient_time` is greater than or equal to total_time.
            - If `method` is not "QR" or "QR_HH".
            - If `sample_times` is not a 1D array of integers.
            - If `log_base` is not positive.
            - If `num_exponents` is larger then the system's dimension.
        TypeError
            - If `u` is not a scalar or array-like type.
            - If `parameters` is not a scalar or array-like type.
            - If `total_time` is not int.
            - If `transient_time` is not int.
            - If `log_base` is not float.
            - If `num_exponents` is not an positive integer.
            - If sample_times cannot be converted to a 1D array of integers.
            - If `method` is not a string.

        Notes
        -----
        - Sample times are automatically sorted and deduplicated

        References
        ----------
        [1] Eckmann & Ruelle, Rev. Mod. Phys 57, 617 (1985)
        [2] Wolf et al., Physica 16D 285-317 (1985)

        Examples
        --------
        >>> # Basic 2D system with ER method
        >>> u0 = np.array([0.1, 0.2])
        >>> params = np.array([0.5, 1.0])
        >>> lyapunov_exponents = system.lyapunov(u0, 10000,
        ...         parameters=params)

        >>> # With convergence history
        >>> lyapunov_exponents = system.lyapunov(u0, 10000,
        ...         parameters=params, return_history=True)
        >>> # Using Householder QR for better stability
        >>> lyapunov_exponents = system.lyapunov(u0, 10000,
        ...         parameters=params, method="QR_HH", return_history=True)
        >>> # With transient time and logarithm base 10
        >>> lyapunov_exponents = system.lyapunov(u0, 10000,
        ...         parameters=params, transient_time=1000,
        ...         log_base=10.0, return_history=True)
        """

        u = validate_initial_conditions(
            u, self.__system_dimension, allow_ensemble=False
        )

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        parameters = validate_parameters(parameters, self.__number_of_parameters)

        validate_non_negative(total_time, "total_time", Integral)
        validate_transient_time(transient_time, total_time, Integral)

        # Validate method
        if not isinstance(method, str):
            raise TypeError("method must be a string")
        method = method.upper()
        if method not in ("QR", "QR_HH"):
            raise ValueError("method must be 'QR' or 'QR_HH'")

        # Validate method for system dimension
        if method == "QR" and self.__system_dimension == 2:
            method = "ER"  # Fallback to QR for higher dimensions

        sample_times_arr: NDArray[np.int64] | None = None
        if return_history:
            if sample_times is not None:
                sample_times_arr = validate_sample_times(sample_times, total_time)
            else:
                sample_size = total_time - (
                    transient_time if transient_time is not None else 0
                )
                sample_times_arr = np.arange(1, sample_size + 1, dtype=np.int64)

        if num_exponents is None:
            num_exponents = self.__system_dimension
        elif num_exponents > self.__system_dimension:
            raise ValueError("num_exponents must be <= system_dimension")
        else:
            validate_non_negative(num_exponents, "num_exponents", Integral)

        validate_non_negative(log_base, "log_base", Real)
        if log_base == 1:
            raise ValueError("The logarithm function is not defined with base 1.")

        # Dispatch to appropriate computation
        if self.__system_dimension == 1:
            result, u = lyapunov_1D(
                u,
                parameters,
                total_time,
                self.__mapping,
                self.__jacobian,
                sample_times_arr,
                return_history=return_history,
                transient_time=transient_time,
                log_base=log_base,
            )
        else:
            if method == "ER":
                if num_exponents == 1:
                    result, u = maximum_lyapunov_er(
                        u,
                        parameters,
                        total_time,
                        self.__mapping,
                        self.__jacobian,
                        sample_times_arr,
                        return_history,
                        transient_time,
                        log_base,
                    )
                else:
                    result, u = lyapunov_er(
                        u,
                        parameters,
                        total_time,
                        self.__mapping,
                        self.__jacobian,
                        sample_times_arr,
                        return_history,
                        transient_time,
                        log_base,
                    )
            else:
                qr_func = qr if method == "QR" else householder_qr
                result, u = lyapunov_qr(
                    u,
                    parameters,
                    total_time,
                    self.__mapping,
                    self.__jacobian,
                    num_exponents,
                    sample_times_arr,
                    qr_func,
                    return_history,
                    transient_time,
                    log_base,
                )

        if return_history:
            if num_exponents == 1:
                result = result.ravel()
            if return_last_state:
                return result, u
            return result

        # non-history case
        result = np.asarray(result)

        if num_exponents == 1:
            value = float(result.ravel()[0])
            if return_last_state:
                return value, u
            return value

        value = result.ravel()
        if return_last_state:
            return value, u
        return value

    def finite_time_lyapunov(
        self,
        u: Union[NDArray[np.float64], Sequence[float], float],
        total_time: int,
        finite_time: int,
        parameters: Union[
            None, float, Sequence[np.float64], NDArray[np.float64]
        ] = None,
        num_exponents: Optional[int] = None,
        method: str = "QR",
        transient_time: Optional[int] = None,
        log_base: float = np.e,
        return_points: bool = False,
    ) -> Union[NDArray[np.float64], Tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """Compute finite-time Lyapunov exponents (FTLE) along trajectory.

        Parameters
        ----------
        u : Union[NDArray[np.float64], Sequence[float]]
            Initial condition of shape (d,) where d is system dimension
        total_time : int
            Total simulation time steps (must be > finite_time, default 10000)
        finite_time : int
            Averaging window size in time steps (default 100)
        parameters : Union[None, float, Sequence[np.float64], NDArray[np.float64]], optional
            System parameters of shape (p,) passed to mapping function
        method : str, optional
            Computation method:
            - "ER": Eckmann-Ruelle (optimal for 2D systems)
            - "QR": Gram-Schmidt QR decomposition
            - "QR_HH": Householder QR (more stable)
        transient_time : Optional[int], optional
            Initial burn-in period to discard (default None → finite_time)

        Returns
        -------
        NDArray[np.float64]
            FTLE matrix of shape (n_windows, d) where:

            - n_windows = (total_time - transient_time) // finite_time
            - Each row contains exponents for one time window
            - Columns are ordered by decreasing exponent magnitude

        Raises
        ------
        ValueError
            - If `u` is not a scalar, or 1D array, or if its shape does not match the expected system dimension.
            - If `parameters` is not None and does not match the expected number of parameters.
            - If `parameters` is None but the system expects parameters.
            - If `parameters` is a scalar or array-like but not 1D.
            - If `total_time` is negative.
            - If `finite_time` is negative or zero.
            - If `trasient_time` is negative.
            - If `transient_time` is greater than or equal to total_time.
            - If `method` is not "QR" or "QR_HH".
            - If `log_base` is not positive
        TypeError
            - If `u` is not a scalar or array-like type.
            - If `parameters` is not a scalar or array-like type.
            - If `total_time` is not int.
            - If `transient_time` is not int.
            - If `log_base` is not float.
            - If `method` is not a string.
            - If `return_points` is not a boolean.

        Notes
        -----
        - FTLE measure local stretching rates over finite intervals
        - For chaotic systems, FTLE → true exponents as finite_time → ∞
        - ER method is faster but limited to 2D systems
        - Results are more reliable when:
        - finite_time >> 1
        - (total_time - transient_time) // finite_time >> 1

        Examples
        --------
        >>> # Basic usage with defaults
        >>> u0 = np.array([0.1, 0.2])
        >>> params = np.array([0.5, 1.0])
        >>> ftle = system.finite_time_lyapunov_exponents(u0, params)

        >>> # With custom parameters
        >>> ftle = system.finite_time_lyapunov_exponents(
        ...     u0, params,
        ...     total_time=5000,
        ...     finite_time=50,
        ...     method="GS"
        ... )
        """

        u = validate_initial_conditions(
            u, self.__system_dimension, allow_ensemble=False
        )

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        else:
            parameters = validate_parameters(parameters, self.__number_of_parameters)

        validate_non_negative(total_time, "total_time", Integral)
        validate_positive(finite_time, "finite_time", Integral)
        validate_finite_time(finite_time, total_time)
        validate_transient_time(transient_time, total_time, Integral)

        # Validate method
        if not isinstance(method, str):
            raise TypeError("method must be a string")
        method = method.upper()
        if method not in ("QR", "QR_HH"):
            raise ValueError("method must be 'QR' or 'QR_HH'")

        if num_exponents is None:
            num_exponents = self.__system_dimension
        elif num_exponents > self.__system_dimension:
            raise ValueError("num_exponents must be <= system_dimension")

        # Validate method for system dimension
        if method == "QR" and self.__system_dimension == 2:
            method = "ER"  # Fallback to QR for higher dimensions

        validate_non_negative(log_base, "log_base", Real)
        if log_base == 1:
            raise ValueError("The logarithm function is not defined with base 1.")

        if not isinstance(return_points, bool):
            raise TypeError("return_points must be a boolean")

        return finite_time_lyapunov(
            u,
            parameters,
            total_time,
            finite_time,
            self.__mapping,
            self.__jacobian,
            num_exponents,
            method=method,
            transient_time=transient_time,
            log_base=log_base,
            return_points=return_points,
        )

    def CLV(
        self,
        u: numeric_like_t,
        total_time: int_t,
        parameters: numeric_like_t | None = None,
        num_clvs: int | None = None,
        transient_time: int_t = 0,
        warmup_time: int_t = 0,
        tail_time: int_t = 0,
        seed: int = 1312,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Compute covariant Lyapunov vectors (CLVs) along a trajectory of this
        discrete-time system.

        The CLVs form a covariant basis of tangent space that transforms under the
        Jacobian consistently with the dynamics. The `i`-th CLV is associated with
        the `i`-th Lyapunov exponent, ordered from the most expanding to the most
        contracting direction.

        This method uses a Ginelli-style algorithm consisting of:
        1. transient evolution
        2. forward QR warm-up
        3. forward storage of orthonormal bases and triangular factors
        4. backward initialization
        5. backward recursion yielding the CLVs

        Parameters
        ----------
        u : numeric_like_t
            Initial condition of shape `(system_dimension,)`.
        total_time : int_t
            Number of map iterations for which CLVs are returned.
        parameters : numeric_like_t | None, optional
            System parameters passed to the mapping function. If None, the stored
            system parameters are used.
        num_clvs : int | None, optional
            Number of CLVs to compute. If None, all CLVs are computed.
        transient_time : int_t, optional
            Number of initial iterations discarded before the CLV computation.
        warmup_time : int_t, optional
            Number of forward QR warm-up iterations used before storing the
            orthonormal bases.
        tail_time : int_t, optional
            Number of additional forward QR iterations used to initialize the
            backward recursion.
        seed : int, optional
            Seed used to initialize the random upper-triangular matrix in the
            backward stage.

        Returns
        -------
        Tuple[NDArray[np.float64], NDArray[np.float64]]
            - `clvs`: array of shape `(total_time + 1, system_dimension, num_clvs)`
              containing the CLVs along the trajectory
            - `traj`: array of shape `(total_time + 1, system_dimension)`
              containing the corresponding trajectory

        Raises
        ------
        ValueError
            - If the system dimension is less than 2.
            - If `total_time` is negative.
            - If `transient_time` is invalid.
            - If `warmup_time` is negative.
            - If `tail_time` is negative.
            - If `num_clvs` is not in the interval `[1, system_dimension]`.
        TypeError
            - If `u` is not a scalar or array-like numeric object.
            - If `parameters` is not a scalar or array-like numeric object.
            - If `total_time`, `transient_time`, `warmup_time`, or `tail_time`
              are not integers.
            - If `seed` is not an integer.

        Notes
        -----
        The quality of the computed CLVs depends on the choices of `warmup_time`
        and `tail_time`. In weakly hyperbolic or nearly tangent regimes, larger
        values may be necessary.

        References
        ----------
        F. Ginelli et al., Characterizing dynamics with covariant Lyapunov
        vectors, Phys. Rev. Lett. 99, 130601 (2007).
        """
        if self.__system_dimension < 2:
            raise ValueError(
                f"System dimension must be >= 2 to compute CLVs, got {self.__system_dimension}."
            )

        u_arr: NDArray[np.float64] = validate_initial_conditions(
            u, self.__system_dimension, allow_ensemble=False
        )

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        parameters_arr: NDArray[np.float64] = validate_parameters(
            parameters, self.__number_of_parameters
        )

        validate_non_negative(total_time, "total_time", Integral)
        validate_transient_time(transient_time, total_time, Integral)
        validate_non_negative(warmup_time, "warmup_time", Integral)
        validate_non_negative(tail_time, "tail_time", Integral)

        if num_clvs is None:
            num_clvs = self.__system_dimension
        else:
            validate_positive(num_clvs, "num_clvs", Integral)
            if num_clvs > self.__system_dimension:
                raise ValueError("num_clvs must be <= system_dimension")

        if not isinstance(seed, Integral):
            raise TypeError("seed must be an integer")

        return compute_clvs(
            u=u_arr,
            parameters=parameters_arr,
            total_time=total_time,
            mapping=self.__mapping,
            jacobian=self.__jacobian,
            num_clvs=num_clvs,
            transient_time=transient_time,
            warmup_time=warmup_time,
            tail_time=tail_time,
            seed=seed,
        )

    def CLV_angles(
        self,
        u: numeric_like_t,
        total_time: int_t,
        parameters: numeric_like_t | None = None,
        subspaces: Sequence[Tuple[Sequence[int], Sequence[int]]] | None = None,
        pairs: Sequence[Tuple[int, int]] | None = None,
        window_time: int_t | None = None,
        transient_time: int_t = 0,
        warmup_time: int_t = 0,
        tail_time: int_t = 0,
        seed: int = 1312,
        use_abs: bool = True,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Compute angle diagnostics derived from covariant Lyapunov vectors (CLVs).

        This method computes CLVs along a trajectory and returns either the full
        time series of requested angles or window-averaged angle diagnostics.

        Parameters
        ----------
        u : numeric_like_t
            Initial condition of shape `(system_dimension,)`.
        total_time : int_t
            Number of map iterations used for the angle diagnostics.
        parameters : numeric_like_t | None, optional
            System parameters passed to the mapping function. If None, the stored
            system parameters are used.
        subspaces : Sequence[Tuple[Sequence[int], Sequence[int]]] | None, optional
            Pairs of CLV index sets defining subspaces whose minimum principal
            angles are to be computed.
        pairs : Sequence[Tuple[int, int]] | None, optional
            Pairs of CLV indices whose mutual angles are to be computed.
        window_time : int_t | None, optional
            If None, the full time series of angles is returned. Otherwise, angles
            are computed in consecutive windows and averaged over each window.
        transient_time : int_t, optional
            Number of initial iterations discarded before the computation.
        warmup_time : int_t, optional
            Forward QR warm-up length passed to the CLV computation.
        tail_time : int_t, optional
            Backward-recursion convergence length passed to the CLV computation.
        seed : int, optional
            Seed forwarded to the CLV computation.
        use_abs : bool, optional
            If True, use the absolute value of the cosine before applying
            `arccos`.

        Returns
        -------
        Tuple[NDArray[np.float64], NDArray[np.float64]]
            If `window_time is None`:
                - `angles`: array of shape `(T, M)`
                - `traj`: trajectory of shape `(T, system_dimension)`

            If `window_time is not None`:
                - `avg_angles`: array of shape `(num_windows, M + 1)`, where the
                  first column contains the window center time index
                - `initial_conditions`: array of shape `(num_windows, system_dimension)`

        Raises
        ------
        ValueError
            - If the system dimension is less than 2.
            - If `total_time` is negative.
            - If `transient_time` is invalid.
            - If `warmup_time` is negative.
            - If `tail_time` is negative.
            - If `window_time` is not None and is not positive.
            - If both `subspaces` and `pairs` are missing.
            - If `subspaces` or `pairs` contain invalid CLV indices.
        TypeError
            - If `u` is not a scalar or array-like numeric object.
            - If `parameters` is not a scalar or array-like numeric object.
            - If `total_time`, `transient_time`, `warmup_time`, `tail_time`, or
              `window_time` are not integers.
            - If `seed` is not an integer.
            - If `use_abs` is not a boolean.

        Notes
        -----
        Small subspace angles indicate near-tangencies between invariant
        directions and are often the most informative hyperbolicity diagnostic.
        Pairwise CLV angles are more fine-grained but less geometrically complete
        than subspace angles.
        """
        if self.__system_dimension < 2:
            raise ValueError(
                f"System dimension must be >= 2 to compute CLV angles, got {self.__system_dimension}."
            )

        if (not subspaces) and (not pairs):
            raise ValueError("At least one of `subspaces` or `pairs` must be provided.")

        u_arr: NDArray[np.float64] = validate_initial_conditions(
            u, self.__system_dimension, allow_ensemble=False
        )

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        parameters_arr: NDArray[np.float64] = validate_parameters(
            parameters, self.__number_of_parameters
        )

        validate_non_negative(total_time, "total_time", Integral)
        validate_transient_time(transient_time, total_time, Integral)
        validate_non_negative(warmup_time, "warmup_time", Integral)
        validate_non_negative(tail_time, "tail_time", Integral)

        if window_time is not None:
            validate_positive(window_time, "window_time", Integral)

        if not isinstance(seed, Integral):
            raise TypeError("seed must be an integer")

        if not isinstance(use_abs, bool):
            raise TypeError("use_abs must be a boolean")

        num_clvs = self.__system_dimension
        subspaces_valid = validate_clv_subspaces(subspaces, num_clvs)
        pairs_valid = validate_clv_pairs(pairs, num_clvs)

        return clv_angles(
            u=u_arr,
            parameters=parameters_arr,
            total_time=total_time,
            mapping=self.__mapping,
            jacobian=self.__jacobian,
            subspaces=subspaces_valid,
            pairs=pairs_valid,
            window_time=window_time,
            transient_time=transient_time,
            warmup_time=warmup_time,
            tail_time=tail_time,
            seed=seed,
            use_abs=use_abs,
        )

    def hurst_exponent(
        self,
        u: numeric_like_t,
        total_time: int_t,
        parameters: numeric_like_t | None = None,
        wmin: int = 2,
        transient_time: int_t | None = None,
    ) -> float | NDArray[np.float64]:
        """
        Estimate the Hurst exponent of each component of a trajectory generated by
        the discrete-time system.

        Parameters
        ----------
        u : numeric_like_t
            Initial condition of shape `(system_dimension,)`.
        total_time : int_t
            Total number of iterations used to generate the trajectory.
        parameters : numeric_like_t | None, optional
            System parameters passed to the mapping function. If None, stored
            parameters are used.
        wmin : int, optional
            Minimum window size used in the rescaled-range analysis. Must satisfy
            `2 <= wmin < effective_time // 2`, where
            `effective_time = total_time - transient_time`.
        transient_time : int_t | None, optional
            Number of initial iterations discarded before generating the trajectory.

        Returns
        -------
        float | NDArray[np.float64]
            - If `system_dimension == 1`, returns a scalar Hurst exponent.
            - Otherwise, returns an array of shape `(system_dimension,)` containing
              one Hurst exponent per coordinate.

        Raises
        ------
        ValueError
            - If `u` is incompatible with the system dimension.
            - If `parameters` does not match the expected number of parameters.
            - If `total_time` is negative.
            - If `transient_time` is invalid.
            - If `wmin < 2`.
            - If `wmin` is too large for the effective trajectory length.
        TypeError
            - If `u` is not a scalar or array-like numeric object.
            - If `parameters` is not a scalar or array-like numeric object.
            - If `total_time` is not an integer.
            - If `transient_time` is not an integer.
            - If `wmin` is not an integer.

        Notes
        -----
        The Hurst exponent is estimated independently for each coordinate of the
        generated trajectory using the rescaled-range method.
        """
        u_arr: NDArray[np.float64] = validate_initial_conditions(
            u, self.__system_dimension, allow_ensemble=False
        )

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        parameters_arr: NDArray[np.float64] = validate_parameters(
            parameters, self.__number_of_parameters
        )

        validate_non_negative(total_time, "total_time", Integral)
        validate_transient_time(transient_time, total_time, Integral)
        validate_positive(wmin, "wmin", Integral)

        effective_time = total_time - (
            transient_time if transient_time is not None else 0
        )

        if wmin < 2:
            raise ValueError("wmin must be >= 2")

        if wmin >= effective_time // 2:
            raise ValueError(
                "wmin must be smaller than half of the effective trajectory length"
            )

        result = hurst_exponent_wrapped(
            u=u_arr,
            parameters=parameters_arr,
            total_time=total_time,
            mapping=self.__mapping,
            wmin=wmin,
            transient_time=transient_time,
        )

        if self.__system_dimension == 1:
            return float(result[0])

        return result

    def finite_time_hurst_exponent(
        self,
        u: numeric_like_t,
        total_time: int_t,
        finite_time: int_t,
        parameters: numeric_like_t | None = None,
        wmin: int = 2,
        return_points: bool = False,
    ) -> NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Compute finite-time Hurst exponents along a trajectory.

        Parameters
        ----------
        u : numeric_like_t
            Initial condition of shape `(system_dimension,)`.
        total_time : int_t
            Total number of iterations used in the computation.
        finite_time : int_t
            Length of each non-overlapping analysis window.
        parameters : numeric_like_t | None, optional
            System parameters passed to the mapping function. If None, stored
            parameters are used.
        wmin : int, optional
            Minimum window size used in the rescaled-range analysis within each
            finite-time window.
        return_points : bool, optional
            If True, also return the final phase-space point of each window.

        Returns
        -------
        NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]]
            - If `return_points=False`, returns an array of shape
              `(num_windows, system_dimension)` containing the finite-time Hurst
              exponents.
            - If `return_points=True`, returns:
                - `h_values`: shape `(num_windows, system_dimension)`
                - `phase_space_points`: shape `(num_windows, system_dimension)`

        Raises
        ------
        ValueError
            - If `u` is incompatible with the system dimension.
            - If `parameters` does not match the expected number of parameters.
            - If `total_time` is negative.
            - If `finite_time` is not positive.
            - If `finite_time > total_time`.
            - If `wmin < 2`.
            - If `wmin >= finite_time // 2`.
        TypeError
            - If `u` is not a scalar or array-like numeric object.
            - If `parameters` is not a scalar or array-like numeric object.
            - If `total_time` is not an integer.
            - If `finite_time` is not an integer.
            - If `wmin` is not an integer.
            - If `return_points` is not a boolean.

        Notes
        -----
        The trajectory is divided into consecutive non-overlapping windows of length
        `finite_time`, and one Hurst exponent per coordinate is computed in each
        window.
        """
        u_arr: NDArray[np.float64] = validate_initial_conditions(
            u, self.__system_dimension, allow_ensemble=False
        )

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        parameters_arr: NDArray[np.float64] = validate_parameters(
            parameters, self.__number_of_parameters
        )

        validate_non_negative(total_time, "total_time", Integral)
        validate_positive(finite_time, "finite_time", Integral)
        validate_finite_time(finite_time, total_time)
        validate_positive(wmin, "wmin", Integral)

        if wmin < 2:
            raise ValueError("wmin must be >= 2")

        if wmin >= finite_time // 2:
            raise ValueError("wmin must be smaller than half of finite_time")

        if not isinstance(return_points, bool):
            raise TypeError("return_points must be a boolean")

        return finite_time_hurst_exponent(
            u=u_arr,
            parameters=parameters_arr,
            total_time=total_time,
            finite_time=finite_time,
            mapping=self.__mapping,
            wmin=wmin,
            return_points=return_points,
        )

    def SALI(
        self,
        u: numeric_like_t,
        total_time: int_t,
        parameters: numeric_like_t | None = None,
        return_history: bool = False,
        sample_times: Sequence[int_t] | NDArray[np.integer] | None = None,
        tol: numeric_t = 1e-16,
        transient_time: int_t | None = None,
        seed: int_t = 1312,
        return_last_state: bool = False,
    ) -> (
        float
        | NDArray[np.float64]
        | Tuple[float, NDArray[np.float64]]
        | Tuple[NDArray[np.float64], NDArray[np.float64]]
    ):
        """
        Compute the Smallest Alignment Index (SALI) for a discrete dynamical system.

        SALI measures the alignment of two deviation vectors evolved in tangent
        space. For chaotic trajectories, SALI typically decays rapidly toward
        zero, whereas for regular trajectories it remains bounded away from zero.

        Parameters
        ----------
        u : numeric_like_t
            Initial condition of shape `(d,)`, where `d` is the system dimension.
        total_time : int_t
            Total number of iterations used in the computation.
        parameters : numeric_like_t | None, optional
            System parameters passed to the mapping function. If None, the
            parameters stored in the system are used.
        return_history : bool, optional
            If True, return SALI evaluated at the requested sampling times.
            Otherwise, return only the final SALI value. Default is False.
        sample_times : Sequence[int_t] | NDArray[np.integer] | None, optional
            Iteration times at which SALI is recorded when `return_history=True`.
            If None and `return_history=True`, SALI is recorded at every iteration
            after the transient.
        tol : numeric_t, optional
            Early stopping threshold. If `SALI < tol`, the computation is
            interrupted. Default is `1e-16`.
        transient_time : int_t | None, optional
            Number of initial iterations to discard before starting the SALI
            computation. If None, no transient is discarded.
        seed : int_t, optional
            Seed used to initialize the deviation vectors. Default is 1312.
        return_last_state : bool, optional
            If True, also return the final state of the trajectory.

        Returns
        -------
        float or NDArray[np.float64] or tuple
            - If `return_history=False` and `return_last_state=False`, returns the
              final SALI value as a scalar.
            - If `return_history=False` and `return_last_state=True`, returns
              `(final_sali, final_state)`.
            - If `return_history=True` and `return_last_state=False`, returns a
              1D array containing the sampled SALI history.
            - If `return_history=True` and `return_last_state=True`, returns
              `(sali_history, final_state)`.

        Raises
        ------
        ValueError
            - If `u` is not compatible with the system dimension.
            - If `parameters` does not match the expected number of parameters.
            - If `total_time` is negative.
            - If `transient_time` is invalid.
            - If `sample_times` is not a valid 1D array of integers.
            - If `tol` is negative.
        TypeError
            - If `u` is not a scalar or array-like numeric object.
            - If `parameters` is not a scalar or array-like numeric object.
            - If `total_time` is not an integer.
            - If `transient_time` is not an integer.
            - If `tol` is not numeric.
            - If `seed` is not an integer.

        Notes
        -----
        SALI tends to zero for chaotic trajectories and remains bounded away from
        zero for regular trajectories. Sample times are automatically sorted and
        deduplicated.

        Examples
        --------
        >>> u0 = np.array([0.1, 0.2])
        >>> params = np.array([0.5, 1.0])

        >>> sali = system.SALI(u0, 10000, parameters=params)

        >>> sali_hist = system.SALI(
        ...     u0, 10000, parameters=params, return_history=True
        ... )

        >>> times = np.array([100, 1000, 5000])
        >>> sali_samples, final_state = system.SALI(
        ...     u0,
        ...     10000,
        ...     parameters=params,
        ...     sample_times=times,
        ...     return_history=True,
        ...     return_last_state=True,
        ... )
        """

        u = validate_initial_conditions(
            u, self.__system_dimension, allow_ensemble=False
        )

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        parameters = validate_parameters(parameters, self.__number_of_parameters)

        validate_non_negative(total_time, "total_time", Integral)
        validate_transient_time(transient_time, total_time, Integral)

        sample_times_arr: NDArray[np.int64] | None = None
        if return_history:
            if sample_times is not None:
                sample_times_arr = validate_sample_times(sample_times, total_time)
            else:
                sample_size = total_time - (
                    transient_time if transient_time is not None else 0
                )
                sample_times_arr = np.arange(1, sample_size + 1, dtype=np.int64)

        validate_non_negative(tol, "tol", Real)

        if not isinstance(seed, Integral):
            raise TypeError("seed must be an integer")

        result, u = sali(
            u,
            parameters,
            total_time,
            self.__mapping,
            self.__jacobian,
            sample_times_arr,
            return_history=return_history,
            transient_time=transient_time,
            tol=tol,
            seed=seed,
        )

        if return_history:
            if return_last_state:
                return result, u
            return result

        value = float(np.asarray(result).ravel()[0])

        if return_last_state:
            return value, u
        return value

    def LDI(
        self,
        u: numeric_like_t,
        total_time: int_t,
        k: int,
        parameters: numeric_like_t | None = None,
        return_history: bool = False,
        sample_times: Sequence[int_t] | NDArray[np.integer] | None = None,
        tol: numeric_t = 1e-16,
        transient_time: int_t | None = None,
        seed: int = 1312,
        return_last_state: bool = False,
    ) -> (
        float
        | NDArray[np.float64]
        | Tuple[float, NDArray[np.float64]]
        | Tuple[NDArray[np.float64], NDArray[np.float64]]
    ):
        """
        Compute the Linear Dependence Index (LDI_k) for a discrete dynamical system.

        LDI_k is computed from the evolution of `k` deviation vectors in tangent
        space and is used to distinguish regular and chaotic motion.

        Parameters
        ----------
        u : numeric_like_t
            Initial condition of shape `(d,)`, where `d` is the system dimension.
        total_time : int_t
            Total number of iterations used in the computation.
        k : int
            Number of deviation vectors used in the computation.
        parameters : numeric_like_t | None, optional
            System parameters passed to the mapping function. If None, the
            parameters stored in the system are used.
        return_history : bool, optional
            If True, return LDI_k evaluated at the requested sampling times.
            Otherwise, return only the final LDI_k value. Default is False.
        sample_times : Sequence[int_t] | NDArray[np.integer] | None, optional
            Iteration times at which LDI_k is recorded when `return_history=True`.
            If None and `return_history=True`, LDI_k is recorded at every iteration
            after the transient.
        tol : numeric_t, optional
            Early stopping threshold. If `LDI_k < tol`, the computation is
            interrupted. Default is `1e-16`.
        transient_time : int_t | None, optional
            Number of initial iterations to discard before starting the LDI_k
            computation. If None, no transient is discarded.
        seed : int, optional
            Seed used to initialize the deviation vectors. Default is 1312.
        return_last_state : bool, optional
            If True, also return the final state of the trajectory.

        Returns
        -------
        float or NDArray[np.float64] or tuple
            - If `return_history=False` and `return_last_state=False`, returns the
              final LDI_k value as a scalar.
            - If `return_history=False` and `return_last_state=True`, returns
              `(final_ldi, final_state)`.
            - If `return_history=True` and `return_last_state=False`, returns a
              1D array containing the sampled LDI_k history.
            - If `return_history=True` and `return_last_state=True`, returns
              `(ldi_history, final_state)`.

        Raises
        ------
        ValueError
            - If `u` is not compatible with the system dimension.
            - If `parameters` does not match the expected number of parameters.
            - If `total_time` is negative.
            - If `transient_time` is invalid.
            - If `sample_times` is not a valid 1D array of integers.
            - If `k` is not in the interval `[2, system_dimension]`.
            - If `tol` is negative.
        TypeError
            - If `u` is not a scalar or array-like numeric object.
            - If `parameters` is not a scalar or array-like numeric object.
            - If `total_time` is not an integer.
            - If `transient_time` is not an integer.
            - If `tol` is not numeric.
            - If `seed` is not an integer.
            - If `k` is not an integer.

        Notes
        -----
        - A set of `k` initially orthonormal deviation vectors is evolved with the
        Jacobian and renormalized at each step. LDI_k is computed as the product
        of the singular values of the deviation-vector matrix.

        - The computation is terminated early if `LDI_k < tol`.

        Examples
        --------
        >>> u0 = np.array([0.1, 0.2, 0.0, 0.0])
        >>> params = np.array([0.5, 1.0])

        >>> ldi = system.LDI(u0, 10000, k=2, parameters=params)

        >>> ldi_hist = system.LDI(
        ...     u0, 10000, k=3, parameters=params, return_history=True
        ... )

        >>> times = np.array([100, 1000, 5000])
        >>> ldi_samples, final_state = system.LDI(
        ...     u0,
        ...     10000,
        ...     k=2,
        ...     parameters=params,
        ...     sample_times=times,
        ...     return_history=True,
        ...     return_last_state=True,
        ... )
        """
        u = validate_initial_conditions(
            u, self.__system_dimension, allow_ensemble=False
        )

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        parameters = validate_parameters(parameters, self.__number_of_parameters)

        validate_non_negative(total_time, "total_time", Integral)
        validate_transient_time(transient_time, total_time, Integral)

        validate_positive(k, "k", Integral)
        if k < 2 or k > self.__system_dimension:
            raise ValueError(f"k must be in range [2, {self.__system_dimension}]")

        sample_times_arr: NDArray[np.int64] | None = None
        if return_history:
            if sample_times is not None:
                sample_times_arr = validate_sample_times(sample_times, total_time)
            else:
                sample_size = total_time - (
                    transient_time if transient_time is not None else 0
                )
                sample_times_arr = np.arange(1, sample_size + 1, dtype=np.int64)

        validate_non_negative(tol, "tol", Real)

        if not isinstance(seed, Integral):
            raise TypeError("seed must be an integer")

        # Call underlying implementation
        result, u = ldi_k(
            u,
            parameters,
            total_time,
            self.__mapping,
            self.__jacobian,
            k,
            sample_times_arr,
            return_history=return_history,
            transient_time=transient_time,
            tol=tol,
            seed=seed,
        )

        if return_history:
            if return_last_state:
                return result, u
            return result

        value = float(np.asarray(result).ravel()[0])

        if return_last_state:
            return value, u
        return value

    def GALI(
        self,
        u: numeric_like_t,
        total_time: int_t,
        k: int,
        parameters: numeric_like_t | None = None,
        return_history: bool = False,
        sample_times: Sequence[int_t] | NDArray[np.integer] | None = None,
        method: str = "QR",
        tol: numeric_t = 1e-16,
        transient_time: int_t | None = None,
        seed: int = 1312,
        return_last_state: bool = False,
    ) -> (
        float
        | NDArray[np.float64]
        | Tuple[float, NDArray[np.float64]]
        | Tuple[NDArray[np.float64], NDArray[np.float64]]
    ):
        """
        Compute the Generalized Alignment Index (GALI_k) for a discrete dynamical system.

        GALI_k quantifies the degree of alignment of `k` deviation vectors evolved in
        tangent space. It measures the contraction of the `k`-dimensional volume
        spanned by these vectors and is used to distinguish regular and chaotic
        motion.

        Parameters
        ----------
        u : numeric_like_t
            Initial condition of shape `(d,)`, where `d` is the system dimension.
        total_time : int_t
            Total number of iterations used in the computation.
        k : int
            Number of deviation vectors used in the computation.
        parameters : numeric_like_t | None, optional
            System parameters passed to the mapping function. If None, the
            parameters stored in the system are used.
        return_history : bool, optional
            If True, return GALI_k evaluated at the requested sampling times.
            Otherwise, return only the final GALI_k value. Default is False.
        sample_times : Sequence[int_t] | NDArray[np.integer] | None, optional
            Iteration times at which GALI_k is recorded when `return_history=True`.
            If None and `return_history=True`, GALI_k is recorded at every iteration
            after the transient.
        method : str, optional
            Method used to compute GALI_k. Supported options are:

            - `"DET"`:
              Compute GALI_k from the Gram matrix `G = V^T V`, where `V` is the
              deviation-vector matrix, through `GALI_k = sqrt(det(G))`.

            - `"QR"`:
              Compute GALI_k from the diagonal of the triangular factor obtained
              from the custom QR decomposition routine `qr`, using
              `GALI_k = prod_i |R_ii|`.

            - `"QR_HH"`:
              Compute GALI_k from the diagonal of the triangular factor obtained
              from the Householder QR decomposition `numpy.linalg.qr`, again using
              `GALI_k = prod_i |R_ii|`.

            Default is `"QR"`.
        tol : numeric_t, optional
            Early stopping threshold. If `GALI_k < tol`, the computation is
            interrupted. Default is `1e-16`.
        transient_time : int_t | None, optional
            Number of initial iterations to discard before starting the GALI_k
            computation. If None, no transient is discarded.
        seed : int, optional
            Seed used to initialize the deviation vectors. Default is 1312.
        return_last_state : bool, optional
            If True, also return the final state of the trajectory.

        Returns
        -------
        float or NDArray[np.float64] or tuple
            - If `return_history=False` and `return_last_state=False`, returns the
              final GALI_k value as a scalar.
            - If `return_history=False` and `return_last_state=True`, returns
              `(final_gali, final_state)`.
            - If `return_history=True` and `return_last_state=False`, returns a
              1D array containing the sampled GALI_k history.
            - If `return_history=True` and `return_last_state=True`, returns
              `(gali_history, final_state)`.

        Raises
        ------
        ValueError
            - If `u` is not compatible with the system dimension.
            - If `parameters` does not match the expected number of parameters.
            - If `total_time` is negative.
            - If `transient_time` is invalid.
            - If `sample_times` is not a valid 1D array of integers.
            - If `k` is not in the interval `[2, system_dimension]`.
            - If `method` is not `"DET"`, `"QR"`, or `"QR_HH"`.
            - If `tol` is negative.
        TypeError
            - If `u` is not a scalar or array-like numeric object.
            - If `parameters` is not a scalar or array-like numeric object.
            - If `total_time` is not an integer.
            - If `transient_time` is not an integer.
            - If `tol` is not numeric.
            - If `seed` is not an integer.
            - If `k` is not an integer.
            - If `method` is not a string.

        Notes
        -----
        GALI_k can be written equivalently as

            `GALI_k = sqrt(det(V^T V))`

        where `V` is the deviation-vector matrix, or as

            `GALI_k = |det(R)| = prod_i |R_ii|`

        when `V = Q R` is a QR decomposition.

        For chaotic trajectories, GALI_k typically decays rapidly toward zero due
        to the progressive alignment of deviation vectors. For regular trajectories,
        the decay is slower or GALI_k may remain bounded away from zero, depending
        on `k` and on the dimension of the invariant set.

        The computation is terminated early if `GALI_k < tol`.

        Examples
        --------
        >>> u0 = np.array([0.1, 0.2, 0.0, 0.0])
        >>> params = np.array([0.5, 1.0])

        >>> gali = system.GALI(u0, 10000, k=2, parameters=params)

        >>> gali_hist = system.GALI(
        ...     u0, 10000, k=3, parameters=params, return_history=True
        ... )

        >>> times = np.array([100, 1000, 5000])
        >>> gali_samples, final_state = system.GALI(
        ...     u0,
        ...     10000,
        ...     k=2,
        ...     parameters=params,
        ...     sample_times=times,
        ...     return_history=True,
        ...     return_last_state=True,
        ... )
        """
        u = validate_initial_conditions(
            u, self.__system_dimension, allow_ensemble=False
        )

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        parameters = validate_parameters(parameters, self.__number_of_parameters)

        validate_non_negative(total_time, "total_time", Integral)
        validate_transient_time(transient_time, total_time, Integral)

        if not isinstance(method, str):
            raise TypeError("method must be a string")
        method = method.upper()
        if method not in ("DET", "QR", "QR_HH"):
            raise ValueError("method must be 'DET', 'QR', or 'QR_HH'")

        validate_positive(k, "k", Integral)
        if k < 2 or k > self.__system_dimension:
            raise ValueError(f"k must be in range [2, {self.__system_dimension}]")

        sample_times_arr: NDArray[np.int64] | None = None
        if return_history:
            if sample_times is not None:
                sample_times_arr = validate_sample_times(sample_times, total_time)
            else:
                sample_size = total_time - (
                    transient_time if transient_time is not None else 0
                )
                sample_times_arr = np.arange(1, sample_size + 1, dtype=np.int64)

        validate_non_negative(tol, "tol", Real)

        if not isinstance(seed, Integral):
            raise TypeError("seed must be an integer")

        # Call underlying implementation
        result, u = gali_k(
            u,
            parameters,
            total_time,
            self.__mapping,
            self.__jacobian,
            k,
            sample_times_arr,
            method=method,
            return_history=return_history,
            transient_time=transient_time,
            tol=tol,
            seed=seed,
        )

        if return_history:
            if return_last_state:
                return result, u
            return result

        value = float(np.asarray(result).ravel()[0])

        if return_last_state:
            return value, u
        return value

    def recurrence_matrix(
        self,
        u: Union[NDArray[np.float64], Sequence[float]],
        total_time: int,
        parameters: Union[
            None, float, Sequence[np.float64], NDArray[np.float64]
        ] = None,
        transient_time: Optional[int] = None,
        **kwargs: Any,
    ) -> NDArray[np.uint8]:
        """
        Compute the recurrence matrix of a univariate or multivariate time series.

        Parameters
        ----------
        u: NDArray
            Time series data. Can be 1D(shape: (N,)) or 2D(shape: (N, d)).
            If 1D, the array is reshaped to (N, 1) automatically.
        total_time: int
            Total number of iterations to simulate.
        parameters: Union[None, float, Sequence[np.float64], NDArray[np.float64]], optional
            Parameters passed to the mapping function.
        transient_time: Optional[int], optional
            Number of initial iterations to discard as transient(default None).
            If None, no transient is removed.
        metric: {"supremum", "euclidean", "manhattan"}, default = "supremum"
            Distance metric used for phase space reconstruction.
        std_metric: {"supremum", "euclidean", "manhattan"}, default = "supremum"
            Distance metric used for standard deviation calculation.
        threshold: float, default = 0.1
            Recurrence threshold(relative to data range).
        threshold_std: bool, default = True
            Whether to scale threshold by data standard deviation.

        Returns
        -------
        recmat: NDArray of shape(N, N), dtype = np.uint8
            Binary recurrence matrix indicating whether each pair of points are within the threshold distance.

        Raises
        ------
        ValueError
            - If `u` is not an 1D array, or if its shape does not match the expected system dimension.
            - If `parameters` is not None and does not match the expected number of parameters.
            - If `parameters` is None but the system expects parameters.
            - If `parameters` is a scalar or array-like but not 1D.
            - If `total_time` is negative.
            - If `trasient_time` is negative.
            - If `transient_time` is greater than or equal to total_time.
            - If `lmin` is not a positive integer or is less than 1.
            - If `metric` or `std_metric` is not a valid string.
            - If `threshold` is not within [0, 1].

        TypeError
            - If `u` is not a scalar or array-like type.
            - If `parameters` is not a scalar or array-like type.
            - If `total_time` is not int.
            - If `transient_time` is not int.
            - If `metric` or `std_metric` cannot be converted to a string.
            - If `threshold` is not a positive float.
            - If `lmin` is not an integer.

        """

        u = validate_initial_conditions(
            u, self.__system_dimension, allow_ensemble=False
        )

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        else:
            parameters = validate_parameters(parameters, self.__number_of_parameters)

        validate_non_negative(total_time, "total_time", Integral)
        validate_transient_time(transient_time, total_time, Integral)

        # Configuration handling
        config = RTEConfig(**kwargs)

        if transient_time is not None:
            u = iterate_mapping(u, parameters, transient_time, self.__mapping)
            total_time -= transient_time

        time_series = generate_trajectory(u, parameters, total_time, self.__mapping)
        eps = calculate_threshold(time_series, config)

        # Recurrence matrix calculation
        recmat = build_recurrence_matrix(time_series, eps, config.metric)
        return recmat

    def recurrence_time_entropy(
        self,
        u: numeric_like_t,
        total_time: int_t,
        parameters: numeric_like_t | None = None,
        transient_time: int_t | None = None,
        **kwargs: Any,
    ) -> rte_return_t:
        """
        Compute the recurrence time entropy (RTE) of a trajectory.

        Parameters
        ----------
        u : numeric_like_t
            Initial condition of shape `(system_dimension,)`.
        total_time : int_t
            Total number of iterations used in the computation.
        parameters : numeric_like_t | None, optional
            System parameters passed to the mapping function. If None, the stored
            system parameters are used.
        transient_time : int_t | None, optional
            Number of initial iterations discarded before the computation.

        Other Parameters
        ----------------
        metric : {"supremum", "euclidean", "manhattan"} or callable, optional
            Pairwise metric used to construct the recurrence matrix.
        std_metric : {"supremum", "euclidean", "manhattan"} or callable, optional
            Metric used in the standard-deviation-based threshold calculation.
        threshold : float, optional
            Threshold parameter. Its meaning depends on `threshold_mode`:
            - direct threshold if `threshold_mode="direct"`
            - standard-deviation scale if `threshold_mode="std"`
            - target recurrence rate if `threshold_mode="rr"`
        threshold_mode : {"direct", "std", "rr"} | None, optional
            Mode used to determine the recurrence threshold.
        threshold_std : bool, optional
            Deprecated legacy option. Kept for backward compatibility.
        lmin : int, optional
            Minimum white vertical line length used in the recurrence-time
            distribution.
        return_final_state : bool, optional
            If True, include the final state of the trajectory in the output.
        return_recmat : bool, optional
            If True, include the recurrence matrix in the output.
        return_p : bool, optional
            If True, include the white-vertical-line distribution in the output.

        Returns
        -------
        float or tuple
            - If no optional outputs are requested, returns the scalar RTE value.
            - Otherwise returns a tuple whose first entry is the RTE value, followed
              by the requested outputs in this order:
                1. final state, if `return_final_state=True`
                2. recurrence matrix, if `return_recmat=True`
                3. white-vertical-line distribution, if `return_p=True`

        Raises
        ------
        ValueError
            - If `u` is incompatible with the system dimension.
            - If `parameters` does not match the expected number of parameters.
            - If `total_time` is negative.
            - If `transient_time` is invalid.
            - If the RTE configuration is invalid.
        TypeError
            - If `u` is not a scalar or array-like numeric object.
            - If `parameters` is not a scalar or array-like numeric object.
            - If `total_time` is not an integer.
            - If `transient_time` is not an integer.

        Notes
        -----
        Higher RTE values generally indicate more complex recurrence-time structure.
        Input validation is handled here; the low-level `RTE` routine assumes valid input.
        """
        u_arr: NDArray[np.float64] = validate_initial_conditions(
            u, self.__system_dimension, allow_ensemble=False
        )

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters

        parameters_arr: NDArray[np.float64] = validate_parameters(
            parameters, self.__number_of_parameters
        )

        validate_non_negative(total_time, "total_time", Integral)
        validate_transient_time(transient_time, total_time, Integral)

        RTEConfig(**kwargs)

        return RTE(
            u=u_arr,
            parameters=parameters_arr,
            total_time=total_time,
            mapping=self.__mapping,
            transient_time=transient_time,
            **kwargs,
        )

    def finite_time_recurrence_time_entropy(
        self,
        u: numeric_like_t,
        total_time: int_t,
        finite_time: int_t,
        parameters: numeric_like_t | None = None,
        return_points: bool = False,
        **kwargs: Any,
    ) -> NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Compute the finite-time recurrence time entropy (RTE) over consecutive
        non-overlapping windows of a trajectory.

        Parameters
        ----------
        u : numeric_like_t
            Initial condition of shape ``(system_dimension,)``.
        total_time : int_t
            Total number of iterations used in the computation.
        finite_time : int_t
            Length of each non-overlapping analysis window.
        parameters : numeric_like_t | None, optional
            System parameters passed to the mapping function. If None, the stored
            system parameters are used.
        return_points : bool, optional
            If True, also return the final phase-space point of each window.
        **kwargs : Any
            Additional keyword arguments forwarded to ``RTE`` through
            ``finite_time_RTE``. Supported options are:

            metric : {"supremum", "euclidean", "manhattan"} or callable, optional
                Pairwise distance metric used to build the recurrence matrix.
            std_metric : {"supremum", "euclidean", "manhattan"} or callable, optional
                Metric used in threshold estimation when ``threshold_mode="std"``.
            threshold : float, optional
                Recurrence threshold value, threshold scale, or target recurrence
                rate depending on ``threshold_mode``.
            threshold_mode : {"direct", "std", "rr"}, optional
                Strategy used to determine the recurrence threshold.
            threshold_std : bool, optional
                Deprecated legacy option. Retained for backward compatibility.
            lmin : int, optional
                Minimum white vertical line length used in the distribution.
            return_final_state : bool, optional
                Ignored here. The finite-time wrapper manages final-state handling
                internally.
            return_recmat : bool, optional
                Whether to include the recurrence matrix in the low-level RTE call.
            return_p : bool, optional
                Whether to include the white-vertical-line distribution in the
                low-level RTE call.

        Returns
        -------
        NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]]
            - If ``return_points=False``, returns an array of shape
              ``(num_windows,)`` containing the finite-time RTE values.
            - If ``return_points=True``, returns:
                - ``rte_values``: array of shape ``(num_windows,)``
                - ``phase_space_points``: array of shape
                  ``(num_windows, system_dimension)`` containing the final point of
                  each window

        Raises
        ------
        ValueError
            - If ``u`` is not compatible with the system dimension.
            - If ``parameters`` does not match the expected number of parameters.
            - If ``total_time`` is negative.
            - If ``finite_time`` is not positive.
            - If ``finite_time`` is larger than ``total_time`` or otherwise invalid.
        TypeError
            - If ``u`` is not a scalar or array-like numeric object.
            - If ``parameters`` is not a scalar or array-like numeric object.
            - If ``total_time`` is not an integer.
            - If ``finite_time`` is not an integer.
            - If ``return_points`` is not a boolean.

        Notes
        -----
        The trajectory is split into consecutive non-overlapping windows of length
        ``finite_time``. One RTE value is computed per window.

        Examples
        --------
        >>> ftrte = system.finite_time_recurrence_time_entropy(
        ...     u0,
        ...     total_time=50000,
        ...     finite_time=100,
        ...     parameters=params,
        ... )
        """
        u_arr: NDArray[np.float64] = validate_initial_conditions(
            u, self.__system_dimension, allow_ensemble=False
        )

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        parameters_arr: NDArray[np.float64] = validate_parameters(
            parameters, self.__number_of_parameters
        )

        validate_non_negative(total_time, "total_time", Integral)
        validate_positive(finite_time, "finite_time", Integral)
        validate_finite_time(finite_time, total_time)

        if not isinstance(return_points, bool):
            raise TypeError("return_points must be a boolean")

        kwargs = dict(kwargs)
        kwargs.pop("return_final_state", None)

        return finite_time_RTE(
            u=u_arr,
            parameters=parameters_arr,
            total_time=total_time,
            finite_time=finite_time,
            mapping=self.__mapping,
            return_points=return_points,
            **kwargs,
        )
