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
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from numba.core.errors import NumbaExperimentalFeatureWarning
from numpy.typing import NDArray

from pynamicalsys.common.recurrence_quantification_analysis import RTEConfig

from pynamicalsys.common.utils import finite_difference_jacobian, qr, householder_qr

from pynamicalsys.common.types import int_t, numeric_like_t, numeric_t, observable_t

from pynamicalsys.discrete_time.dynamical_indicators import (
    RTE,
    finite_time_hurst_exponent,
    finite_time_RTE,
    hurst_exponent_wrapped,
    lagrangian_descriptors,
)

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
from pynamicalsys.discrete_time.trajectory_analysis import (
    bifurcation_diagram,
    calculate_manifolds,
    classify_stability,
    eigenvalues_and_eigenvectors,
    ensemble_time_average,
    ensemble_trajectories,
    escape_basin_and_time_entering,
    escape_time_exiting,
    find_periodic_orbit,
    find_periodic_orbit_symmetry_line,
    generate_trajectory,
    iterate_mapping,
    period_counter,
    rotation_number,
    survival_probability,
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

from .time_series_metrics import TimeSeriesMetrics as tsm


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

    def get_parameters(self) -> NDArray[np.float64]:
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
        u: Union[NDArray[np.float64], Sequence[float], float],
        total_time: int,
        parameters: Union[
            None, float, Sequence[np.float64], NDArray[np.float64]
        ] = None,
        transient_time: Optional[int] = None,
    ) -> NDArray[np.float64]:
        """Generate trajectory for either single initial condition or ensemble of initial conditions.

        Automatically dispatches to appropriate implementation based on input dimensionality.
        For ensembles, trajectories are concatenated along time dimension for efficient storage.

        Parameters
        ----------
        u : Union[NDArray[np.float64], Sequence[float]]
            Initial condition(s):
            - Single IC: 1D array of shape (d,) where d is system dimension
            - Ensemble: 2D array of shape (n, d) for n initial conditions
            - Also accepts sequence types that will be converted to numpy arrays
            - Scalar (will be converted to 1D array)
        total_time : int
            Total number of iterations to compute
        parameters : Union[NDArray[np.float64], Sequence[float], float], optional
            Parameters of the dynamical system, shape (p,) where p is number of parameters
        transient_time : Optional[int], optional
            Number of initial iterations to discard as transient, by default None
            If provided, must be less than total_time

        Returns
        -------
        NDArray[np.float64]
            Time series array:

            - Single IC: shape (sample_size, d)
            - Ensemble: shape (sample_size * n, d) where sample_size = total_time - (transient_time or 0)

        Raises
        ------
        ValueError
            - If `u` is not a scalar, 1D, or 2D array, or if its shape does not match the expected system dimension.
            - If `u` is a 1D array but its length does not match the system dimension, or if `u` is a 2D array but does not match the expected shape for an ensemble.
            - If `parameters` is not None and does not match the expected number of parameters.
            - If `parameters` is None but the system expects parameters.
            - If `parameters` is a scalar or array-like but not 1D.
            - If `total_time` is negative.
            - If `trasient_time` is negative.
            - If `transient_time` is greater than or equal to total_time.

        TypeError
            - If `u` is not a scalar or array-like type.
            - If `parameters` is not a scalar or array-like type.
            - If `total_time` is not int.
            - If `transient_time` is not int.

        Notes
        -----
        - For ensembles, use reshape() to separate trajectories: result.reshape(n, sample_size, d)

        Examples
        --------
        >>> # Single initial condition
        >>> u0 = np.array([0.1, 0.2])
        >>> ts = system.trajectory(u0, 5000, parameters=[0.5, 1.0])
        >>> ts.shape  # (5000, 2)

        >>> # Ensemble of 100 initial conditions
        >>> ics = np.random.rand(100, 2)
        >>> ts = system.trajectory(ics, 10000, parameters=[1.0, 0.1], transient_time=1000)
        >>> separated = ts.reshape(100, 9000, 2)  # 9000 = 10000-1000
        """

        u = validate_initial_conditions(u, self.__system_dimension, allow_ensemble=True)

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        else:
            parameters = validate_parameters(parameters, self.__number_of_parameters)

        validate_non_negative(total_time, "total_time", Integral)
        validate_transient_time(transient_time, total_time, type_=Integral)

        if u.ndim == 1:
            result = generate_trajectory(
                u, parameters, total_time, self.__mapping, transient_time=transient_time
            )
            if self.__system_dimension == 1:
                return result[:, 0]
            else:
                return result
        else:
            return ensemble_trajectories(
                u, parameters, total_time, self.__mapping, transient_time=transient_time
            )

    def bifurcation_diagram(
        self,
        u: Union[NDArray[np.float64], Sequence[float], float],
        param_index: int,
        param_range: Union[NDArray[np.float64], Tuple[float, float, int]],
        total_time: int,
        parameters: Optional[NDArray[np.float64]] = None,
        transient_time: Optional[int] = None,
        continuation: bool = False,
        return_last_state: bool = False,
        observable_index: int = 0,
    ) -> Union[
        Tuple[NDArray[np.float64], NDArray[np.float64]],
        Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    ]:
        """Compute bifurcation diagram by varying a specified parameter.

        Parameters
        ----------
        u : Union[NDArray[np.float64], Sequence[float]]
            Initial condition vector. Can be:
            - 1D numpy array of shape (d,) where d is system dimension
            - Sequence that can be converted to numpy array
            - Scalar (will be converted to 1D array)
        parameters : Union[None, float, Sequence[np.float64], NDArray[np.float64]], optional
            Base parameter array of shape (p,)
        param_index : int
            Index of parameter to vary in parameters array (0 <= param_index < p)
        param_range : Union[NDArray[np.float64], Tuple[float, float, int]]
            Parameter range specification, either:
            - Precomputed 1D array of parameter values, or
            - Tuple (start, stop, num_points) for linear spacing
        total_time : int, optional
            Total iterations per parameter value, by default 10000
        transient_time : Optional[int], optional
            Burn-in iterations to discard (default: 20% of total_time)
        continuation: bool, optional
            Whether to perform a continuation sweep, i.e., the initial condition for the next parameter value is the last state of the previous parameter.
        return_last_state: bool, optional
            Whether to return the last state at the last parameter value.
        observable_index: int, optional
            Defines the coordinate to be used in the bifurcation diagram (default = 0).

        Returns
        -------
        Tuple[NDArray[np.float64], NDArray[np.float64]]
            Tuple containing:

            - parameter_values: 1D array of varied parameter values
            - observables: 1D array of observable values (after transients)

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
            - If `param_index` is negative or out of bounds for the number of parameters.
            - If `observable_index` is negative or out of bounds for the system dimension.
        TypeError
            - If `u` is not a scalar or array-like type.
            - If `parameters` is not a scalar or array-like type.
            - If `total_time` is not int.
            - If `transient_time` is not int.

        Notes
        -----
        - Uses Numba-optimized bifurcation_diagram function
        - For large total_time, consider using a smaller transient_time
        - The observable function should be vectorized for best performance

        Examples
        --------
        >>> # Basic usage with precomputed parameter range
        >>> param_range = np.linspace(0.5, 1.5, 100)
        >>> u0 = np.array([0.1, 0.1])
        >>> param_vals, obs = sys.bifurcation_diagram(
        ...     u0, 0, param_range, 5000, parameters=[0.5, 1.0])

        >>> # With tuple parameter range
        >>> param_range = (0.5, 1.5, 100)
        >>> param_vals, obs = sys.bifurcation_diagram(
        ...     u0, 0, param_range, 5000, parameters=[0.5, 1.0])

        """

        u = validate_initial_conditions(
            u, self.__system_dimension, allow_ensemble=False
        )

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

        validate_non_negative(param_index, "param_index", Integral)
        if param_index >= self.__number_of_parameters:
            raise ValueError(
                f"param_index {param_index} out of bounds for system with {self.__number_of_parameters} parameters"
            )
        param_values = validate_and_convert_param_range(param_range)

        validate_non_negative(total_time, "total_time", Integral)
        validate_transient_time(transient_time, total_time, Integral)

        validate_non_negative(observable_index, "observable_index", Integral)
        if observable_index >= self.__system_dimension:
            raise ValueError(
                f"observable_index {observable_index} out of bounds for system dimension {self.__system_dimension}"
            )

        def observable_fn(x):
            return x[observable_index]

        return bifurcation_diagram(
            u=u,
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
        u: Union[NDArray[np.float64], Sequence[float], float],
        max_time: int = 10000,
        parameters: Union[
            None, float, Sequence[np.float64], NDArray[np.float64]
        ] = None,
        transient_time: Optional[int] = None,
        tolerance: float = 1e-10,
        min_period: int = 1,
        max_period: int = 1000,
        stability_checks: int = 3,
    ) -> int:
        """Compute the period of a trajectory.

        This function determines the smallest period p where the system satisfies:
        ||x_{n+p} - x_n|| < tolerance for consecutive states after transients.

        Parameters
        ----------
        u : Union[NDArray[np.float64], Sequence[float]], float
            Initial condition of the system. Can be:
            - 1D numpy array of shape (d,) where d is system dimension
            - Sequence that can be converted to numpy array
            - Scalar (will be converted to 1D array)
        parameters : Union[None, float, Sequence[np.float64], NDArray[np.float64]], optional
            Parameters of the dynamical system, shape (p,)
        max_time : int, optional
            Total number of iterations to compute, by default 10000
            Must be sufficiently large to detect periodicity
        transient_time : Optional[int], optional
            Number of initial iterations to discard as transient, by default None
            If None, uses 10% of total_time
        tolerance : float, optional
            Numerical tolerance for period detection (default: 1e-10)
        min_period : int, optional
            Minimum period to consider (default: 1)
        max_period : int, optional
            Maximum period to consider (default: 1000)
        stability_checks : int, optional
            Number of consecutive period matches required (default: 3)

        Returns
        -------
        int
            The detected period of the trajectory.

            - A positive integer indicates a periodic orbit.
            - -1 indicates aperiodic or chaotic behavior.
            - 1 indicates a fixed point.

        Raises
        ------
        ValueError
            - If `u` is not a scalar, or 1D array, or if its shape does not match the expected system dimension.
            - If `parameters` is not None and does not match the expected number of parameters.
            - If `parameters` is None but the system expects parameters.
            - If `parameters` is a scalar or array-like but not 1D.
            - If `max_time` is negative.
            - If `trasient_time` is negative.
            - If `transient_time` is greater than or equal to total_time.
            - If `min_period` is negative or zero.
            - If `max_period` is negative or less than min_period.
            - If `stability_checks` is not larger than 1.
            - If `tolerance` is negative or zero.
        TypeError
            - If `u` is not a scalar or array-like type.
            - If `parameters` is not a scalar or array-like type.
            - If `max_time` is not int.
            - If `transient_time` is not int.
            - If `min_period` is not int.
            - If `max_period` is not int.
            - If `stability_checks` is not int.

        Notes
        -----
        - For reliable results, max_time should be much larger than expected period
        - Fixed points return period 1 (constant signal)
        - Chaotic trajectories return period -1 (no significant autocorrelation peaks)

        Examples
        --------
        >>> # Basic usage
        >>> u0 = np.array([0.1, 0.2])
        >>> params = np.array([0.5, 1.0])
        >>> period = system.period(u0, params)

        >>> # With custom time parameters
        >>> period = system.period(
        ...     u0, params, total_time=5000, transient_time=500)
        """

        #  Validate initial condition
        u = validate_initial_conditions(
            u, self.__system_dimension, allow_ensemble=False
        )

        # Validate parameters
        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        else:
            parameters = validate_parameters(parameters, self.__number_of_parameters)

        # Validate time parameters
        validate_non_negative(max_time, "total_time", Integral)
        validate_transient_time(transient_time, max_time, Integral)

        # Validate min and max period
        validate_positive(min_period, "min_period", Integral)
        validate_positive(max_period, "max_period", Integral)

        # Validate stability checks
        if not isinstance(stability_checks, Integral) or stability_checks < 1:
            raise ValueError("stability_checks must be positive integer")

        # Validade tolerance
        validate_non_negative(tolerance, "tolerance", Real)

        return period_counter(
            u=u,
            parameters=parameters,
            mapping=self.__mapping,
            total_time=max_time,
            transient_time=transient_time,
            tolerance=tolerance,
            min_period=min_period,
            max_period=max_period,
            stability_checks=stability_checks,
        )

    def find_periodic_orbit(
        self,
        grid_points: Union[NDArray[np.float64], Sequence[float]],
        period: int,
        parameters: Union[None, NDArray[np.float64], Sequence[float], float] = None,
        tolerance: float = 1e-5,
        max_iter: int = 1000,
        convergence_threshold: float = 1e-15,
        tolerance_decay_factor: float = 1 / 2,
        verbose: bool = False,
        symmetry_line: Optional[Callable] = None,
        axis: Optional[int] = None,
    ) -> NDArray[np.float64]:
        """
        Find a periodic orbit using iterative grid refinement.

        Parameters
        ----------
        grid_points : np.ndarray
            n
        period : int
            The period of the orbit to find. Must be ≥ 1.
        parameters : Union[None, NDArray[np.float64], Sequence[float], float], optional
            Array of system parameters (shape (p,)) or a scalar to be broadcasted.
        tolerance : float, optional
            Initial periodicity tolerance. Must be positive. Default is 1e-5.
        max_iter : int, optional
            Maximum number of refinement iterations. Must be ≥ 1. Default is 1000.
        convergence_threshold : float, optional
            Convergence threshold for both position and bounding box. Must be positive.
            Default is 1e-15.
        tolerance_decay_factor : float, optional
            Factor by which to reduce tolerance each iteration. Must be in (0, 1).
            Default is 0.25.
        verbose : bool, optional
            Whether to print iteration progress and convergence information. Default is False.
        symmetry_line : Optional[Callable], optional
            A callable function representing a symmetry line in the system. If provided,
            the search will be restricted to points on this line.
        axis : Optional[int], optional
            Axis of symmetry line. Must be 0 (x-axis) or 1 (y-axis)

        Returns
        -------
        np.ndarray
            A 1D array of shape (2,) representing the coordinates of the found periodic orbit.

        Raises
        ------
        ValueError
            - If grid_points is not of shape (grid_size_x, grid_size_y, 2).
            - If period is less than 1.
            - If tolerance is not positive.
            - If max_iter is not positive.
            - If convergence_threshold is not positive.
            - If tolerance_decay_factor is not in (0, 1).
            - If symmetry_line is provided but axis is not specified.
            - If symmetry_line is not callable.
            - If axis is not 0 (x-axis) or 1 (y-axis).
            - If system_dimension is not 2D.
            - If grid_points is a scalar or not a 3D array when symmetry_line is None.

        TypeError
            - If grid_points is not a numpy array or a sequence that can be converted to a numpy array.
            - If parameters is not None and not a numpy array or sequence that can be converted to a numpy array.
            - If period is not an integer.
            - If tolerance is not float.
            - If max_iter is not int.
            - If convergence_threshold is not float.
            - If tolerance_decay_factor is not float.
            - If axis is not an integer.

        Notes
        -----
        This function wraps the core `find_periodic_orbit` function using the instance's mapping.
        The underlying implementation performs a multiscale search for periodic points
        through iterative refinement around previously found periodic locations.
        """

        if self.__system_dimension != 2:
            raise ValueError("find_periodic_orbit is only implemented for 2D systems")

        # Check if symmetry line is provided
        if symmetry_line is not None and axis is None:
            raise ValueError("axis must be provided when symmetry_line is specified")

        # Check if symmetry line is valid
        if symmetry_line is not None and not callable(symmetry_line):
            raise ValueError("symmetry_line must be a callable function")

        # Check if axis is valid
        if axis is not None and axis not in [0, 1]:
            raise ValueError("axis must be 0 (x-axis) or 1 (y-axis)")

        if np.isscalar(grid_points):
            raise ValueError(
                "grid_points must be a 3D array with shape (grid_size_x, grid_size_y, 2) if symmetry_line is None and 1D array otherwise"
            )

        grid_points = np.asarray(grid_points, dtype=np.float64)
        # Validate grid points
        if symmetry_line is None:
            if grid_points.ndim != 3 or grid_points.shape[2] != 2:
                raise ValueError(
                    "grid_points must be a 3D array with shape (grid_size_x, grid_size_y, 2)"
                )
        else:
            if grid_points.ndim != 1:
                raise ValueError(
                    "grid_points must be a 1D array when symmetry_line is provided"
                )

        # Validate parameters
        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        else:
            parameters = validate_parameters(parameters, self.__number_of_parameters)

        # Validate period
        validate_positive(period, "period", Integral)

        # Validate tolerance
        validate_non_negative(tolerance, "tolerance", Real)

        # Validate max_iter
        validate_positive(max_iter, "max_iter", Integral)

        # Validate convergence threshold
        validate_non_negative(convergence_threshold, "convergence_threshold", Real)

        # Validate tolerance decay factor
        validate_non_negative(tolerance_decay_factor, "tolerance_decay_factor", Real)

        if tolerance_decay_factor >= 1:
            raise ValueError("tolerance_decay_factor must be in (0, 1)")

        if symmetry_line is not None:
            return find_periodic_orbit_symmetry_line(
                grid_points,
                parameters,
                self.__mapping,
                period,
                symmetry_line,
                axis,
                tolerance=tolerance,
                max_iter=max_iter,
                convergence_threshold=convergence_threshold,
                tolerance_decay_factor=tolerance_decay_factor,
                verbose=verbose,
            )
        else:
            return find_periodic_orbit(
                grid_points,
                parameters,
                self.__mapping,
                period,
                tolerance=tolerance,
                max_iter=max_iter,
                convergence_threshold=convergence_threshold,
                tolerance_decay_factor=tolerance_decay_factor,
                verbose=verbose,
            )

    def eigenvalues_and_eigenvectors(
        self,
        u: Union[NDArray[np.float64], Sequence[float]],
        period: int,
        parameters: Union[
            None, float, Sequence[np.float64], NDArray[np.float64]
        ] = None,
        normalize: bool = True,
        sort_by_magnitude: bool = True,
    ) -> Tuple[NDArray[np.complex128], NDArray[np.complex128]]:
        """
        Compute eigenvalues and eigenvectors of the Jacobian matrix for a periodic orbit.

        Parameters
        ----------
        u : Union[NDArray[np.float64], Sequence[float]]
            Initial condition of the system. Can be:
            - 1D numpy array of shape (d,) where d is the system dimension
            - Sequence that can be converted to numpy array
        parameters : Union[None, float, Sequence[np.float64], NDArray[np.float64]], optional
            System parameters of shape (p,).
        period : int
            Period of the orbit (must be ≥ 1).
        normalize : bool, optional
            Whether to normalize eigenvectors to unit length (default is True).
        sort_by_magnitude : bool, optional
            Whether to sort eigenvalues and eigenvectors by the magnitude of the eigenvalues (default is True).

        Returns
        -------
        Tuple[NDArray[np.complex128], NDArray[np.complex128]]

            - eigenvalues : (d,) array of complex eigenvalues.
            - eigenvectors : (d, d) array where each column is a normalized eigenvector corresponding to an eigenvalue.

        Raises
        ------
        ValueError
            - If `u` is not a scalar, or 1D array, or if its shape does not match the expected system dimension.
            - If `parameters` is not None and does not match the expected number of parameters.
            - If `parameters` is None but the system expects parameters.
            - If `parameters` is a scalar or array-like but not 1D.
            - If `period` is negative or zero.
        TypeError
            - If `u` is not a scalar or array-like type.
            - If `parameters` is not a scalar or array-like type.
            - If `period` is not int.

        Notes
        -----
        - Computes the Jacobian matrix over `period` iterations using the product of Jacobians.
        - Eigenvectors indicate local directions of stretching or contraction in phase space.
        - Complex eigenvalues appear in conjugate pairs in real-valued systems.

        Examples
        --------
        >>> # Example usage
        >>> from pynamicalsys import DiscreteDynamicalSystem as dds
        >>> obj = dds(model="henon map")
        >>> u0 = np.array([0.1, 0.2])
        >>> params = np.array([1.0, 0.1])
        >>> evals, evecs = obj.eigenvalues_and_eigenvectors(
        ...     u0, params, period=3)
        """

        # Validate initial condition
        u = validate_initial_conditions(
            u, self.__system_dimension, allow_ensemble=False
        )

        # Validate parameters
        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        else:
            parameters = validate_parameters(parameters, self.__number_of_parameters)

        # Validate period
        validate_positive(period, "period", Integral)

        return eigenvalues_and_eigenvectors(
            u,
            parameters,
            self.__mapping,
            self.__jacobian,
            period,
            normalize,
            sort_by_magnitude,
        )

    def classify_stability(
        self,
        u: Union[NDArray[np.float64], Sequence[float]],
        period: int,
        parameters: Union[
            None, float, Sequence[np.float64], NDArray[np.float64]
        ] = None,
        threshold: float = 1.0,
        tol: float = 1e-8,
    ) -> Dict[str, Union[str, NDArray[np.complex128]]]:
        """
        Classify the stability of a periodic orbit using the eigenvalues of the Jacobian matrix for a 2D discrete map.

        Parameters
        ----------
        u : Union[NDArray[np.float64], Sequence[float]]
            Initial condition of the system. Can be:
            - 1D numpy array of shape (2,) where 2 is the system dimension
            - Sequence that can be converted to numpy array
        period : int
            Period of the orbit (must be ≥ 1).
        parameters : Union[None, float, Sequence[np.float64], NDArray[np.float64]], optional
            System parameters of shape (p,).
        threshold : float, optional
            Threshold for stability classification (default is 1.0).
        tol : float, optional
            Tolerance for numerical stability checks (default is 1e-8).

        Returns
        -------
        dict
            Dictionary with:

            - "classification": str
            - "eigenvalues": ndarray
            - "eigenvectors": ndarray

        Raises
        ------
        ValueError
            - If `u` is not a scalar, or 1D array, or if its shape does not match the expected system dimension.
            - If `parameters` is not None and does not match the expected number of parameters.
            - If `parameters` is None but the system expects parameters.
            - If `parameters` is a scalar or array-like but not 1D.
            - If `period` is negative or zero.
        TypeError
            - If `u` is not a scalar or array-like type.
            - If `parameters` is not a scalar or array-like type.
            - If `period` is not int.

        Notes
        -----
        - The classification is based on the eigenvalues of the Jacobian matrix.
        - The eigenvalues are computed over `period` iterations using the product of Jacobians.
        - The classification can be one of:
            - "stable node": All eigenvalues have magnitudes < threshold.
            - "stable spiral": Complex conjugate eigenvalues with magnitudes < threshold.
            - "unstable node": All eigenvalues have magnitudes > threshold.
            - "unstable spiral": Complex conjugate eigenvalues with magnitudes > threshold.
            - "saddle": One eigenvalue > threshold and one < threshold.
            - "center": Real eigenvalues with magnitudes ≈ threshold.
            - "elliptic": Complex eigenvalues with magnitudes ≈ threshold.
            - "marginal or degenerate": Eigenvalues with magnitudes ≈ 1.

        Examples
        --------
        >>> u0 = np.array([0.1, 0.2])
        >>> params = np.array([1.0, 0.1])
        >>> stability = obj.classify_stability(u0, params, period=3)
        >>> print(stability["classification"])  # e.g., "stable node"
        >>> print(stability["eigenvalues"])  # Eigenvalues of the Jacobian
        >>> print(stability["eigenvectors"])
        """

        if self.__system_dimension != 2:
            raise ValueError("classify_stability is only implemented for 2D systems")

        u = validate_initial_conditions(
            u, self.__system_dimension, allow_ensemble=False
        )

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        else:
            parameters = validate_parameters(parameters, self.__number_of_parameters)

        validate_positive(period, "period", Integral)

        return classify_stability(
            u,
            parameters,
            self.__mapping,
            self.__jacobian,
            period,
            threshold=threshold,
            tol=tol,
        )

    def manifold(
        self,
        u: Union[NDArray[np.float64], Sequence[float]],
        period: int,
        parameters: Union[
            None, float, Sequence[np.float64], NDArray[np.float64]
        ] = None,
        delta: float = 1e-4,
        n_points: Union[NDArray[np.int32], List[int], int] = 100,
        iter_time: Union[List[int], int] = 100,
        stability: str = "unstable",
    ) -> List[np.ndarray]:
        """Calculate stable or unstable manifolds of a saddle periodic orbit of the system.

        Parameters
        ----------
        u : Union[NDArray[np.float64], Sequence[float]]
            Initial condition of the system. Can be:
            - 1D numpy array of shape (2,) where 2 is the system dimension
            - Sequence that can be converted to numpy array
        period : int
            Period of the orbit (must be ≥ 1)
        parameters : Union[None, float, Sequence[np.float64], NDArray[np.float64]], optional
            Parameters of the dynamical system. Can be:

            - 1D numpy array of shape (p,) where p is the number of parameters
            - Sequence that can be converted to numpy array
            - Scalar value (will be broadcasted)
        delta : float, optional
            Initial displacement from orbit (default: 1e-4)
        n_points : Union[List[int], int], optional
            Number of points per branch (default: 100)
        iter_time : Union[List[int], int], optional
            Iterations per branch (default: 100)
        stability : str, optional
            "stable" or "unstable" manifold (default: "unstable")

        Returns
        -------
        List[np.ndarray]
            List containing two arrays: [0] is upper branch manifold points and [1] is lower branch manifold points. Each array has shape (n_points * iter_time, 2)

        Raises
        ------
        ValueError
            - If `u` is not a scalar, or 1D array, or if its shape does not match the expected system dimension.
            - If `parameters` is not None and does not match the expected number of parameters.
            - If `parameters` is None but the system expects parameters.
            - If `parameters` is a scalar or array-like but not 1D.
            - If `period` is negative or zero.
            - If `delta` is negative or zero.
            - If `n_points` is not a positive integer or a list of two positive integers.
            - If `iter_time` is not a positive integer or a list of two positive integers.
            - If `stability` is not "stable" or "unstable".
            - If system dimension is not 2D.
        TypeError
            - If `u` is not a scalar or array-like type.
            - If `parameters` is not a scalar or array-like type.
            - If `period` is not int.
        RuntimeError
            - If `stability` is "stable" but backwards mapping function is not defined.

        Notes
        -----
        - Works only for 2D systems
        - The periodic orbit must be a saddle point
        - Manifold quality depends on:
        - delta (smaller = closer to linear approximation)
        - n_points (more = smoother manifold)
        - iter_time (more = longer manifold)

        Examples
        --------
        >>> # Example usage
        >>> from pynamicalsys import DiscreteDynamicalSystem as dds
        >>> # Define the system
        >>> obj = dds(model="standard map")
        >>> # Calculate unstable manifold
        >>> mani = obj.manifold(
        ...     orbit_point, params,
        ...     period=3, delta=1e-5, n_points=200, iter_time=500)
        >>> upper_branch, lower_branch = manifolds
        """

        if self.__system_dimension != 2:
            raise ValueError("manifold is only implemented for 2D systems")

        if self.__backwards_mapping is None and stability == "stable":
            raise RuntimeError("Backwards mapping function must be provided")

        u = validate_initial_conditions(
            u, self.__system_dimension, allow_ensemble=False
        )

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        else:
            parameters = validate_parameters(parameters, self.__number_of_parameters)

        validate_positive(period, "period", Integral)

        validate_non_negative(delta, "delta", Real)

        # Validate n_points
        if isinstance(n_points, int):
            # If n_points is a single integer, make it a list of two identical integers
            n_points = [n_points] * 2
            n_points = np.asarray(n_points, dtype=np.int32)
        elif isinstance(n_points, (list, np.ndarray)):
            if len(n_points) != 2:
                raise ValueError("n_points must be a list or array of two integers")
            if not all(isinstance(n, int) and n > 0 for n in n_points):
                raise ValueError("n_points must be a list of two positive integers")
            n_points = np.asarray(n_points, dtype=np.int32)
        else:
            raise ValueError("n_points must be an int or a list of two ints")

        return calculate_manifolds(
            u,
            parameters,
            self.__mapping,
            self.__backwards_mapping,
            self.__jacobian,
            period,
            delta=delta,
            n_points=n_points,
            iter_time=iter_time,
            stability=stability,
        )

    def rotation_number(
        self,
        u: Union[NDArray[np.float64], Sequence[float], float],
        total_time: int,
        parameters: Union[
            None, float, Sequence[np.float64], NDArray[np.float64]
        ] = None,
        mod: int = 1,
    ) -> float:
        """Compute the rotation number of a trajectory.

        Parameters
        ----------
        u : Union[NDArray[np.float64], Sequence[float], float]
            Initial condition of the system. Can be:
            - 1D numpy array of shape (d,) where d is system dimension
            - Sequence that can be converted to numpy array
            - Scalar value
        total_time : int
            Total number of iterations to compute
        parameters : Union[None, float,
            Sequence[np.float64], NDArray[np.float64]]
            Parameters of the dynamical system, shape (p,)
        mod : int, optional
            Modulus for the rotation number calculation, by default 1

        Returns
        -------
        float
            The computed rotation number.

        Raises
        ------
        ValueError
            - If `u` is not a scalar, or 1D array, or if its shape does not match the expected system dimension.
            - If `parameters` is not None and does not match the expected number of parameters.
            - If `parameters` is None but the system expects parameters.
            - If `parameters` is a scalar or array-like but not 1D.
            - If `total_time` is negative.
        TypeError
            - If `u` is not a scalar or array-like type.
            - If `parameters` is not a scalar or array-like type.
            - If `total_time` is not int.

        Notes
        -----
        - The rotation number is a measure of the average angular displacement
          of a trajectory in phase space.
        - It is computed as the limit of the average angular displacement
          over a large number of iterations.
        - The rotation number is useful for analyzing the behavior of
          periodic orbits and chaotic dynamics.

        Examples
        --------
        >>> # Basic usage
        >>> u0 = np.array([0.1, 0.2])
        >>> params = np.array([0.5, 1.0])
        >>> rotation_num = system.compute_rotation_number(u0, params)
        >>> # With custom time parameters
        >>> rotation_num = system.compute_rotation_number(
        ...     u0, params, total_time=5000)
        """

        if self.__system_dimension != 2:
            raise ValueError("rotation_number is only implemented for 2D systems")

        u = validate_initial_conditions(
            u, self.__system_dimension, allow_ensemble=False
        )

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        else:
            parameters = validate_parameters(parameters, self.__number_of_parameters)

        validate_non_negative(total_time, "total_time", Integral)

        return rotation_number(u, parameters, total_time, self.__mapping, mod=mod)

    def escape_analysis(
        self,
        u: Union[NDArray[np.float64], Sequence[float]],
        max_time: int,
        exits: Union[List[NDArray[np.float64]], NDArray[np.float64]],
        parameters: Union[
            None, float, Sequence[np.float64], NDArray[np.float64]
        ] = None,
        escape: str = "entering",
        hole_size: Optional[float] = None,
    ) -> Tuple[int, int]:
        """Compute escape basin index and time for a single trajectory.

        Parameters
        ----------
        u : Union[NDArray[np.float64], Sequence[float]]
            Initial state vector of shape (d,) where d is system dimension.
            Can be any sequence convertible to numpy array.
        max_time : int
            Maximum number of iterations to simulate (must be positive).
        exits : Union[List[NDArray[np.float64]], NDArray[np.float64]]
            - Exit regions specification:
                - List of d arrays of shape (2,) representing [min, max] per dimension
                - Array of shape (n_exits, d, 2) for multiple exit regions
        parameters : Union[None, float, Sequence[np.float64], NDArray[np.float64]], optional
            System parameters of shape (p,) passed to the mapping function.
        escape : str, optional
            Escape condition type: "entering" or "exiting" (default "entering").
        hole_size : Optional[float], optional
            Size of the hole (default None, meaning no size constraint). Only used for "entering" escape type.

        Returns
        -------
        Tuple[int, int]
            A tuple containing:

            - exit_index: 0-based index of escape region (-1 if no escape)
            - escape_time: Time step of escape (max_time if no escape)

        Raises
        ------
        ValueError
            - If `u` is not a scalar, or 1D array, or if its shape does not match the expected system dimension.
            - If `parameters` is not None and does not match the expected number of parameters.
            - If `parameters` is None but the system expects parameters.
            - If `parameters` is a scalar or array-like but not 1D.
            - If `max_time` is negative or zero.
            - If `exits` is not a list of (d,2) arrays or (n,d,2) array.
            - If `escape` is not "entering" or "exiting".
            - If exit regions do not match system dimension.
            - If exit regions do not provide [min, max] pairs.
        TypeError
            - If `u` is not a scalar or array-like type.
            - If `parameters` is not a scalar or array-like type.
            - If `max_time` is not int.

        Notes
        -----
        - For "entering": trajectory must enter the exit region
        - For "exiting": trajectory must exit the region of interest
        - Exit regions are defined as hyperrectangles [min, max] in each dimension

        Examples
        --------
        >>> # Single exit region (entering)
        >>> u0 = np.array([0.1, 0.2])
        >>> params = np.array([1.0, 0.1])
        >>> exit_region = np.array([[-1, 1], [-1, 1]])  # 2D box
        >>> idx, time = sys.escape_analysis(u0, params, 1000, exit_region)

        >>> # Multiple exit regions (exiting)
        >>> exits = [
        ...     np.array([[0, 1], [0, 1]]),  # First exit region
        ...     np.array([[-1, 0], [-1, 0]])  # Second exit region
        ... ]
        >>> idx, time = sys.escape_analysis(u0, params, 1000, exits, "exiting")
        """

        u = validate_initial_conditions(
            u, self.__system_dimension, allow_ensemble=False
        )

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        else:
            parameters = validate_parameters(parameters, self.__number_of_parameters)

        validate_non_negative(max_time, "max_time", Integral)

        # Validate escape type
        if escape not in ("entering", "exiting"):
            raise ValueError("escape must be either 'entering' or 'exiting'")

        if escape == "entering" and hole_size is None:
            raise ValueError("hole_size must be specified for 'entering' escape type")

        # Process exit regions
        if escape == "entering":
            # If exits is a list, convert to an array
            if isinstance(exits, list):
                exits_arr = np.stack(exits, axis=0)
            else:
                exits_arr = np.asarray(exits, dtype=np.float64)

            # If exits is a single point, convert to 2D array
            if exits_arr.ndim == 1:
                exits_arr = exits_arr.reshape(1, -1)

            # Validate exits array shape
            if exits_arr.ndim != 2:
                raise ValueError(
                    "Exits must be a list of (d,) arrays or a 2D array of shape (n, d)"
                )

            # Validate exits dimension
            if exits_arr.shape[1] != self.__system_dimension:
                raise ValueError(
                    f"Exit region dimension {exits_arr.shape[1]} != system dimension {self.__system_dimension}"
                )

            # Create the exit regions as hyperrectangles
            # Stack per coordinate axis
            lower = exits_arr - hole_size / 2
            upper = exits_arr + hole_size / 2
            exits_arr = np.stack([lower.T, upper.T], axis=1).transpose(2, 0, 1)

        if escape == "exiting":
            if isinstance(exits, list):
                exits_arr = np.asarray(exits, dtype=np.float64)
            else:
                exits_arr = np.asarray(exits, dtype=np.float64)

            # Validate exits array shape
            if exits_arr.ndim != 2 or exits_arr.shape[1] != 2:
                raise ValueError(
                    "Exits must be a 2D array of shape (d, 2) for exiting escape type"
                )

            # Validate exits dimension
            if exits_arr.shape[0] != self.__system_dimension:
                raise ValueError(
                    f"Exit region dimension {exits_arr.shape[0]} != system dimension {self.__system_dimension}"
                )

        # Dispatch to appropriate computation
        if escape == "entering":
            return escape_basin_and_time_entering(
                u=u,
                parameters=parameters,
                mapping=self.__mapping,
                max_time=max_time,
                exits=exits_arr,
            )
        else:
            return escape_time_exiting(
                u=u,
                parameters=parameters,
                mapping=self.__mapping,
                max_time=max_time,
                region_limits=exits_arr,
            )

    def survival_probability(
        self, escape_times: Union[NDArray[np.int32], Sequence[int]], max_time: np.int32
    ) -> Tuple[NDArray[np.int64], NDArray[np.float64]]:
        """Compute the survival probability based on escape times.

        Parameters
        ----------
        escape_times : Union[NDArray[np.float64], Sequence[int]]
            Array of escape times for N trajectories where:
            - escape_times[i] = time when i-th trajectory escaped
            - Use max_time for trajectories that didn't escape
            - Should be shape (N,) with dtype=int32
        max_time : int
            Maximum simulation time (must be > 0)

        Returns
        -------
        NDArray[np.float64][float64]
            Survival probability curve S(t) where:

            - S[0] = 1.0 (all trajectories survive at t=0)
            - S[t] = fraction surviving at time t
            - Shape (max_time + 1,)

        Raises
        ------
        ValueError
            - If escape_times contains values > max_time
            - If escape_times contains negative values
            - If max_time <= 0
        TypeError
            - If escape_times cannot be converted to int32 array

        Notes
        -----
        - S(t) = P(T > t) where T is escape time
        - Implemented via survival_probability() function
        - For N trajectories: S(t) = (number of T_i > t) / N

        Examples
        --------
        >>> escape_times = np.array([5, 10, 10, 20], dtype=np.int32)
        >>> surv = system.compute_survival_probability(escape_times, 20)
        >>> surv[0]   # 1.0 at t=0
        >>> surv[5]   # 0.75 at t=5
        >>> surv[10]  # 0.25 at t=10
        >>> surv[20]  # 0.0 at t=20
        """
        # Input validation
        try:
            escape_arr = np.asarray(escape_times, dtype=np.int32)
        except (TypeError, ValueError) as e:
            raise TypeError("escape_times must be convertible to int32 array") from e

        if escape_arr.ndim != 1:
            raise ValueError("escape_times must be 1D array")

        validate_non_negative(max_time, "max_time", Integral)

        if np.any(escape_arr < 0):
            raise ValueError("escape_times cannot contain negative values")

        if np.any(escape_arr > max_time):
            raise ValueError(f"escape_times cannot exceed max_time ({max_time})")

        # Compute survival probability
        return survival_probability(escape_arr, max_time)

    def diffusion_coefficient(
        self,
        u: Union[NDArray[np.float64], Sequence[Sequence[float]]],
        total_time: int,
        parameters: Union[
            None, float, Sequence[np.float64], NDArray[np.float64]
        ] = None,
        axis: int = 1,
    ) -> np.float64:
        """Compute the diffusion coefficient from ensemble trajectories.

        Parameters
        ----------
        u : Union[NDArray[np.float64], Sequence[Sequence[float]]]
            Initial conditions array where:
            - Shape (N, d) for N trajectories in d-dimensional space
            - Can be list of lists or numpy array
        total_time : int
            Number of iterations to compute (must be ≥ 1)
        parameters : Union[None, float, Sequence[np.float64], NDArray[np.float64]], optional
            System parameters passed to mapping function, shape (p,)
        axis : int, default=1
            Coordinate index to compute diffusion (0 for x, 1 for y, etc.)

        Returns
        -------
        float
            Diffusion coefficient D calculated as:
            D = ⟨(y(t) - y(0))²⟩/(2t) where y is typically the second coordinate and ⟨·⟩ denotes ensemble average

        Raises
        ------
        ValueError
            - If `u` is not a 2D array, or if its shape does not match the expected system dimension.
            - If `parameters` is not None and does not match the expected number of parameters.
            - If `parameters` is None but the system expects parameters.
            - If `parameters` is a scalar or array-like but not 1D.
            - If `total_time` is negative or zero.
            - If `axis` is not valid for the system dimension.
        TypeError
            - If `u` is not a scalar or array-like type.
            - If `parameters` is not a scalar or array-like type.
            - If `total_time` is not int.
            - If `axis` is not int.

        Notes
        -----
        - Uses the system's mapping function for evolution
        - For accurate results, use:
        - total_time >> 1
        - N >> 1 initial conditions
        - Implements Einstein relation for discrete time

        Examples
        --------
        >>> # With numpy array input
        >>> ics = np.random.rand(100, 2)  # 100 trajectories in 2D
        >>> params = np.array([0.5, 1.0])
        >>> D = system.diffusion_coefficient(ics, params, 1000)

        >>> # With list input
        >>> ics = [[0.1, 0.2], [0.3, 0.4]]  # 2 trajectories
        >>> D = system.diffusion_coefficient(ics, params, 500)
        """

        u = validate_initial_conditions(u, self.__system_dimension, allow_ensemble=True)

        if u.ndim != 2:
            raise ValueError(
                f"Initial conditions must be a 2D array of shape (N, d), got shape {u.shape}"
            )

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        else:
            parameters = validate_parameters(parameters, self.__number_of_parameters)

        validate_non_negative(total_time, "total_time", Integral)

        validate_axis(axis, self.__system_dimension)

        return diffusion_coefficient(
            u, parameters, total_time, self.__mapping, axis=axis
        )

    def average_in_time(
        self,
        u: Union[NDArray[np.float64], Sequence[Sequence[float]]],
        total_time: int,
        parameters: Union[
            None, float, Sequence[np.float64], NDArray[np.float64]
        ] = None,
        sample_times: Optional[Union[NDArray[np.float64], Sequence[int]]] = None,
        axis: int = 1,
    ) -> NDArray[np.float64]:
        """Compute time evolution of coordinate average across trajectories.

        Parameters
        ----------
        u : Union[NDArray[np.float64], Sequence[Sequence[float]]]
            Initial conditions array where:
            - Shape (N, d) for N trajectories in d-dimensional space
            - Can be list of lists or numpy array
        total_time : int
            Total number of iterations to compute (must be ≥ 1)
        parameters : Union[None, float, Sequence[np.float64], NDArray[np.float64]], optional
            System parameters passed to mapping function, shape (p,)
        sample_times : Optional[Union[NDArray[np.float64], Sequence[int]]], default=None
            Specific time steps to record (1D array of integers). If None,
            records at every time step from 0 to total_time.
        axis : int, default=1
            Coordinate index to average over (0 for x, 1 for y, etc.)

        Returns
        -------
        NDArray[np.float64]
            Array of average values with shape:

            - (len(sample_times),) if sample_times provided
            - (total_time + 1,) if sample_times=None

        Raises
        ------
        ValueError
            - If `u` is not a 2D array, or if its shape does not match the expected system dimension.
            - If `parameters` is not None and does not match the expected number of parameters.
            - If `parameters` is None but the system expects parameters.
            - If `parameters` is a scalar or array-like but not 1D.
            - If `total_time` is negative or zero.
            - If `sample_times` contains invalid values.
            - If `sample_times` is not a 1D array of integers.
            - If `axis` is not valid for the system dimension.
        TypeError
            - If `u` is not a scalar or array-like type.
            - If `parameters` is not a scalar or array-like type.
            - If `total_time` is not int.
            - If `axis` is not int.

        Notes
        -----
        - Uses the system's mapping function for trajectory evolution
        - For smooth results, use N >> 1 initial conditions
        - The average is computed as ⟨xᵢ(t)⟩ where i is the axis index
        - First output value (t=0) is the initial average

        Examples
        --------
        >>> # Basic usage with default sampling
        >>> ics = np.random.rand(100, 2)  # 100 trajectories in 2D
        >>> params = np.array([1.0, 0.1])
        >>> avg = system.average_in_time(ics, params, 1000)

        >>> # With custom sampling times
        >>> times = np.linspace(0, 1000, 11, dtype=int)
        >>> avg = system.average_in_time(ics, params, 1000, times)
        """

        u = validate_initial_conditions(u, self.__system_dimension, allow_ensemble=True)

        if u.ndim != 2:
            raise ValueError(
                f"Initial conditions must be a 2D array of shape (N, d), got shape {u.shape}"
            )

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        else:
            parameters = validate_parameters(parameters, self.__number_of_parameters)

        validate_non_negative(total_time, "total_time", Integral)

        sample_times_arr = validate_sample_times(sample_times, total_time)

        validate_axis(axis, self.__system_dimension)

        return average_vs_time(
            u,
            parameters,
            total_time,
            self.__mapping,
            sample_times=sample_times_arr,
            axis=axis,
        )

    def cumulative_average(
        self,
        u: Union[NDArray[np.float64], Sequence[Sequence[float]]],
        total_time: int,
        parameters: Union[
            None, float, Sequence[np.float64], NDArray[np.float64]
        ] = None,
        sample_times: Optional[Union[NDArray[np.float64], Sequence[int]]] = None,
        axis: int = 1,
    ) -> NDArray[np.float64]:
        """Compute cumulative average of a coordinate across trajectories.

        Parameters
        ----------
        u : Union[NDArray[np.float64], Sequence[Sequence[float]]]
            Initial conditions array where:
            - Shape (N, d) for N trajectories in d-dimensional space
            - Can be list of lists or numpy array
        parameters : Union[None, float, Sequence[np.float64], NDArray[np.float64]], optional
            System parameters passed to mapping function, shape (p,)
        total_time : int
            Total number of iterations to compute (must be ≥ 1)
        sample_times : Optional[Union[NDArray[np.float64], Sequence[int]]], default=None
            Specific time steps to record (1D array of integers). If None,
            records at every time step from 0 to total_time.
        axis : int, default=1
            Coordinate index to average over (0 for x, 1 for y, etc.)

        Returns
        -------
        NDArray[np.float64]
            Array of average values with shape:

            - (len(sample_times),) if sample_times provided
            - (total_time + 1,) if sample_times=None

        Raises
        ------
        ValueError
            - If `u` is not a 2D array, or if its shape does not match the expected system dimension.
            - If `parameters` is not None and does not match the expected number of parameters.
            - If `parameters` is None but the system expects parameters.
            - If `parameters` is a scalar or array-like but not 1D.
            - If `total_time` is negative or zero.
            - If `sample_times` contains invalid values.
            - If `sample_times` is not a 1D array of integers.
            - If `axis` is not valid for the system dimension.
        TypeError
            - If `u` is not a scalar or array-like type.
            - If `parameters` is not a scalar or array-like type.
            - If `total_time` is not int.
            - If `axis` is not int.

        Notes
        -----
        - Uses the system's mapping function for trajectory evolution
        - For smooth results, use N >> 1 initial conditions
        - The average is computed as ⟨xᵢ(t)⟩ where i is the axis index
        - First output value (t=0) is the initial average

        Examples
        --------
        >>> # Basic usage with default sampling
        >>> ics = np.random.rand(100, 2)  # 100 trajectories in 2D
        >>> params = np.array([1.0, 0.1])
        >>> avg = system.cumulative_average(ics, params, 1000)

        >>> # With custom sampling times
        >>> times = np.linspace(0, 1000, 11, dtype=int)
        >>> avg = system.cumulative_average(ics, params, 1000, times)
        """

        u = validate_initial_conditions(u, self.__system_dimension, allow_ensemble=True)

        if u.ndim != 2:
            raise ValueError(
                f"Initial conditions must be a 2D array of shape (N, d), got shape {u.shape}"
            )

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        else:
            parameters = validate_parameters(parameters, self.__number_of_parameters)

        validate_non_negative(total_time, "total_time", Integral)

        sample_times_arr = validate_sample_times(sample_times, total_time)

        validate_axis(axis, self.__system_dimension)

        return cumulative_average_vs_time(
            u,
            parameters,
            total_time,
            self.__mapping,
            sample_times=sample_times_arr,
            axis=axis,
        )

    def root_mean_squared(
        self,
        u: Union[NDArray[np.float64], Sequence[Sequence[float]]],
        total_time: int,
        parameters: Union[
            None, float, Sequence[np.float64], NDArray[np.float64]
        ] = None,
        sample_times: Optional[Union[NDArray[np.float64], Sequence[int]]] = None,
        axis: int = 1,
    ) -> NDArray[np.float64]:
        """Compute root mean squared (RMS) evolution of a coordinate across trajectories.

        Parameters
        ----------
        u : Union[NDArray[np.float64], Sequence[Sequence[float]]]
            Initial conditions array where:
            - Shape (N, d) for N trajectories in d-dimensional space
            - Can be list of lists or numpy array
        total_time : int
            Total number of iterations to compute (must be ≥ 1)
        parameters : Union[None, float, Sequence[np.float64], NDArray[np.float64]], optional
            System parameters passed to mapping function, shape (p,)
            Must be 1D float array
        sample_times : Optional[Union[NDArray[np.float64], Sequence[int]]], default=None
            Specific time steps to record (1D array of integers). If None,
            records at every time step from 0 to total_time.
        axis : int, default=1
            Coordinate index for RMS calculation (0 for x, 1 for y, etc.)

        Returns
        -------
        NDArray[np.float64]
            root mean squared values with shape:

            - (len(sample_times),) if sample_times provided
            - (total_time + 1,) if sample_times=None

        Raises
        ------
        ValueError
            - If `u` is not a 2D array, or if its shape does not match the expected system dimension.
            - If `parameters` is not None and does not match the expected number of parameters.
            - If `parameters` is None but the system expects parameters.
            - If `parameters` is a scalar or array-like but not 1D.
            - If `total_time` is negative or zero.
            - If `sample_times` contains invalid values.
            - If `sample_times` is not a 1D array of integers.
            - If `axis` is not valid for the system dimension.
        TypeError
            - If `u` is not a scalar or array-like type.
            - If `parameters` is not a scalar or array-like type.
            - If `total_time` is not int.
            - If `axis` is not int.

        Notes
        -----
        - root mean squared is computed as sqrt(⟨xᵢ(t)²⟩) where:
        - i is the axis index
        - ⟨·⟩ denotes ensemble average
        - First output value (t=0) is the initial RMS
        - For diffusion analysis, often used with axis=1 (y-coordinate)

        Examples
        --------
        >>> # Basic usage with default sampling
        >>> ics = np.random.rand(100, 2)  # 100 trajectories in 2D
        >>> params = np.array([1.0, 0.1], dtype=np.float64)
        >>> rms = system.root_mean_squared(ics, params, 1000)

        >>> # With custom sampling times and x-coordinate (axis=0)
        >>> times = np.arange(0, 1001, 100, dtype=int)
        >>> rms = system.root_mean_squared(ics, params, 1000, times, axis=0)
        """

        u = validate_initial_conditions(u, self.__system_dimension, allow_ensemble=True)

        if u.ndim != 2:
            raise ValueError(
                f"Initial conditions must be a 2D array of shape (N, d), got shape {u.shape}"
            )

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        else:
            parameters = validate_parameters(parameters, self.__number_of_parameters)

        validate_non_negative(total_time, "total_time", Integral)

        sample_times_arr = validate_sample_times(sample_times, total_time)

        validate_axis(axis, self.__system_dimension)

        return root_mean_squared(
            u,
            parameters,
            total_time,
            self.__mapping,
            sample_times=sample_times_arr,
            axis=axis,
        )

    def mean_squared_displacement(
        self,
        u: Union[NDArray[np.float64], Sequence[Sequence[float]]],
        total_time: int,
        parameters: Union[
            None, float, Sequence[np.float64], NDArray[np.float64]
        ] = None,
        sample_times: Optional[Union[NDArray[np.int32], Sequence[int]]] = None,
        axis: int = 1,
    ) -> NDArray[np.float64]:
        """Compute the Mean Squared Displacement (MSD) for system trajectories.

        Parameters
        ----------
        u : Union[NDArray[np.float64], Sequence[Sequence[float]]]
            Initial conditions array where:
            - Shape (N, d) for N trajectories in d-dimensional space
            - Can be list of lists or numpy array
        total_time : int
            Total number of iterations (must be > transient_time)
        parameters : Union[None, float, Sequence[np.float64], NDArray[np.float64]], optional
            System parameters of shape (p,) passed to mapping function
        sample_times : Optional[Union[NDArray[np.float64], Sequence[int]]], default=None
            Specific time steps to record (1D array of integers). If None,
            records at every time step after transient_time.
        axis : int, default=1
            Coordinate index to analyze (0 for x, 1 for y, etc.)
        transient_time : Optional[int], default=None
            Initial iterations to discard (default: 0 if None)

        Returns
        -------
        NDArray[np.float64]
            Mean Squared Displacement values with shape:

            - (len(sample_times),) if sample_times provided
            - (total_time - transient_time,) if sample_times=None

        Raises
        ------
        ValueError
            - If `u` is not a 2D array, or if its shape does not match the expected system dimension.
            - If `parameters` is not None and does not match the expected number of parameters.
            - If `parameters` is None but the system expects parameters.
            - If `parameters` is a scalar or array-like but not 1D.
            - If `total_time` is negative or zero.
            - If `sample_times` contains invalid values.
            - If `sample_times` is not a 1D array of integers.
            - If `axis` is not valid for the system dimension.
        TypeError
            - If `u` is not a scalar or array-like type.
            - If `parameters` is not a scalar or array-like type.
            - If `total_time` is not int.
            - If `axis` is not int.

        Notes
        -----
        - Mean Squared Displacement is calculated as ⟨(x_i(t) - x_i(0))²⟩ where ⟨·⟩ is ensemble average
        - For normal diffusion, Mean Squared Displacement ∝ t
        - For anomalous diffusion, Mean Squared Displacement ∝ t^α (α≠1)
        - Uses parallel processing for efficient computation

        Examples
        --------
        >>> # Basic usage with default sampling
        >>> ics = np.random.rand(100, 2)  # 100 trajectories in 2D
        >>> params = np.array([1.0, 0.1])
        >>> msd_vals = system.mean_squared_displacement(ics, params, 1000)

        >>> # With custom sampling times
        >>> times = np.arange(0, 1000, 10, dtype=int)
        >>> msd_vals = system.mean_squared_displacement(ics, params, 1000, sample_times=times)
        """

        u = validate_initial_conditions(u, self.__system_dimension, allow_ensemble=True)

        if u.ndim != 2:
            raise ValueError(
                f"Initial conditions must be a 2D array of shape (N, d), got shape {u.shape}"
            )

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        else:
            parameters = validate_parameters(parameters, self.__number_of_parameters)

        validate_non_negative(total_time, "total_time", Integral)

        sample_times_arr = validate_sample_times(sample_times, total_time)

        validate_axis(axis, self.__system_dimension)

        return mean_squared_displacement(
            u,
            parameters,
            total_time,
            self.__mapping,
            sample_times=sample_times_arr,
            axis=axis,
        )

    def ensemble_time_average(
        self,
        u: Union[NDArray[np.float64], Sequence[Sequence[float]]],
        total_time: int,
        parameters: Union[
            None, float, Sequence[np.float64], NDArray[np.float64]
        ] = None,
        axis: int = 1,
    ) -> NDArray[np.float64]:
        """Compute ensemble time average of a coordinate across trajectories.

        Parameters
        ----------
        u : Union[NDArray[np.float64], Sequence[Sequence[float]]]
            Initial conditions array where:
            - Shape (N, d) for N trajectories in d-dimensional space
            - Can be list of lists or numpy array
        total_time : int
            Total number of iterations to compute (must be ≥ 1)
        parameters : Union[None, float, Sequence[np.float64], NDArray[np.float64]], optional
            System parameters passed to mapping function, shape (p,)
        axis : int, default=1
            Coordinate index to average over (0 for x, 1 for y, etc.)

        Returns
        -------
        NDArray[np.float64]
            Array of average values with shape (u.shape[0],)

        Raises
        ------
        ValueError
            - If `u` is not a 2D array, or if its shape does not match the expected system dimension.
            - If `parameters` is not None and does not match the expected number of parameters.
            - If `parameters` is None but the system expects parameters.
            - If `parameters` is a scalar or array-like but not 1D.
            - If `total_time` is negative or zero.
            - If `axis` is not valid for the system dimension.
        TypeError
            - If `u` is not a scalar or array-like type.
            - If `parameters` is not a scalar or array-like type.
            - If `total_time` is not int.
            - If `axis` is not int.

        Notes
        -----
        - Uses the system's mapping function for trajectory evolution
        - For smooth results, use N >> 1 initial conditions
        - The average is computed as ⟨xᵢ(t)⟩ where i is the axis index
        - First output value (t=0) is the initial average

        Examples
        --------
        >>> # Basic usage with default axis (1)
        >>> ics = np.random.rand(100, 2)  # 100 trajectories in 2D
        >>> params = np.array([1.0, 0.1])
        >>> avg = system.ensemble_time_average(ics, params, 1000)
        >>> # With custom axis (0 for x-coordinate)
        >>> avg_x = system.ensemble_time_average(ics, params, 1000, axis=0)
        """

        u = validate_initial_conditions(u, self.__system_dimension, allow_ensemble=True)

        if u.ndim != 2:
            raise ValueError(
                f"Initial conditions must be a 2D array of shape (N, d), got shape {u.shape}"
            )

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        else:
            parameters = validate_parameters(parameters, self.__number_of_parameters)

        validate_non_negative(total_time, "total_time", Integral)

        validate_axis(axis, self.__system_dimension)

        return ensemble_time_average(
            u, parameters, self.__mapping, total_time, axis=axis
        )

    def recurrence_times(
        self,
        u: Union[NDArray[np.float64], Sequence[float], float],
        total_time: int,
        parameters: Union[
            None, float, Sequence[np.float64], NDArray[np.float64]
        ] = None,
        eps: float = 1e-2,
        transient_time: Optional[int] = None,
    ) -> NDArray[np.float64]:
        """
        Compute recurrence times to a neighborhood around the initial condition.

        Parameters
        ----------
        u : Union[NDArray[np.float64], list, tuple]
            Initial condition vector (shape: `(neq,)`). Will be converted to a contiguous float64 NumPy array.
        total_time : int
            Total number of iterations to simulate. Must be a positive integer.
        parameters : Union[None, float, Sequence[np.float64], NDArray[np.float64]], optional
            System parameters passed to the mapping function. Scalars and sequences will be converted automatically.
        eps : float, optional
            Size of the neighborhood for recurrence detection (default is 1e-2).
            Must be a positive number.
        transient_time : Optional[int], optional
            Initial iterations to discard (default is None, meaning no transient time).
            If provided, must be a non-negative integer.

        Returns
        -------
        NDArray[np.float64]
            Array of recurrence times (time steps between re-entries into the neighborhood). Returns an empty array if no recurrences occur.

        Raises
        ------
        TypeError
            - If `u` is not a scalar, or 1D array, or if its shape does not match the expected system dimension.
            - If `parameters` is not None and does not match the expected number of parameters.
            - If `parameters` is None but the system expects parameters.
            - If `parameters` is a scalar or array-like but not 1D.
            - If `total_time` is negative.
            - If `trasient_time` is negative.
            - If `transient_time` is greater than or equal to total_time.
            - If `eps` is not a positive float.
        TypeError
            - If `u` is not a scalar or array-like type.
            - If `parameters` is not a scalar or array-like type
            - If `total_time` is not int.
            - If `transient_time` is not int.
            - If `eps` is not float.


        Notes
        -----
        - This method wraps a JIT-compiled function for performance.
        - A recurrence is counted when the system state re-enters the axis-aligned hypercube:
            [u - eps/2, u + eps/2]^d
        - This is commonly used in nonlinear dynamics to study:
            - Stickiness
            - Poincaré recurrences
            - Mixing and ergodicity

        Examples
        --------
        >>> u0 = [0.1, 0.1]
        >>> parameters = [0.6, 0.4]
        >>> rec_times = system.recurrence_times(u0, parameters, 10000, eps=0.01)
        >>> print(rec_times)
        array([400, 523, 861, ...])
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

        validate_non_negative(eps, "eps", Real)

        return recurrence_times(
            u,
            parameters,
            total_time,
            self.__mapping,
            eps,
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
        u: Union[NDArray[np.float64], Sequence[float], float],
        total_time: int,
        parameters: Union[
            None, float, Sequence[np.float64], NDArray[np.float64]
        ] = None,
        wmin: int = 2,
        transient_time: Optional[int] = None,
    ) -> NDArray[np.float64]:
        """
        Estimate the Hurst exponent for a system trajectory using the rescaled range (R/S) method.

        Parameters
        ----------
        u : NDArray[np.float64]
            Initial condition vector of shape (n,).
        total_time : int
            Total number of iterations used to generate the trajectory.
        parameters : Union[None, float, Sequence[np.float64], NDArray[np.float64]], optional
            Parameters passed to the mapping function.
        wmin : int, optional
            Minimum window size for the rescaled range calculation. Default is 2.
        transient_time : Optional[int], optional
            Number of initial iterations to discard as transient. If `None`, no transient is removed. Default is `None`.

        Returns
        -------
        NDArray[np.float64]
            Estimated Hurst exponents for each dimension of the input vector `u`, of shape (n,).

        Raises
        ------
        ValueError
            - If `u` is not a 2D array, or if its shape does not match the expected system dimension.
            - If `parameters` is not None and does not match the expected number of parameters.
            - If `parameters` is None but the system expects parameters.
            - If `parameters` is a scalar or array-like but not 1D.
            - If `total_time` is negative or zero.
            - If `transient_time` is negative or greater than or equal to `total_time`.
            - If `wmin` is not a positive integer or is less than 2 or greater than total_time // 2.

        TypeError
            - If `u` is not a scalar or array-like type.
            - If `parameters` is not a scalar or array-like type.
            - If `total_time` is not int.
            - If `wmin` is not a positive integer.

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

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        else:
            parameters = validate_parameters(parameters, self.__number_of_parameters)

        validate_non_negative(total_time, "total_time", Integral)
        validate_transient_time(transient_time, total_time, Integral)

        validate_positive(wmin, "wmin", Integral)

        if (
            wmin < 2
            or wmin
            >= (total_time - (transient_time if transient_time is not None else 0)) // 2
        ):
            raise ValueError(
                f"`wmin` must be an integer >= 2 and <= total_time / 2. Got {wmin}."
            )

        result = hurst_exponent_wrapped(
            u,
            parameters,
            total_time,
            self.__mapping,
            wmin=wmin,
            transient_time=transient_time,
        )

        if self.__system_dimension == 1:
            return result[0]
        else:
            return result

    def finite_time_hurst_exponent(
        self,
        u: Union[NDArray[np.float64], Sequence[float], float],
        total_time: int,
        finite_time: int,
        parameters: Union[
            None, float, Sequence[np.float64], NDArray[np.float64]
        ] = None,
        wmin: int = 2,
        return_points: bool = False,
    ) -> Union[NDArray[np.float64], Tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """Compute finite-time Hurst exponent along a trajectory.

        Parameters
        ----------
        u : Union[NDArray[np.float64], Sequence[float]]
            Initial condition of shape (d,) where d is system dimension
        total_time : int
            Total simulation time steps (must be > finite_time
        finite_time : int
            Averaging window size in time steps
        parameters : Union[None, float, Sequence[np.float64], NDArray[np.float64]], optional
            System parameters of shape (p,) passed to mapping function
        wmin : int, optional
            Minimum window size for the rescaled range calculation (default 2)
        return_points : bool, optional
            If True, returns full evolution (default False)

        Returns
        -------
        Union[NDArray[np.float64], Tuple[NDArray[np.float64], NDArray[np.float64]]]
            - If return_points=False: Hurst exponent(scalar)
            - If return_points=True: Tuple of (Hurst history, final state) where Hurst history is 1D array of values

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
            - If `wmin` is not a positive integer or is less than 2 or greater than total_time // 2.

        TypeError
            - If `u` is not a scalar or array-like type.
            - If `parameters` is not a scalar or array-like type.
            - If `total_time` is not int.
            - If `finite_time` is not int.
            - If `wmin` is not a positive integer.
            - If `return_points` is not a boolean.

        Notes
        -----
        - Finite-time Hurst exponent measures local scaling behavior over finite intervals
        - For chaotic systems, FTHE → true exponents as finite_time → ∞
        - Results are more reliable when:
        - finite_time >> 1
        - (total_time - transient_time) // finite_time >> 1

        Examples
        --------
        >>> # Basic usage with defaults
        >>> u0 = np.array([0.1, 0.2])
        >>> params = np.array([0.5, 1.0])
        >>> fthe = system.finite_time_hurst_exponent(u0, 100000, 100, parameters=params)

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

        return finite_time_hurst_exponent(
            u,
            parameters,
            total_time,
            finite_time,
            self.__mapping,
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

    def __lagrangian_descriptors(
        self,
        u: Union[NDArray[np.float64], Sequence[float]],
        parameters: Union[float, Sequence[np.float64], NDArray[np.float64]],
        total_time: int = 10000,
        transient_time: Optional[int] = None,
    ) -> NDArray[np.float64]:
        """Compute Lagrangian Descriptors(LDs) for the dynamical system.

        Parameters
        ----------
        u: Union[NDArray[np.float64], Sequence[float]]
            Initial condition of shape(d,) where d is system dimension.
            Can be any sequence convertible to numpy array.
        parameters: Union[None, float, Sequence[np.float64], NDArray[np.float64]], optional
            System parameters of shape(p,) passed to mapping functions.
        total_time: int, optional
            Total number of iterations to compute(default 10000, must be > 0).
        transient_time: Optional[int], optional
            Number of initial iterations to discard(default None → no transient).

        Returns
        -------
        NDArray[np.float64]
            Array of shape(2,) containing:

            - [0]: Forward Lagrangian descriptor
            - [1]: Backward Lagrangian descriptor

        Raises
        ------
        NotImplementedError
            If mapping is not defined
            If backwards mapping is not defined for this system
        ValueError
            If initial condition has wrong dimension
            If parameters are invalid
            If time parameters are invalid
        TypeError
            If inputs cannot be converted to required types

        Notes
        -----
        - LDs reveal phase space structures and invariant manifolds
        - Higher values indicate stronger stretching in phase space
        - For meaningful results:
        - Use total_time >> 1 (typically ≥ 1000)
        - Ensure mapping and backwards_mapping are exact inverses
        - Transient period helps avoid initialization artifacts

        Examples
        --------
        >>>  # Basic usage
        >>> u0 = np.array([0.1, 0.2])
        >>> params = np.array([0.5, 1.0])
        >>> lds = system.compute_lagrangian_descriptors(u0, params)
        >>> forward_ld, backward_ld = lds

        >>>  # With transient period
        >>> lds = system.compute_lagrangian_descriptors(
        ...     u0, params, total_time=5000, transient_time=1000)
        """

        # Check if mapping function is defined
        if self.__mapping is None:
            raise RuntimeError("Mapping function must be provided")

        # Check if jacobian function is defined
        if self.__backwards_mapping is None:
            raise RuntimeError("Backwards mapping function must be provided")

        # Input validation
        try:
            u_arr = np.asarray(u, dtype=np.float64)
            if u_arr.ndim != 1:
                raise ValueError("Initial condition must be 1D array")
        except (TypeError, ValueError) as e:
            raise TypeError(
                "Initial condition must be convertible to 1D float array"
            ) from e

        if np.isscalar(parameters):
            parameters = np.array([parameters], dtype=np.float64)
        elif not isinstance(parameters, np.ndarray):
            parameters = np.asarray(parameters, dtype=np.float64)

        if len(u_arr) != self.__system_dimension:
            raise ValueError(
                f"Initial condition dimension {len(u_arr)} != system dimension {self.__system_dimension}"
            )

        if not isinstance(total_time, int) or total_time <= 0:
            raise ValueError("total_time must be positive integer")

        if transient_time is not None:
            if not isinstance(transient_time, int) or transient_time < 0:
                raise ValueError("transient_time must be non-negative integer")
            if transient_time >= total_time:
                raise ValueError("transient_time must be < total_time")

        # Call the compiled computation function
        return lagrangian_descriptors(
            u_arr,
            parameters,
            total_time,
            self.__mapping,
            self.__backwards_mapping,
            transient_time=transient_time,
        )

    def recurrence_matrix(
        self,
        u: Union[NDArray[np.float64], Sequence[float]],
        total_time: int,
        parameters: Union[
            None, float, Sequence[np.float64], NDArray[np.float64]
        ] = None,
        transient_time: Optional[int] = None,
        **kwargs: Any,
    ) -> NDArray[np.float64]:
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

        # Recurrence matrix calculation
        TSM = tsm(time_series)
        recmat = TSM.recurrence_matrix(
            threshold=float(config.threshold),
            metric=config.metric,
            std_metric=config.std_metric,
            threshold_std=config.threshold_std,
        )

        return recmat

    def recurrence_time_entropy(
        self,
        u: Union[NDArray[np.float64], Sequence[float]],
        total_time: int,
        parameters: Union[
            None, float, Sequence[np.float64], NDArray[np.float64]
        ] = None,
        transient_time: Optional[int] = None,
        **kwargs: Any,
    ):
        """Compute Recurrence Time Entropy(RTE) for dynamical system analysis.

        Parameters
        ----------
        u: Union[NDArray[np.float64], Sequence[float]]
            Initial condition of shape(d,) where d is system dimension
        total_time: int
            Number of iterations to simulate(must be > 100 for meaningful results)
        parameters: Union[None, float, Sequence[np.float64], NDArray[np.float64]], optional
            System parameters of shape(p,) passed to mapping function
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

        Notes
        -----
        - Higher RTE indicates more complex dynamics
        - For reliable results:
            - Use total_time > 1000
            - Typical threshold range: 0.01-0.3
            - Set min_recurrence_time = 2 to ignore single-point recurrences
        - Implementation follows[1]

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

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        else:
            parameters = validate_parameters(parameters, self.__number_of_parameters)

        validate_non_negative(total_time, "total_time", Integral)
        validate_transient_time(transient_time, total_time, Integral)

        return RTE(
            u,
            parameters,
            total_time,
            self.__mapping,
            transient_time=transient_time,
            **kwargs,
        )

    def finite_time_recurrence_time_entropy(
        self,
        u: Union[NDArray[np.float64], Sequence[float]],
        total_time: int,
        finite_time: int,
        parameters: Union[
            None, float, Sequence[np.float64], NDArray[np.float64]
        ] = None,
        return_points: bool = False,
        **kwargs: Any,
    ) -> Union[NDArray[np.float64], Tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """Compute the finite-time Recurrence Time Entropy(RTE) for dynamical system analysis.

        Parameters
        ----------
        u: Union[NDArray[np.float64], Sequence[float]]
            Initial condition of shape(d,) where d is system dimension
        total_time: int
            Number of iterations to simulate(must be > 100 for meaningful results)
        finite_time: int
            Averaging window size in time steps
        parameters: Union[None, float, Sequence[np.float64], NDArray[np.float64]], optional
            System parameters of shape(p,) passed to mapping function
        return_points: bool, default = False
            Whether to return the finite-time RTE phase space points
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
        NDArray[np.float64]

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

        Notes
        -----
        - Higher RTE indicates more complex dynamics
        - For reliable results:
            - Use total_time > 1000
            - Typical threshold range: 0.01-0.3
            - Set min_recurrence_time = 2 to ignore single-point recurrences
        - Implementation follows [1]

        References
        ----------
        [1] Sales et al., Chaos 33, 033140 (2023)

        Examples
        --------
        >>>  # Basic usage
        >>> ftrte = system.finite_time_recurrence_time_entropy(u0, params, 50000, 100)

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

        return finite_time_RTE(
            u,
            parameters,
            total_time,
            finite_time,
            self.__mapping,
            return_points=return_points,
            **kwargs,
        )
