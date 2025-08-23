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

from numbers import Integral, Real
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
from numpy.typing import NDArray

from pynamicalsys.common.utils import householder_qr, qr
from pynamicalsys.continuous_time.chaotic_indicators import (
    LDI,
    SALI,
    GALI,
    lyapunov_exponents,
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
)
from pynamicalsys.continuous_time.numerical_integrators import (
    estimate_initial_step,
    rk4_step_wrapped,
    rk45_step_wrapped,
)
from pynamicalsys.continuous_time.trajectory_analysis import (
    ensemble_trajectories,
    evolve_system,
    generate_trajectory,
)
from pynamicalsys.continuous_time.validators import (
    validate_initial_conditions,
    validate_non_negative,
    validate_parameters,
    validate_times,
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
            "has_jacobian": True,
            "has_variational_equations": True,
            "equations_of_motion": lorenz_system,
            "jacobian": lorenz_jacobian,
            "dimension": 3,
            "number_of_parameters": 3,
            "parameters": ["sigma", "rho", "beta"],
        },
        "henon heiles": {
            "description": "Two d.o.f. Hénon-Heiles system",
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
            "has_jacobian": True,
            "has_variational_equations": True,
            "equations_of_motion": rossler_system_4D,
            "jacobian": rossler_system_4D_jacobian,
            "dimension": 4,
            "number_of_parameters": 4,
            "parameters": ["a", "b", "c", "d"],
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
        number_of_parameters: Optional[int] = None,
    ) -> None:

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
            self.__number_of_parameters = model_info["number_of_parameters"]

            if jacobian is not None:
                self.__jacobian = jacobian

        elif (
            equations_of_motion is not None
            and system_dimension is not None
            and number_of_parameters is not None
        ):
            self.__equations_of_motion = equations_of_motion
            self.__jacobian = jacobian

            validate_non_negative(system_dimension, "system_dimension", Integral)
            validate_non_negative(
                number_of_parameters, "number_of_parameters", Integral
            )

            self.__system_dimension = system_dimension
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
        u: NDArray[np.float64],
        total_time: float,
        parameters: Union[None, Sequence[float], NDArray[np.float64]] = None,
    ) -> NDArray[np.float64]:
        """
        Evolve the dynamical system from the given initial conditions over a specified time period.

        Parameters
        ----------
        u : NDArray[np.float64]
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
        u: NDArray[np.float64],
        total_time: float,
        parameters: Union[None, Sequence[float], NDArray[np.float64]] = None,
        transient_time: Optional[float] = None,
    ) -> NDArray[np.float64]:
        """
        Compute the trajectory of the dynamical system over a specified time period.

        Parameters
        ----------
        u : NDArray[np.float64]
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
        >>> u = [0.1, 0.1, 0.1]  # Initial condition
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

    def lyapunov(
        self,
        u: NDArray[np.float64],
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
        u : NDArray[np.float64]
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
            log_base=log_base,
            QR=qr_func,
        )
        if return_history:
            return np.array(result)
        else:
            return np.array(result[0])

    def SALI(
        self,
        u: NDArray[np.float64],
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
        u : NDArray[np.float64]
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
        u: NDArray[np.float64],
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
        u : NDArray[np.float64]
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
        u: NDArray[np.float64],
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
        u : NDArray[np.float64]
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
