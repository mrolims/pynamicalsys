# hamiltonian_systems.py

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

from pynamicalsys.hamiltonian_systems.models import (
    henon_heiles_grad_T,
    henon_heiles_grad_V,
    henon_heiles_hess_T,
    henon_heiles_hess_V,
)

from pynamicalsys.hamiltonian_systems.numerical_integrators import (
    velocity_verlet_2nd_step_tangent,
    yoshida_4th_step,
    velocity_verlet_2nd_step,
    yoshida_4th_step_tangent,
)

from pynamicalsys.hamiltonian_systems.validators import (
    validate_initial_conditions,
    validate_non_negative,
    validate_parameters,
)

from pynamicalsys.common.utils import qr, householder_qr

from pynamicalsys.hamiltonian_systems.trajectory_analysis import (
    generate_trajectory,
    ensemble_trajectories,
    generate_poincare_section,
    ensemble_poincare_section,
)

from pynamicalsys.hamiltonian_systems.chaotic_indicators import (
    lyapunov_spectrum,
    maximum_lyapunov_exponent,
    SALI,
    LDI,
    GALI,
)


class HamiltonianSystem:
    """
    Class for defining, simulating, and analyzing Hamiltonian dynamical systems.

    This class provides access to predefined Hamiltonian models (e.g., the
    Hénon-Heiles system) or allows the user to define a custom Hamiltonian
    system via gradient and optional Hessian functions. It supports multiple
    numerical symplectic integrators and a variety of trajectory and chaos
    analysis tools, such as Lyapunov exponents, SALI, LDI, and GALI.

    Examples
    --------
    >>> from pynamicalsys import HamiltonianSystem
    >>> hs = HamiltonianSystem(model="henon heiles")
    >>> hs.available_models()
    ['henon heiles']
    >>> hs.available_integrators()
    ['svy4', 'vv2']
    """

    __AVAILABLE_MODELS: Dict[str, Dict[str, Any]] = {
        "henon heiles": {
            "description": "two d.o.f. Hénon-Heiles Hamiltonian system",
            "has hessian": True,
            "grad_T": henon_heiles_grad_T,
            "grad_V": henon_heiles_grad_V,
            "hess_T": henon_heiles_hess_T,
            "hess_V": henon_heiles_hess_V,
            "degrees of freedom": 2,
            "number of parameters": 0,
            "parameters": [],
        },
    }

    __AVAILABLE_INTEGRATORS: Dict[str, Dict[str, Any]] = {
        "svy4": {
            "description": "4th order Yoshida method",
            "integrator": yoshida_4th_step,
            "tangent integrator": yoshida_4th_step_tangent,
        },
        "vv2": {
            "description": "2nd order velocity Verlet method",
            "integrator": velocity_verlet_2nd_step,
            "tangent integrator": velocity_verlet_2nd_step_tangent,
        },
    }

    def __init__(
        self,
        model: Optional[str] = None,
        grad_T: Optional[Callable] = None,
        grad_V: Optional[Callable] = None,
        hess_T: Optional[Callable] = None,
        hess_V: Optional[Callable] = None,
        degrees_of_freedom: Optional[int] = None,
        number_of_parameters: Optional[int] = None,
    ) -> None:
        if model is not None and (grad_T is not None or grad_V is not None):
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
            self.__grad_T = model_info["grad_T"]
            self.__grad_V = model_info["grad_V"]
            self.__hess_T = model_info["hess_T"]
            self.__hess_V = model_info["hess_V"]
            self.__degrees_of_freedom = model_info["degrees of freedom"]
            self.__number_of_parameters = model_info["number of parameters"]
        elif (
            grad_T is not None
            and grad_V is not None
            and degrees_of_freedom is not None
            and number_of_parameters is not None
        ):
            if not callable(grad_T) or not callable(grad_V):
                raise TypeError(
                    "The custom system (grad V and grad T) must be callable"
                )

            self.__grad_T = grad_T
            self.__grad_V = grad_V

            self.__hess_T = hess_T
            self.__hess_V = hess_V

            if (
                self.__hess_T is not None
                and self.__hess_V is not None
                and not callable(self.__hess_T)
                and not callable(self.__hess_V)
            ):
                raise TypeError("Custom Hessian functions must be callable")

            validate_non_negative(degrees_of_freedom, "degrees_of_freedom", Integral)
            validate_non_negative(
                number_of_parameters, "number_of_parameters", Integral
            )

            self.__degrees_of_freedom = degrees_of_freedom
            self.__number_of_parameters = number_of_parameters
        else:
            raise ValueError(
                "Must specify either a model name or custom system function (grad V and grad T) with its dimension and number of paramters."
            )

        self.__integrator = "svy4"
        self.__integrator_func = yoshida_4th_step
        self.__tangent_integrator_func = yoshida_4th_step_tangent
        self.__time_step = 1e-2

    @classmethod
    def available_models(cls) -> List[str]:
        """
        List the available predefined Hamiltonian models.

        Returns
        -------
        list of str
            Names of the supported models.
        """
        return list(cls.__AVAILABLE_MODELS.keys())

    @classmethod
    def available_integrators(cls) -> List[str]:
        """
        List the available predefined Hamiltonian models.

        Returns
        -------
        list of str
            Names of the supported models.
        """
        return list(cls.__AVAILABLE_INTEGRATORS.keys())

    @property
    def info(self) -> Dict[str, Any]:
        """
        Information dictionary for the selected model.

        Returns
        -------
        dict
            Dictionary containing metadata such as description, gradients,
            Hessians, degrees of freedom, and parameters.

        Raises
        ------
        ValueError
            If no predefined model was used to initialize the system.
        """

        if self.__model is None:
            raise ValueError(
                "The 'info' property is only available when a model is provided."
            )

        model = self.__model.lower()

        return self.__AVAILABLE_MODELS[model]

    @property
    def integrator_info(self) -> Dict[str, Any]:
        """
        Information dictionary for the current integrator.

        Returns
        -------
        dict
            Dictionary containing the integrator description and associated
            step functions.
        """
        integrator = self.__integrator.lower()

        return self.__AVAILABLE_INTEGRATORS[integrator]

    def integrator(self, integrator, time_step=1e-2) -> None:
        """
        Set the numerical integrator and time step.

        Parameters
        ----------
        integrator : str
            Name of the integrator. Options:
            - 'svy4' : 4th order Yoshida method
            - 'vv2'  : 2nd order velocity-Verlet
        time_step : float, optional
            Integration time step (default is 1e-2).

        Raises
        ------
        ValueError
            If `time_step` is negative or if `integrator` is not available.
        TypeError
            If `integrator` is not a string or `time_step` is not numeric.

        Examples
        --------
        >>> from pynamicalsys import HamiltonianSystem
        >>> HamiltonianSystem.available_integrators()
        ['svy4', 'vv2']
        >>> hs = HamiltonianSystem(model="henon heiles")
        >>> hs.integrator("svy4", time_step=0.001) #  To use the SVY4 integrator with a time step of 10^{-3}
        >>> hs.integrator("vv2", time_step=0.001) #  To use the VV2 integrator
        """

        if not isinstance(integrator, str):
            raise ValueError("integrator must be a string.")
        validate_non_negative(time_step, "time_step", type_=Real)

        if integrator in self.__AVAILABLE_INTEGRATORS:
            self.__integrator = integrator.lower()
            integrator_info = self.__AVAILABLE_INTEGRATORS[self.__integrator]
            self.__integrator_func = integrator_info["integrator"]
            self.__tangent_integrator_func = integrator_info["tangent integrator"]
            self.__time_step = time_step

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

    def step(
        self,
        q: NDArray[np.float64],
        p: NDArray[np.float64],
        parameters: Union[None, Sequence[float], NDArray[np.float64]] = None,
    ) -> NDArray[np.float64]:
        """
        Advance the system by one integration step.

        Parameters
        ----------
        q : ndarray
            Initial generalized coordinates (shape: (dof,) or (N, dof)).
        p : ndarray
            Initial generalized momenta (same shape as `q`).
        parameters : array-like, optional
            System parameters, if required.

        Returns
        -------
        q_new, p_new : tuple of ndarray
            Updated coordinates and momenta after one integration step.

        Raises
        ------
        ValueError
            If `q` and `p` have mismatched shapes.
        TypeError
            If inputs are not scalar or array-like.
        """
        q = validate_initial_conditions(q, self.__degrees_of_freedom)
        q = q.copy()
        p = validate_initial_conditions(p, self.__degrees_of_freedom)
        p = p.copy()

        if q.ndim != p.ndim:
            raise ValueError("q and p must have the same dimension and shape")

        parameters = validate_parameters(parameters, self.__number_of_parameters)

        if q.ndim == 1:
            q, p = self.__integrator_func(
                q, p, self.__time_step, self.__grad_T, self.__grad_V
            )
        else:
            num_ic = q.shape[0]
            for i in range(num_ic):
                q[i], p[i] = self.__integrator_func(
                    q[i], p[i], self.__time_step, self.__grad_T, self.__grad_V
                )

        return q, p

    def trajectory(
        self,
        q: NDArray[np.float64],
        p: NDArray[np.float64],
        total_time: np.float64,
        parameters: Union[None, Sequence[float], NDArray[np.float64]] = None,
    ) -> NDArray[np.float64]:
        """
        Generate a trajectory for the system.

        Parameters
        ----------
        q : ndarray
            Initial coordinates.
        p : ndarray
            Initial momenta.
        total_time : float
            Total integration time.
        parameters : array-like, optional
            System parameters.

        Returns
        -------
        trajectory : ndarray
            Trajectory data with shape depending on whether ensemble or single
            ICs are provided.

        Raises
        ------
        ValueError
            If `q` and `p` shapes mismatch or if `total_time` is negative.
        """
        q = validate_initial_conditions(q, self.__degrees_of_freedom)
        q = q.copy()

        p = validate_initial_conditions(p, self.__degrees_of_freedom)
        p = p.copy()

        if q.ndim != p.ndim:
            raise ValueError("q and p must have the same dimension and shape")

        parameters = validate_parameters(parameters, self.__number_of_parameters)

        validate_non_negative(total_time, "total time", type_=Real)

        if q.ndim == 1:
            trajectory_function = generate_trajectory
        else:
            trajectory_function = ensemble_trajectories

        return trajectory_function(
            q,
            p,
            total_time,
            parameters,
            self.__grad_T,
            self.__grad_V,
            self.__time_step,
            self.__integrator_func,
        )

    def poincare_section(
        self,
        q: NDArray[np.float64],
        p: NDArray[np.float64],
        total_time: np.float64,
        parameters: Union[None, Sequence[float], NDArray[np.float64]] = None,
        section_index: int = 0,
        section_value: float = 0.0,
        crossing: int = 1,
    ) -> NDArray[np.float64]:
        """
        Compute a Poincaré section of the trajectory.

        Parameters
        ----------
        q, p : ndarray
            Initial coordinates and momenta.
        total_time : float
            Total simulation time.
        parameters : array-like, optional
            System parameters.
        section_index : int, default=0
            Index of the phase space coordinate for the section.
        section_value : float, default=0.0
            Value at which the section is taken.
        crossing : {-1, 0, 1}, default=1
            Direction of crossing:
            -1 : downward
             0 : all crossings
             1 : upward

        Returns
        -------
        section_points : ndarray
            Points of the trajectory lying on the Poincaré section.

        Raises
        ------
        ValueError
            If shapes mismatch, if `section_index` is invalid,
            or if `crossing` not in {-1, 0, 1}.
        TypeError
            If `section_value` is not numeric.
        """
        q = validate_initial_conditions(q, self.__degrees_of_freedom)
        q = q.copy()

        p = validate_initial_conditions(p, self.__degrees_of_freedom)
        p = p.copy()

        if q.ndim != p.ndim:
            raise ValueError("q and p must have the same dimension and shape")

        parameters = validate_parameters(parameters, self.__number_of_parameters)

        validate_non_negative(total_time, "total time", type_=Real)

        validate_non_negative(section_index, "section_index")
        if section_index >= 2 * self.__degrees_of_freedom:
            raise ValueError(
                "section_index must be less or equal to the system dimension"
            )

        if not isinstance(section_value, Real):
            raise TypeError("section_value must be a valid number")

        if crossing not in [-1, 0, 1]:
            raise ValueError(
                "crossing must be either -1, 0, or 1, indicating downward, all crossings, and upward crossings, respectively"
            )

        if q.ndim == 1:
            poincare_section_function = generate_poincare_section
        else:
            poincare_section_function = ensemble_poincare_section

        return poincare_section_function(
            q,
            p,
            total_time,
            parameters,
            self.__grad_T,
            self.__grad_V,
            self.__time_step,
            self.__integrator_func,
            section_index,
            section_value,
            crossing,
        )

    def lyapunov(
        self,
        q: NDArray[np.float64],
        p: NDArray[np.float64],
        total_time: np.float64,
        parameters: Union[None, Sequence[float], NDArray[np.float64]] = None,
        num_exponents: Optional[int] = None,
        return_history: bool = False,
        seed: int = 13,
        log_base: np.float64 = np.e,
        qr_interval: int = 1,
        method: str = "QR",
    ) -> NDArray[np.float64]:
        """
        Compute Lyapunov exponents.

        Parameters
        ----------
        q, p : ndarray
            Initial conditions (single orbit only).
        total_time : float
            Total integration time.
        parameters : array-like, optional
            System parameters.
        num_exponents : int, optional
            Number of exponents to compute (default: full spectrum).
        return_history : bool, default=False
            If True, return time evolution instead of final values.
        seed : int, default=13
            Random seed for initial deviation vectors.
        log_base : float, default=e
            Logarithm base for exponent calculation.
        qr_interval : int, default=1
            Interval for reorthonormalization.
        method : {'QR', 'QR_HH'}, default='QR'
            QR decomposition method.

        Returns
        -------
        exponents : ndarray
            Computed Lyapunov exponents.

        Raises
        ------
        ValueError
            If Hessians are missing, if inputs are invalid, or if base=1.
        TypeError
            If types are inconsistent with expectations.
        """

        if self.__hess_T is None or self.__hess_V is None:
            raise ValueError(
                "Hessian functions are required to compute the Lyapunov exponents"
            )

        q = validate_initial_conditions(
            q, self.__degrees_of_freedom, allow_ensemble=False
        )
        q = q.copy()
        p = validate_initial_conditions(
            p, self.__degrees_of_freedom, allow_ensemble=False
        )
        p = p.copy()

        validate_non_negative(total_time, "total time", type_=Real)

        parameters = validate_parameters(parameters, self.__number_of_parameters)

        if num_exponents is None:
            num_exponents = 2 * self.__degrees_of_freedom
        elif num_exponents > 2 * self.__degrees_of_freedom or num_exponents < 1:
            raise ValueError("num_exponents must be <= system_dimension")
        else:
            validate_non_negative(num_exponents, "num_exponents", Integral)

        if not isinstance(return_history, bool):
            raise TypeError("return_history must be True or False")

        if not isinstance(seed, Integral):
            raise TypeError("seed must be an integer")

        validate_non_negative(log_base, "log_base", Real)
        if log_base == 1:
            raise ValueError("The logarithm function is not defined with base 1")

        validate_non_negative(qr_interval, "qr_interval", Integral)

        if not isinstance(method, str):
            raise TypeError("method must be a string")

        method = method.upper()
        if method == "QR":
            qr_func = qr
        elif method == "QR_HH":
            qr_func = householder_qr
        else:
            raise ValueError("method must be QR or QR_HH")

        if num_exponents > 1:
            result = lyapunov_spectrum(
                q,
                p,
                total_time,
                self.__time_step,
                parameters,
                self.__grad_T,
                self.__grad_V,
                self.__hess_T,
                self.__hess_V,
                num_exponents,
                qr_interval,
                return_history,
                seed,
                log_base,
                qr_func,
                self.__integrator_func,
                self.__tangent_integrator_func,
            )
        else:
            result = maximum_lyapunov_exponent(
                q,
                p,
                total_time,
                self.__time_step,
                parameters,
                self.__grad_T,
                self.__grad_V,
                self.__hess_T,
                self.__hess_V,
                return_history,
                seed,
                log_base,
                self.__integrator_func,
                self.__tangent_integrator_func,
            )

        if return_history:
            return np.array(result)
        else:
            return np.array(result[0])

    def SALI(
        self,
        q: NDArray[np.float64],
        p: NDArray[np.float64],
        total_time: float,
        parameters: Union[None, Sequence[float], NDArray[np.float64]] = None,
        return_history: bool = False,
        seed: int = 13,
        threshold: float = 1e-16,
    ) -> NDArray[np.float64]:
        """
        Compute the Smaller Alignment Index (SALI).

        SALI distinguishes between chaotic and regular motion by evolving two
        deviation vectors and monitoring their alignment over time. In chaotic
        systems, SALI tends exponentially to zero; in regular systems, it
        stabilizes to a nonzero value.

        Parameters
        ----------
        q, p : ndarray
            Initial coordinates and momenta (1D arrays).
        total_time : float
            Total integration time.
        parameters : array-like, optional
            System parameters.
        return_history : bool, default=False
            If True, return SALI evolution over time.
        seed : int, default=13
            Random seed for initializing deviation vectors.
        threshold : float, default=1e-8
            Early termination threshold for SALI. If SALI ≤ threshold,
            integration stops.

        Returns
        -------
        sali : ndarray
            - If `return_history=True`, array of shape (N, 2) with columns
              [time, SALI].
            - If `return_history=False`, array of shape (1, 2) with the final
              [time, SALI].

        Raises
        ------
        ValueError
            If Hessians are missing, if `total_time` is negative, or if
            `threshold` ≤ 0.
        TypeError
            If input types are invalid (e.g., non-numeric values).
        """
        if self.__hess_T is None or self.__hess_V is None:
            raise ValueError(
                "Hessian functions are required to compute the Lyapunov exponents"
            )

        q = validate_initial_conditions(
            q, self.__degrees_of_freedom, allow_ensemble=False
        )
        q = q.copy()
        p = validate_initial_conditions(
            p, self.__degrees_of_freedom, allow_ensemble=False
        )
        p = p.copy()

        validate_non_negative(total_time, "total time", type_=Real)

        parameters = validate_parameters(parameters, self.__number_of_parameters)

        if not isinstance(return_history, bool):
            raise TypeError("return_history must be True or False")

        if not isinstance(seed, Integral):
            raise TypeError("seed must be an integer")

        validate_non_negative(threshold, "threshold", Real)

        result = SALI(
            q,
            p,
            total_time,
            self.__time_step,
            parameters,
            self.__grad_T,
            self.__grad_V,
            self.__hess_T,
            self.__hess_V,
            return_history,
            seed,
            self.__integrator_func,
            self.__tangent_integrator_func,
            threshold,
        )

        if return_history:
            return np.array(result)
        else:
            return result[0]

    def LDI(
        self,
        q: NDArray[np.float64],
        p: NDArray[np.float64],
        total_time: float,
        k: int,
        parameters: Union[None, Sequence[float], NDArray[np.float64]] = None,
        return_history: bool = False,
        seed: int = 13,
        threshold: float = 1e-16,
    ) -> NDArray[np.float64]:
        """
        Compute the Linear Dependence Index (LDI).

        LDI measures the linear dependence among `k` deviation vectors evolved
        along a trajectory. It is computed from the product of singular values
        of the deviation matrix. In chaotic systems, LDI tends rapidly to zero;
        in regular systems, it remains bounded away from zero.

        Parameters
        ----------
        q, p : ndarray
            Initial coordinates and momenta (1D arrays).
        total_time : float
            Total integration time.
        parameters : array-like, optional
            System parameters.
        k : int, default=2
            Number of deviation vectors to evolve.
        return_history : bool, default=False
            If True, return LDI evolution over time.
        seed : int, default=13
            Random seed for initializing deviation vectors.
        threshold : float, default=1e-8
            Early termination threshold for LDI. If LDI ≤ threshold,
            integration stops.

        Returns
        -------
        ldi : ndarray
            - If `return_history=True`, array of shape (N, 2) with columns
              [time, LDI].
            - If `return_history=False`, array of shape (1, 2) with the final
              [time, LDI].

        Raises
        ------
        ValueError
            If Hessians are missing, if `k` ≤ 1, if `total_time` is negative,
            or if `threshold` ≤ 0.
        TypeError
            If input types are invalid (e.g., `k` not an integer).
        """
        if self.__hess_T is None or self.__hess_V is None:
            raise ValueError(
                "Hessian functions are required to compute the Lyapunov exponents"
            )

        q = validate_initial_conditions(
            q, self.__degrees_of_freedom, allow_ensemble=False
        )
        q = q.copy()
        p = validate_initial_conditions(
            p, self.__degrees_of_freedom, allow_ensemble=False
        )
        p = p.copy()

        validate_non_negative(total_time, "total time", type_=Real)

        parameters = validate_parameters(parameters, self.__number_of_parameters)

        validate_non_negative(k, "k", Integral)
        if k <= 1 or k > 2 * self.__degrees_of_freedom:
            raise ValueError("k must be 2 < k < system dimension")

        if not isinstance(return_history, bool):
            raise TypeError("return_history must be True or False")

        if not isinstance(seed, Integral):
            raise TypeError("seed must be an integer")

        validate_non_negative(threshold, "threshold", Real)

        result = LDI(
            q,
            p,
            total_time,
            self.__time_step,
            parameters,
            self.__grad_T,
            self.__grad_V,
            self.__hess_T,
            self.__hess_V,
            k,
            return_history,
            seed,
            self.__integrator_func,
            self.__tangent_integrator_func,
            threshold,
        )

        if return_history:
            return np.array(result)
        else:
            return result[0]

    def GALI(
        self,
        q: NDArray[np.float64],
        p: NDArray[np.float64],
        total_time: float,
        k: int,
        parameters: Union[None, Sequence[float], NDArray[np.float64]] = None,
        return_history: bool = False,
        seed: int = 13,
        threshold: float = 1e-16,
    ) -> NDArray[np.float64]:
        """
        Compute the Generalized Alignment Index (GALI).

        GALI extends SALI by considering the evolution of `k` deviation
        vectors. It is defined as the volume of the parallelepiped formed by
        the normalized deviation vectors (via the wedge product). In chaotic
        systems, GALI decays exponentially; in regular systems, it follows a
        power law or stabilizes.

        Parameters
        ----------
        q, p : ndarray
            Initial coordinates and momenta (1D arrays).
        total_time : float
            Total integration time.
        parameters : array-like, optional
            System parameters.
        k : int, default=2
            Number of deviation vectors to evolve.
        return_history : bool, default=False
            If True, return GALI evolution over time.
        seed : int, default=13
            Random seed for initializing deviation vectors.
        threshold : float, default=1e-8
            Early termination threshold for GALI. If GALI ≤ threshold,
            integration stops.

        Returns
        -------
        gali : ndarray
            - If `return_history=True`, array of shape (N, 2) with columns
              [time, GALI].
            - If `return_history=False`, array of shape (1, 2) with the final
              [time, GALI].

        Raises
        ------
        ValueError
            If Hessians are missing, if `k` ≤ 1, if `total_time` is negative,
            or if `threshold` ≤ 0.
        TypeError
            If input types are invalid (e.g., `k` not an integer).
        """
        if self.__hess_T is None or self.__hess_V is None:
            raise ValueError(
                "Hessian functions are required to compute the Lyapunov exponents"
            )

        q = validate_initial_conditions(
            q, self.__degrees_of_freedom, allow_ensemble=False
        )
        q = q.copy()
        p = validate_initial_conditions(
            p, self.__degrees_of_freedom, allow_ensemble=False
        )
        p = p.copy()

        validate_non_negative(total_time, "total time", type_=Real)

        parameters = validate_parameters(parameters, self.__number_of_parameters)

        validate_non_negative(k, "k", Integral)
        if k <= 1 or k > 2 * self.__degrees_of_freedom:
            raise ValueError("k must be 2 < k < system dimension")

        if not isinstance(return_history, bool):
            raise TypeError("return_history must be True or False")

        if not isinstance(seed, Integral):
            raise TypeError("seed must be an integer")

        validate_non_negative(threshold, "threshold", Real)

        result = GALI(
            q,
            p,
            total_time,
            self.__time_step,
            parameters,
            self.__grad_T,
            self.__grad_V,
            self.__hess_T,
            self.__hess_V,
            k,
            return_history,
            seed,
            self.__integrator_func,
            self.__tangent_integrator_func,
            threshold,
        )

        if return_history:
            return np.array(result)
        else:
            return result[0]
