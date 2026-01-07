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
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from pynamicalsys.common.utils import householder_qr, qr
from pynamicalsys.common.validators import (
    validate_clv_pairs,
    validate_non_negative,
    validate_parameters,
    validate_clv_subspaces,
)

from pynamicalsys.hamiltonian_systems.chaotic_indicators import (
    GALI,
    LDI,
    SALI,
    hurst_exponent_wrapped,
    lyapunov_spectrum,
    maximum_lyapunov_exponent,
    recurrence_time_entropy,
    compute_clvs,
    clv_angles,
)
from pynamicalsys.hamiltonian_systems.models import (
    henon_heiles_grad_T,
    henon_heiles_grad_V,
    henon_heiles_hess_T,
    henon_heiles_hess_V,
)
from pynamicalsys.hamiltonian_systems.numerical_integrators import (
    velocity_verlet_2nd_step,
    velocity_verlet_2nd_step_traj_tan,
    yoshida_4th_step,
    yoshida_4th_step_traj_tan,
)
from pynamicalsys.hamiltonian_systems.trajectory_analysis import (
    ensemble_poincare_section,
    ensemble_trajectories,
    generate_poincare_section,
    generate_trajectory,
)
from pynamicalsys.hamiltonian_systems.validators import (
    validate_initial_conditions,
)


class HamiltonianSystem:
    """
    Class for defining, simulating, and analyzing separable Hamiltonian dynamical systems.

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
            "traj tan integrator": yoshida_4th_step_traj_tan,
        },
        "vv2": {
            "description": "2nd order velocity Verlet method",
            "integrator": velocity_verlet_2nd_step,
            "traj tan integrator": velocity_verlet_2nd_step_traj_tan,
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
        parameters: Optional[Sequence] = None,
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
            self.__parameters = None
            self.__number_of_parameters = model_info["number of parameters"]
        elif (
            grad_T is not None
            and grad_V is not None
            and degrees_of_freedom is not None
            and (parameters is not None or number_of_parameters is not None)
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
            if number_of_parameters is not None:
                validate_non_negative(
                    number_of_parameters, "number_of_parameters", Integral
                )

            self.__degrees_of_freedom = degrees_of_freedom
            self.__parameters = parameters
            if self.__parameters is not None:
                self.__number_of_parameters = len(self.__parameters)
                self.__parameters = validate_parameters(
                    self.__parameters, self.__number_of_parameters
                )
            else:
                self.__number_of_parameters = number_of_parameters
        else:
            raise ValueError(
                "Must specify either a model name or custom system function (grad V and grad T) with its dimension and parameters or number of paramters."
            )

        self.__integrator = "svy4"
        self.__integrator_func = yoshida_4th_step
        self.__traj_tan_integrator_func = yoshida_4th_step_traj_tan
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
            self.__traj_tan_integrator_func = integrator_info["traj tan integrator"]
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
        q: NDArray[np.float64],
        p: NDArray[np.float64],
        parameters: Union[None, Sequence[float], NDArray[np.float64]] = None,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
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

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        else:
            parameters = validate_parameters(parameters, self.__number_of_parameters)

        if q.ndim == 1:
            q, p = self.__integrator_func(
                q, p, self.__time_step, self.__grad_T, self.__grad_V, parameters
            )
        else:
            num_ic = q.shape[0]
            for i in range(num_ic):
                q[i], p[i] = self.__integrator_func(
                    q[i],
                    p[i],
                    self.__time_step,
                    self.__grad_T,
                    self.__grad_V,
                    parameters,
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

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        else:
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
        num_intersections: int,
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

            - -1 : downward
            - 0 : all crossings
            - 1 : upward

        Returns
        -------
        section_points : ndarray
            Points of the trajectory lying on the Poincaré section. The first column is the time of each crossing.

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

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        else:
            parameters = validate_parameters(parameters, self.__number_of_parameters)

        validate_non_negative(
            num_intersections, "num_intersections time", type_=Integral
        )

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
            num_intersections,
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

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        else:
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
                self.__traj_tan_integrator_func,
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
                self.__traj_tan_integrator_func,
            )

        if return_history:
            return np.array(result)
        else:
            return np.array(result[0])

    def CLV(
        self,
        q,
        p,
        total_time,
        parameters=None,
        num_clvs=None,
        warmup_time=0.0,
        tail_time=0.0,
        qr_time_step=None,
        seed=13,
        poincare_section=False,
        section_index=None,
        section_value=None,
        crossing=None,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Compute Covariant Lyapunov Vectors (CLVs) along a trajectory of this continuous-time
        Hamiltonian system.

        The CLVs form a covariant (time-dependent) basis of the tangent space that transforms
        under the linearized flow exactly as the dynamics does. The i-th CLV is associated with
        the i-th Lyapunov exponent (ordered from largest to smallest) and provides the local
        expanding/contracting directions in phase space.

        This routine implements a Ginelli-style algorithm adapted to continuous-time flows
        and symplectic integration:
        (i) a forward QR iteration to obtain the orthonormal backward Lyapunov vectors (BLVs),
        followed by (ii) a backward recursion in the triangular coefficient space to recover
        covariant vectors.

        Parameters
        ----------
        q, p : array_like, shape (dof,)
            Initial generalized coordinates and momenta.
        total_time : float
            Total integration time over which CLVs are returned. The output contains
            ``T = floor(total_time / qr_time_step) + 1`` sampling points (after warm-up),
            corresponding to QR checkpoint times.
        parameters : None or array_like
            Model parameters. If ``None``, this method uses the system's stored parameters.
        num_clvs : int, optional
            Number of CLVs to compute/return. Defaults to ``2*dof``.
            Must satisfy ``1 <= num_clvs <= 2*dof``.
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
            Time interval between QR re-orthonormalizations and storage of tangent-space data.
            Must satisfy ``qr_time_step >= time_step``. If ``None``, defaults to ``time_step``.
        seed : int, default=13
            Seed used to initialize the random upper-triangular matrix in the backward stage.
            The final CLVs should be insensitive to the seed if ``tail_time`` (and the window)
            are large enough.
        poincare_section : bool, default=False
            If True, return the trajectory restricted to a Poincaré section defined by
            ``section_index``, ``section_value``, and ``crossing``.
        section_index : int, optional
            Index of the coordinate defining the Poincaré section.
        section_value : float, optional
            Value of the section coordinate.
        crossing : int, optional
            Crossing rule:
            ``+1`` upward crossings, ``-1`` downward crossings, ``0`` both.

        Returns
        -------
        clvs : ndarray, shape (T, 2*dof, num_clvs)
            Covariant Lyapunov vectors sampled at QR checkpoint times. At each time index ``t``,
            ``clvs[t, :, i]`` is the i-th CLV (column vector). Columns are normalized to unit
            Euclidean norm.
        traj : ndarray
            Trajectory sampled at the same times as ``clvs``. If ``poincare_section=False``,
            the array has shape ``(T, 2*dof + 1)`` with columns ``[t, q..., p...]``.
            If ``poincare_section=True``, the trajectory is restricted to the Poincaré section
            crossings.

        Raises
        ------
        ValueError
            If ``total_time`` is negative.
            If ``num_clvs`` is greater than ``2*dof``.
            If ``warmup_time`` is negative.
            If ``tail_time`` is negative.
            If ``qr_time_step`` is smaller than the integrator time step.
            If Poincaré section parameters are inconsistent.
        TypeError
            If ``q`` or ``p`` cannot be interpreted as valid initial conditions.
            If ``parameters`` cannot be interpreted as valid system parameters.

        Notes
        -----
        Step-by-step algorithm (Ginelli-style, continuous time)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        Let ``Φ^t`` be the Hamiltonian flow and ``DΦ^t`` its tangent map. Let
        ``p = num_clvs`` and ``d = 2*dof``.

        1. Forward QR warm-up (optional)\\
        Initialize an orthonormal matrix ``Q`` of size ``d × p``. Integrate the system
        forward in time together with the tangent vectors and periodically perform QR
        decompositions at intervals ``qr_time_step``:
            ``Z = DΦ^{Δt} @ Q``\\
            ``Z = Q⁺ R``\\
        replacing ``Q ← Q⁺`` for a duration ``warmup_time``. This drives ``Q`` toward
        the backward Lyapunov vectors (BLVs).

        2. Forward storage window\\
        Continue the integration for ``total_time``, storing at each QR checkpoint the
        orthonormal matrices ``Q_k`` and the associated upper-triangular matrices ``R_k``.

        3. Tail / backward initialization\\
        Perform ``tail_time`` additional forward QR steps beyond the storage window and
        use the resulting triangular matrices to drive an arbitrary upper-triangular
        matrix ``A`` toward its asymptotic limit. This removes dependence on the random
        seed in the backward recursion.

        4. Backward recursion and CLV reconstruction\\
        Starting from the end of the storage window and moving backward:
            ``A_{k-1} = R_{k-1}^{-1} A_k``\\
        and reconstruct CLVs as:
            ``V_k = Q_k @ A_k``\\
        Columns are normalized after reconstruction.

        Choosing ``warmup_time`` and ``tail_time``
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        The role of ``warmup_time`` and ``tail_time`` is identical to the discrete-time case,
        but measured in *physical time* rather than iterations. Weak chaos, near-integrable
        regimes, or near-degeneracies in the Lyapunov spectrum typically require much larger
        values to obtain well-conditioned CLVs.

        Limitations
        -----------
        - This method is defined only for systems with at least one degree of freedom
        (phase-space dimension >= 2).
        - CLVs are sampled only at QR checkpoint times, not at every integrator step.

        References
        ----------
        F. Ginelli et al., Characterizing dynamics with covariant Lyapunov
        vectors. Phys. Rev. Lett. 99, 130601 (2007).
        """
        q = validate_initial_conditions(
            q, self.__degrees_of_freedom, allow_ensemble=False
        )
        q = q.copy()
        p = validate_initial_conditions(
            p, self.__degrees_of_freedom, allow_ensemble=False
        )
        p = p.copy()

        validate_non_negative(total_time, "total time", type_=Real)
        validate_non_negative(warmup_time, "warmup time", type_=Real)
        validate_non_negative(tail_time, "tail time", type_=Real)

        if qr_time_step is not None:
            validate_non_negative(qr_time_step, "qr time step", type_=Real)
            if qr_time_step < self.__time_step:
                raise ValueError(
                    f"qr_time_step must be >= time_step. Got {qr_time_step}"
                )
        else:
            qr_time_step = self.__time_step

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        else:
            parameters = validate_parameters(parameters, self.__number_of_parameters)

        if num_clvs is None:
            num_clvs = 2 * self.__degrees_of_freedom
        elif num_clvs > 2 * self.__degrees_of_freedom:
            raise ValueError(f"num_clvs must be <= 2 * dof. Got {num_clvs}")
        else:
            validate_non_negative(num_clvs, "num_clvs", type_=Integral)

        if not isinstance(poincare_section, bool):
            raise TypeError(
                f"poincare_section must be of type bool. Got {poincare_section} with type {type(poincare_section)}"
            )

        if poincare_section:
            if section_index is None or section_value is None or crossing is None:
                raise ValueError(
                    "When poincare_section=True, you must also inform section_index, section_value, AND crossing"
                )
            else:
                validate_non_negative(section_index, "section_index", type_=Integral)
                if section_index > 2 * self.__degrees_of_freedom:
                    raise ValueError(
                        f"section_index must be <= 2 * dof. Got {section_index}"
                    )
                if crossing not in [-1, 0, 1]:
                    raise ValueError(
                        f"crossing must be -1 (downward crossing), 0 (both directions), or 1 (upward crossing). Got {crossing}"
                    )

        return compute_clvs(
            q,
            p,
            total_time,
            self.__time_step,
            parameters,
            self.__grad_T,
            self.__grad_V,
            self.__hess_T,
            self.__hess_V,
            num_clvs,
            warmup_time,
            tail_time,
            qr_time_step,
            seed,
            qr,
            self.__traj_tan_integrator_func,
            poincare_section,
            section_index,
            section_value,
            crossing,
        )

    def CLV_angles(
        self,
        q: Union[NDArray[np.float64], Sequence[np.float64]],
        p: Union[NDArray[np.float64], Sequence[np.float64]],
        total_time: float,
        parameters: Union[
            None, float, NDArray[np.float64], Sequence[np.float64]
        ] = None,
        subspaces: Optional[Sequence[Tuple[Sequence[int], Sequence[int]]]] = None,
        pairs: Optional[Sequence[Tuple[int, int]]] = None,
        warmup_time: float = 0.0,
        tail_time: float = 0.0,
        qr_time_step: Optional[float] = None,
        seed: int = 13,
        poincare_section: bool = False,
        section_index: int = 0,
        section_value: float = 0.0,
        crossing: int = 1,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Compute angle diagnostics derived from Covariant Lyapunov Vectors (CLVs) for this
        continuous-time Hamiltonian system.

        This method computes CLVs along a trajectory (see
        :meth:`CLV`) and returns time series of angles between
        user-selected CLV subspaces and/or between individual CLV pairs. These angles
        quantify the local geometric relations between invariant directions and are
        commonly used to study hyperbolicity, near-tangencies, and weak chaos.

        At least one of ``subspaces`` or ``pairs`` must be provided.

        Parameters
        ----------
        q, p : array_like, shape (dof,)
            Initial generalized coordinates and momenta.
        total_time : float
            Total integration time used for the angle diagnostics.
        parameters : None or array_like, optional
            Model parameters. If ``None``, this method uses the system's stored parameters.
        subspaces : sequence of (sequence of int, sequence of int), optional
            List of subspace pairs for which minimum principal angles are computed.
            Each element ``(A, B)`` defines two subspaces
            ``E_A(t) = span{v_i(t) : i in A}`` and
            ``E_B(t) = span{v_j(t) : j in B}``.
        pairs : sequence of (int, int), optional
            List of CLV index pairs ``(i, j)`` for which pairwise angles
            ``angle(v_i(t), v_j(t))`` are computed.
        warmup_time : float, default=0
            Forward QR warm-up duration passed to the CLV computation. See
            :meth:`CLV`.
        tail_time : float, default=0
            Backward-recursion convergence duration passed to the CLV computation. See
            :meth:`CLV`.
        qr_time_step : float, optional
            QR checkpoint interval passed to the CLV computation.
        seed : int, default=13
            Seed forwarded to the CLV computation.
        poincare_section : bool, default=False
            If True, the returned trajectory corresponds to a Poincaré section.
        section_index : int, default=0
            Index of the coordinate defining the section.
        section_value : float, default=0.0
            Value of the section coordinate.
        crossing : int, default=1
            Crossing rule for the Poincaré section.

        Returns
        -------
        angles : ndarray, shape (T, M)
            Angle time series in radians. Columns are ordered as:
            1) angles for all requested CLV pairs (in the order given).
            2) followed by angles for all requested subspace pairs (in the order given),
        traj : ndarray
            Trajectory sampled at the same times as ``angles`` (QR checkpoints), or the
            corresponding Poincaré section crossings if ``poincare_section=True``.

        Raises
        ------
        ValueError
            If neither ``subspaces`` nor ``pairs`` is provided.
            If any subspace or pair specification is invalid.
            If ``total_time``, ``warmup_time``, or ``tail_time`` is negative.
        TypeError
            If ``q`` or ``p`` cannot be interpreted as valid initial conditions.
            If ``parameters`` cannot be interpreted as valid system parameters.

        Notes
        -----
        Interpretation of CLV angles
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        - **Pairwise angles** measure the instantaneous alignment between two specific
        covariant directions.
        - **Subspace angles** (minimum principal angles) measure the closest approach
        between two invariant subspaces and provide the natural generalization of the
        2D CLV angle to higher-dimensional flows.

        Near-zero subspace angles signal near-tangencies between invariant bundles and
        are the most informative diagnostic for weak hyperbolicity and mixed dynamics.

        Limitations
        -----------
        - Angles are computed only at QR checkpoint times.
        - In the presence of strong degeneracies in the Lyapunov spectrum, CLVs and their
        angles may fluctuate strongly and require long averaging windows.

        See Also
        --------
        covariant_lyapunov_vectors : Computes CLVs for continuous-time Hamiltonian systems.
        """
        q = validate_initial_conditions(
            q, self.__degrees_of_freedom, allow_ensemble=False
        )
        q = q.copy()
        p = validate_initial_conditions(
            p, self.__degrees_of_freedom, allow_ensemble=False
        )
        p = p.copy()

        validate_non_negative(total_time, "total time", type_=Real)
        validate_non_negative(warmup_time, "warmup time", type_=Real)
        validate_non_negative(tail_time, "tail time", type_=Real)

        if qr_time_step is not None:
            validate_non_negative(qr_time_step, "qr time step", type_=Real)
            if qr_time_step < self.__time_step:
                raise ValueError(
                    f"qr_time_step must be >= time_step. Got {qr_time_step}"
                )
        else:
            qr_time_step = self.__time_step

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        else:
            parameters = validate_parameters(parameters, self.__number_of_parameters)

        pairs = validate_clv_pairs(pairs, 2 * self.__degrees_of_freedom)

        subspaces = validate_clv_subspaces(subspaces, 2 * self.__degrees_of_freedom)

        if not isinstance(poincare_section, bool):
            raise TypeError(
                f"poincare_section must be of type bool. Got {poincare_section} with type {type(poincare_section)}"
            )

        if poincare_section:
            if section_index is None or section_value is None or crossing is None:
                raise ValueError(
                    "When poincare_section=True, you must also inform section_index, section_value, AND crossing"
                )
            else:
                validate_non_negative(section_index, "section_index", type_=Integral)
                if section_index > 2 * self.__degrees_of_freedom:
                    raise ValueError(
                        f"section_index must be <= 2 * dof. Got {section_index}"
                    )
                if crossing not in [-1, 0, 1]:
                    raise ValueError(
                        f"crossing must be -1 (downward crossing), 0 (both directions), or 1 (upward crossing). Got {crossing}"
                    )

        return clv_angles(
            q,
            p,
            total_time,
            self.__time_step,
            parameters,
            self.__grad_T,
            self.__grad_V,
            self.__hess_T,
            self.__hess_V,
            warmup_time,
            tail_time,
            qr_time_step,
            seed,
            qr,
            self.__traj_tan_integrator_func,
            poincare_section,
            section_index,
            section_value,
            crossing,
            subspaces,
            pairs,
        )

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

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        else:
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
            self.__traj_tan_integrator_func,
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

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        else:
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
            self.__traj_tan_integrator_func,
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

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        else:
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
            self.__traj_tan_integrator_func,
            threshold,
        )

        if return_history:
            return np.array(result)
        else:
            return result[0]

    def recurrence_time_entropy(
        self,
        q: Union[NDArray[np.float64], Sequence[float]],
        p: Union[NDArray[np.float64], Sequence[float]],
        num_intersections: int,
        parameters: Union[None, Sequence[float], NDArray[np.float64]] = None,
        section_index: int = 0,
        section_value: float = 0.0,
        crossing: int = 1,
        **kwargs,
    ):
        """
        Compute the recurrence time entropy (RTE) for a Hamiltonian system.

        Parameters
        ----------
        q, p : Union[NDArray[np.float64], Sequence[float]]
            Initial coordinates and momenta (1D arrays).
        num_intersections: int
            Number of intersections to record in the Poincaré section.
        parameters : array-like, optional
            System parameters.
        section_index : Optional[int]
            Index of the coordinate to define the Poincaré section (0-based). Only used when map_type="PS".
        section_value : Optional[float]
            Value of the coordinate at which the section is defined. Only used when map_type="PS".
        crossing : Optional[int]
            Specifies the type of crossing to consider:
            - 1 : positive crossing (from below to above section_value)
            - -1 : negative crossing (from above to below section_value)
            - 0 : all crossings
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
            - If `q` or `p` are not a 1D array matching the number of degrees of freedom.
            - If `parameters` is not `None` and does not match the expected number of parameters.
            - If `parameters` is `None` but the system expects parameters.
            - If `parameters` is a scalar or array-like but not 1D.
            - If `section_index` is negative or ≥ system dimension.
            - If `crossing` is not one of {-1, 0, 1}.
        TypeError
            - If `q` or `p` are not a scalar or array-like type.
            - If `parameters` is not a scalar or array-like type.
            - If `section_value` is not a real.
            - If `crossing` is not an integer.
            - If `sampling_time` is not a real number.

        Notes
        -----
        - Higher RTE indicates more complex dynamics
        - Set min_recurrence_time = 2 to ignore single-point recurrences
        - Implementation follows [1]

        References
        ----------
        [1] Sales et al., Chaos 33, 033140 (2023)
        """
        q = validate_initial_conditions(
            q, self.__degrees_of_freedom, allow_ensemble=False
        )
        q = q.copy()
        p = validate_initial_conditions(
            p, self.__degrees_of_freedom, allow_ensemble=False
        )
        p = p.copy()

        validate_non_negative(num_intersections, "num_intersections", type_=Real)

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        else:
            parameters = validate_parameters(parameters, self.__number_of_parameters)

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

        return recurrence_time_entropy(
            q,
            p,
            num_intersections,
            parameters,
            self.__grad_T,
            self.__grad_V,
            self.__time_step,
            self.__integrator_func,
            section_index,
            section_value,
            crossing,
            **kwargs,
        )

    def hurst_exponent(
        self,
        q: Union[NDArray[np.float64], Sequence[float]],
        p: Union[NDArray[np.float64], Sequence[float]],
        num_intersections: int,
        parameters: Union[None, Sequence[float], NDArray[np.float64]] = None,
        wmin: int = 2,
        section_index: int = 0,
        section_value: float = 0.0,
        crossing: int = 1,
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
        wmin : int, optional
            Minimum window size for the rescaled range calculation. Default is 2.
        section_index : Optional[int]
            Index of the coordinate to define the Poincaré section (0-based). Only used when map_type="PS".
        section_value : Optional[float]
            Value of the coordinate at which the section is defined. Only used when map_type="PS".
        crossing : Optional[int]
            Specifies the type of crossing to consider:
            - 1 : positive crossing (from below to above section_value)
            - -1 : negative crossing (from above to below section_value)
            - 0 : all crossings

        Returns
        -------
        NDArray[np.float64]
            Estimated Hurst exponents for each dimension of the system (2 * dof).

        Raises
        ------
        TypeError
            - If `map_type` is not a string.
            - If `section_value` is not a real number.
            - If `crossing` is not an integer.
        ValueError
            - If `section_index` is negative or ≥ system dimension.
            - If `crossing` is not in {-1, 0, 1}.
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
        q = validate_initial_conditions(
            q, self.__degrees_of_freedom, allow_ensemble=False
        )
        q = q.copy()
        p = validate_initial_conditions(
            p, self.__degrees_of_freedom, allow_ensemble=False
        )
        p = p.copy()

        validate_non_negative(num_intersections, "num_intersections", type_=Real)

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        else:
            parameters = validate_parameters(parameters, self.__number_of_parameters)

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

        if wmin < 2 or wmin >= num_intersections // 2:
            raise ValueError(
                f"`wmin` must be an integer >= 2 and <= total_time / 2. Got {wmin}."
            )

        return hurst_exponent_wrapped(
            q,
            p,
            num_intersections,
            parameters,
            self.__grad_T,
            self.__grad_V,
            self.__time_step,
            self.__integrator_func,
            section_index,
            section_value,
            crossing,
            wmin,
        )
