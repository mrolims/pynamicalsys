# hamiltonian_systems.py

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

from numbers import Integral, Real
from typing import Any, Dict, List, Sequence, Union

import numpy as np
from numpy.typing import NDArray


from pynamicalsys.common.types import (
    int_t,
    numeric_t,
    numeric_like_t,
    system_func_t,
    symplectic_step_t,
    symplectic_tangent_step_t,
)

from pynamicalsys.hamiltonian_systems.models import (
    henon_heiles_grad_T,
    henon_heiles_grad_V,
    henon_heiles_hess_T,
    henon_heiles_hess_V,
    henon_heiles_eom,
    henon_heiles_hess_H,
)

from pynamicalsys.hamiltonian_systems.fixed_step import (
    implicit_midpoint_step,
    velocity_verlet_2nd_step,
    yoshida_4th_step,
)

from pynamicalsys.hamiltonian_systems.tangent import (
    implicit_midpoint_step_traj_tan,
    velocity_verlet_2nd_step_traj_tan,
    yoshida_4th_step_traj_tan,
)

from pynamicalsys.hamiltonian_systems.trajectory import (
    generate_trajectory,
    ensemble_trajectories,
)

from pynamicalsys.hamiltonian_systems.poincare import (
    ensemble_poincare_section,
    generate_poincare_section,
)


from pynamicalsys.hamiltonian_systems.lyapunov import (
    lyapunov_spectrum_sep,
    lyapunov_spectrum_imp,
    largest_lyapunov_exponent_sep,
    largest_lyapunov_exponent_imp,
)

from pynamicalsys.hamiltonian_systems.clv import (
    clv_angles_imp,
    clv_angles_sep,
    compute_clvs_imp,
    compute_clvs_sep,
)

from pynamicalsys.hamiltonian_systems.sali import sali_imp, sali_sep

from pynamicalsys.hamiltonian_systems.ldi import ldi_k_imp, ldi_k_sep

from pynamicalsys.hamiltonian_systems.gali import gali_k_imp, gali_k_sep

from pynamicalsys.hamiltonian_systems.rte import (
    recurrence_time_entropy as recurrence_time_entropy_core,
)

from pynamicalsys.hamiltonian_systems.hurst import hurst_exponent_wrapped

from pynamicalsys.common.validators import (
    validate_clv_pairs,
    validate_non_negative,
    validate_positive,
    validate_parameters,
    validate_clv_subspaces,
)

from pynamicalsys.hamiltonian_systems.validators import (
    validate_initial_conditions,
)


class HamiltonianSystem:
    """
    Class for defining, integrating, and analyzing Hamiltonian systems.

    This class represents Hamiltonian systems H(q, p), where `q` denotes
    the generalized coordinates and `p` the conjugate momenta. Two ways
    of specifying the system are supported:

    - **Separable** systems, H(q, p) = T(p) + V(q), specified via gradient
      functions for the kinetic and potential energies (`grad_T`, `grad_V`),
      with optional Hessians (`hess_T`, `hess_V`) for tangent-space
      computations. These are integrated with explicit symplectic methods
      (velocity Verlet, fourth-order Yoshida).
    - **General (possibly non-separable)** systems H(q, p), specified via
      the full equations of motion (`eom`) zdot = f(z), with z = (q, p) and
      the Hessian of H with respect to the combined state z (`hess_H`).
      These are integrated with the implicit midpoint method.

    A system can be created either from one of the built-in models or from
    user-supplied functions of either kind above.

    The class provides symplectic fixed-step integration routines together with
    tools for trajectory generation and nonlinear-dynamics analysis, including
    Poincaré sections, Lyapunov exponents, covariant Lyapunov vectors (CLVs),
    SALI, LDI, GALI, recurrence time entropy, and Hurst exponent estimation.

    Parameters
    ----------
    model : str or None, optional
        Name of a built-in Hamiltonian model.
    grad_T : callable or None, optional
        Gradient of the kinetic energy with respect to the momenta.
        Required (with `grad_V`) for separable systems.
    grad_V : callable or None, optional
        Gradient of the potential energy with respect to the coordinates.
        Required (with `grad_T`) for separable systems.
    hess_T : callable or None, optional
        Hessian of the kinetic energy with respect to the momenta.
    hess_V : callable or None, optional
        Hessian of the potential energy with respect to the coordinates.
    eom : callable or None, optional
        Equations of motion of the system, with signature
        ``eom(q, p, parameters) -> (qdot, pdot)``, where
        ``qdot = dH/dp`` and ``pdot = -dH/dq``. The return value must be a
        tuple in this exact order, `(qdot, pdot)`, not `(pdot, qdot)`.
        Required (with `hess_H`) for general, possibly non-separable
        systems integrated with the implicit midpoint method.
    hess_H : callable or None, optional
        Full Hessian of the Hamiltonian H with respect to the combined
        state z = (q, p), with signature ``hess_H(q, p, parameters)``
        returning an array of shape `(2 * degrees_of_freedom, 2 *
        degrees_of_freedom)`. Required (with `eom`) for general, possibly
        non-separable systems integrated with the implicit midpoint method.
    degrees_of_freedom : int or None, optional
        Number of degrees of freedom of the custom system.
    parameters : array_like or None, optional
        Parameter vector for the system.
    number_of_parameters : int or None, optional
        Number of parameters expected by the custom system.

    Notes
    -----
    - Custom systems must be specified either as a separable system via
      `grad_T`/`grad_V` (with optional `hess_T`/`hess_V`), or as a general
      system via `eom`/`hess_H`, together with `degrees_of_freedom`.
    - Lyapunov exponents, CLVs, SALI, LDI, and GALI require Hessian
      functions (`hess_T`/`hess_V` for separable systems).
    - Ensemble trajectory and reduced-map methods accept multiple initial
      conditions when supported by the corresponding wrapper.

    See Also
    --------
    ContinuousDynamicalSystem : Class for general continuous-time dynamical systems.
    DiscreteDynamicalSystem : Class for discrete-time maps.
    """

    __AVAILABLE_MODELS: Dict[str, Dict[str, Any]] = {
        "henon heiles": {
            "description": "two d.o.f. Hénon-Heiles Hamiltonian system",
            "grad_T": henon_heiles_grad_T,
            "grad_V": henon_heiles_grad_V,
            "hess_T": henon_heiles_hess_T,
            "hess_V": henon_heiles_hess_V,
            "eom": henon_heiles_eom,
            "hess_H": henon_heiles_hess_H,
            "degrees of freedom": 2,
            "number of parameters": 0,
            "parameters": [],
        },
    }

    __AVAILABLE_INTEGRATORS: Dict[str, Dict[str, Any]] = {
        "svy4": {
            "description": "4th order Yoshida method",
            "integrator": yoshida_4th_step,
            "tangent integrator": yoshida_4th_step_traj_tan,
        },
        "vv2": {
            "description": "2nd order velocity Verlet method",
            "integrator": velocity_verlet_2nd_step,
            "tangent integrator": velocity_verlet_2nd_step_traj_tan,
        },
        "imp": {
            "description": "Implicit midpoint method",
            "integrator": implicit_midpoint_step,
            "tangent integrator": implicit_midpoint_step_traj_tan,
        },
    }

    def __init__(
        self,
        model: str | None = None,
        grad_T: system_func_t | None = None,
        grad_V: system_func_t | None = None,
        hess_T: system_func_t | None = None,
        hess_V: system_func_t | None = None,
        eom: system_func_t | None = None,
        hess_H: system_func_t | None = None,
        degrees_of_freedom: int | None = None,
        parameters: numeric_like_t | None = None,
        number_of_parameters: int | None = None,
    ) -> None:
        self.__model: str
        self.__grad_T: system_func_t | None
        self.__grad_V: system_func_t | None
        self.__hess_T: system_func_t | None
        self.__hess_V: system_func_t | None
        self.__eom: system_func_t | None
        self.__hess_H: system_func_t | None
        self.__system_func_1: system_func_t  # Either grad_T or eom
        self.__system_func_2: system_func_t  # Either grad_V or hess_H
        self.__system_func_3: system_func_t  # Only hess_T
        self.__system_func_4: system_func_t  # Only hess_V
        self.__degrees_of_freedom: int
        self.__parameters: NDArray[np.float64] | None
        self.__number_of_parameters: int
        self.__integrator: str
        self.__integrator_func: symplectic_step_t
        self.__traj_tan_integrator_func: symplectic_tangent_step_t
        self.__time_step: np.float64
        self.__tol: np.float64
        self.__max_iter: int

        if model is not None and (
            (grad_T is not None or grad_V is not None)
            or (eom is not None or hess_H is not None)
        ):
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
            self.__system_func_1 = model_info["grad_T"]
            self.__system_func_2 = model_info["grad_V"]
            self.__hess_T = model_info["hess_T"]
            self.__hess_V = model_info["hess_V"]
            self.__system_func_3 = model_info["hess_T"]
            self.__system_func_4 = model_info["hess_V"]
            self.__eom = model_info["eom"]
            self.__hess_H = model_info["hess_H"]
            self.__integrator = "svy4"
            self.__integrator_func = yoshida_4th_step
            self.__traj_tan_integrator_func = yoshida_4th_step_traj_tan
            self.__degrees_of_freedom = int(model_info["degrees of freedom"])
            self.__parameters = None
            self.__number_of_parameters = int(model_info["number of parameters"])

        elif degrees_of_freedom is not None:
            if (grad_T is not None and not callable(grad_T)) or (
                grad_V is not None and not callable(grad_V)
            ):
                raise TypeError("Custom grad_T and grad_V must be callable")
            self.__grad_T = grad_T
            self.__grad_V = grad_V

            if hess_T is not None and not callable(hess_T):
                raise TypeError("Custom hess_T must be callable or None")
            self.__hess_T = hess_T

            if hess_V is not None and not callable(hess_V):
                raise TypeError("Custom hess_V must be callable or None")
            self.__hess_V = hess_V

            if (eom is not None and not callable(eom)) or (
                hess_H is not None and not callable(hess_H)
            ):
                raise TypeError("Custom eom and hess_H must be callable")
            self.__eom = eom
            self.__hess_H = hess_H

            validate_positive(degrees_of_freedom, "degrees_of_freedom", Integral)

            if number_of_parameters is not None:
                validate_non_negative(
                    number_of_parameters, "number_of_parameters", Integral
                )

            validated_parameters: NDArray[np.float64] | None
            validated_number_of_parameters: int

            if parameters is not None:
                if isinstance(parameters, (int, float, np.integer, np.floating)):
                    validated_number_of_parameters = 1
                else:
                    parameters_arr = np.asarray(parameters, dtype=np.float64)
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
                    parameters, validated_number_of_parameters
                )
            else:
                if number_of_parameters is None:
                    raise ValueError(
                        "For a custom Hamiltonian system, you must provide either parameters or number_of_parameters."
                    )

                validated_number_of_parameters = int(number_of_parameters)
                validated_parameters = None

            self.__model = "custom"
            self.__degrees_of_freedom = int(degrees_of_freedom)
            self.__parameters = validated_parameters
            self.__number_of_parameters = validated_number_of_parameters

        else:
            raise ValueError(
                "Must specify either a model name or a custom Hamiltonian system with grad_T and grad_V or eom and hess_H, degrees_of_freedom, and parameters or number_of_parameters."
            )

        if grad_T is not None and grad_V is not None:
            self.__system_func_1 = grad_T
            self.__system_func_2 = grad_V
            if hess_T is not None and hess_V is not None:
                self.__system_func_3 = hess_T
                self.__system_func_4 = hess_V
            self.__integrator = "svy4"
            self.__integrator_func = yoshida_4th_step
            self.__traj_tan_integrator_func = yoshida_4th_step_traj_tan
        elif eom is not None and hess_H is not None:
            self.__system_func_1 = eom
            self.__system_func_2 = hess_H
            self.__integrator = "imp"
            self.__integrator_func = implicit_midpoint_step
            self.__traj_tan_integrator_func = implicit_midpoint_step_traj_tan
        self.__time_step = np.float64(1e-2)
        self.__tol = np.float64(1e-12)
        self.__max_iter = 50

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

    # def __get_pss_func(self):
    #
    #     if self.__integrator in ["vv2", "svy4"]:
    #         return generate_poincare_section_sep
    #     return generate_poincare_section_midpoint

    def integrator(
        self,
        integrator: str,
        time_step: numeric_t = np.float64(1e-2),
        tol: numeric_t = np.float64(1e-12),
        max_iter: int = 50,
    ) -> None:
        """
        Set the symplectic integrator and integration time step.

        Parameters
        ----------
        integrator : str
            Name of the integrator. Available options are:
            - `'svy4'`: 4th-order Yoshida method
            - `'vv2'`: 2nd-order velocity-Verlet method
        time_step : numeric_t, optional
            Integration time step. Must be a positive real number.
        tol : numeric_t, optional
            Newton convergence tolerance on the residual norm.
        max_iter : int, optional
            Maximum Newton iterations per step.

        Raises
        ------
        TypeError
            If `integrator` is not a string.
            If `time_step`, `tol`, or `max_iter` are not real numbers.
        ValueError
            If `time_step`, `tol`, or `max_iter` are not positive.
            If `integrator` is not implemented.

        Examples
        --------
        >>> from pynamicalsys import HamiltonianSystem
        >>> HamiltonianSystem.available_integrators()
        ['svy4', 'vv2', 'imp']
        >>> ds = HamiltonianSystem(model="henon heiles")
        >>> ds.integrator("svy4", time_step=0.001)
        >>> ds.integrator("vv2", time_step=0.001)
        >>> ds.integrator("imp", time_step=0.01, tol=1e-14, max_iter=100)
        """
        if not isinstance(integrator, str):
            raise TypeError("integrator must be a string")

        validate_positive(time_step, "time_step", Real)
        validate_positive(tol, "tol", Real)
        validate_positive(max_iter, "max_iter", Integral)

        integrator = integrator.lower()

        if integrator not in self.__AVAILABLE_INTEGRATORS:
            available = "\n".join(
                f"- {name}: {info['description']}"
                for name, info in self.__AVAILABLE_INTEGRATORS.items()
            )
            raise ValueError(
                f"Integrator '{integrator}' not implemented. Available integrators:\n{available}"
            )

        integrator_info = self.__AVAILABLE_INTEGRATORS[integrator]

        if integrator in {"svy4", "vv2"}:
            if self.__grad_T is None or self.__grad_V is None:
                raise ValueError(
                    "Cannot set integrator to `vv2` or `svy4` without providing grad_T and grad_V"
                )
            self.__system_func_1 = self.__grad_T
            self.__system_func_2 = self.__grad_V
            if self.__hess_T is not None and self.__hess_V is not None:
                self.__system_func_3 = self.__hess_T
                self.__system_func_4 = self.__hess_V

        elif integrator == "imp":
            if self.__eom is None or self.__hess_H is None:
                raise ValueError(
                    "Cannot set integrator to `imp` without providing eom and hess_H"
                )
            self.__system_func_1 = self.__eom
            self.__system_func_2 = self.__hess_H
        else:
            raise ValueError(f"Unknown integrator: {integrator}")

        self.__integrator = integrator
        self.__integrator_func = integrator_info["integrator"]
        self.__traj_tan_integrator_func = integrator_info["tangent integrator"]
        self.__time_step = np.float64(time_step)
        self.__tol = np.float64(tol)
        self.__max_iter = max_iter

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
        q: numeric_like_t,
        p: numeric_like_t,
        parameters: numeric_like_t | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Advance the Hamiltonian system by one integration step.

        Parameters
        ----------
        q : numeric_like_t
            Generalized coordinates. Must define either a 1D array of shape
            `(dof,)` or a 2D array of shape `(num_ic, dof)`.
        p : numeric_like_t
            Generalized momenta. Must have the same shape as `q`.
        parameters : numeric_like_t | None, optional
            System parameters. If `None`, the parameters stored in the instance
            are used.

        Returns
        -------
        tuple[NDArray[np.float64], NDArray[np.float64]]
            Updated coordinates and momenta after one integration step.

            - If `q` and `p` are 1D, returns `(q_new, p_new)` with shape `(dof,)`.
            - If `q` and `p` are 2D, returns `(q_new, p_new)` with shape
              `(num_ic, dof)`.

        Raises
        ------
        ValueError
            If `q` and `p` do not have the same shape.
            If the number of parameters does not match the expected number.
        TypeError
            If `q` or `p` cannot be interpreted as valid initial conditions.
        """
        q = validate_initial_conditions(q, self.__degrees_of_freedom)
        p = validate_initial_conditions(p, self.__degrees_of_freedom)

        if q.shape != p.shape:
            raise ValueError("q and p must have the same shape")

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        else:
            parameters = validate_parameters(parameters, self.__number_of_parameters)

        if q.ndim == 1:
            q_new, p_new = self.__integrator_func(
                q,
                p,
                self.__time_step,
                self.__system_func_1,
                self.__system_func_2,
                parameters,
                self.__tol,
                self.__max_iter,
            )
            return q_new, p_new

        q_new = q.copy()
        p_new = p.copy()

        for i in range(q.shape[0]):
            q_new[i], p_new[i] = self.__integrator_func(
                q[i],
                p[i],
                self.__time_step,
                self.__system_func_1,
                self.__system_func_2,
                parameters,
                self.__tol,
                self.__max_iter,
            )

        return q_new, p_new

    def trajectory(
        self,
        q: numeric_like_t,
        p: numeric_like_t,
        total_time: numeric_t,
        parameters: numeric_like_t | None = None,
    ) -> NDArray[np.float64]:
        """
        Generate a trajectory for the Hamiltonian system.

        Parameters
        ----------
        q : numeric_like_t
            Initial generalized coordinates. Must define either a 1D array of
            shape `(dof,)` or a 2D array of shape `(num_ic, dof)`.
        p : numeric_like_t
            Initial generalized momenta. Must have the same shape as `q`.
        total_time : numeric_t
            Total integration time.
        parameters : numeric_like_t | None, optional
            System parameters. If `None`, the parameters stored in the instance
            are used.

        Returns
        -------
        NDArray[np.float64]
            Trajectory data.

            - If `q` and `p` are 1D, returns an array of shape
              `(num_steps + 1, 2 * dof + 1)`, where the first column is time,
              the next `dof` columns are the coordinates, and the final `dof`
              columns are the momenta.
            - If `q` and `p` are 2D, returns an array of shape
              `(num_ic, num_steps + 1, 2 * dof + 1)`.

        Raises
        ------
        ValueError
            If `q` and `p` do not have the same shape.
            If `total_time` is not positive.
            If the number of parameters does not match the expected number.
        TypeError
            If `total_time` is not a real number.
            If `q` or `p` cannot be interpreted as valid initial conditions.
        """
        q = validate_initial_conditions(q, self.__degrees_of_freedom)
        p = validate_initial_conditions(p, self.__degrees_of_freedom)

        if q.shape != p.shape:
            raise ValueError("q and p must have the same shape")

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        else:
            parameters = validate_parameters(parameters, self.__number_of_parameters)

        validate_positive(total_time, "total_time", Real)
        total_time = np.float64(total_time)

        if q.ndim == 1:
            return generate_trajectory(
                q=q,
                p=p,
                total_time=total_time,
                parameters=parameters,
                system_func_1=self.__system_func_1,
                system_func_2=self.__system_func_2,
                time_step=self.__time_step,
                integrator=self.__integrator_func,
                tol=self.__tol,
                max_iter=self.__max_iter,
            )

        return ensemble_trajectories(
            q=q,
            p=p,
            total_time=total_time,
            parameters=parameters,
            system_func_1=self.__system_func_1,
            system_func_2=self.__system_func_2,
            time_step=self.__time_step,
            integrator=self.__integrator_func,
            tol=self.__tol,
            max_iter=self.__max_iter,
        )

    def poincare_section(
        self,
        q: numeric_like_t,
        p: numeric_like_t,
        num_intersections: int_t,
        parameters: numeric_like_t | None = None,
        section_index: int_t = 0,
        section_value: numeric_t = 0.0,
        crossing: int_t = 1,
        periodic_section_coordinate: bool = False,
        period: numeric_t = np.float64(2.0 * np.pi),
    ) -> NDArray[np.float64]:
        """
        Compute a Poincaré section of the Hamiltonian trajectory.

        Parameters
        ----------
        q : numeric_like_t
            Initial generalized coordinates. Must define either a 1D array of
            shape `(dof,)` or a 2D array of shape `(num_ic, dof)`.
        p : numeric_like_t
            Initial generalized momenta. Must have the same shape as `q`.
        num_intersections : int_t
            Number of section crossings to record.
        parameters : numeric_like_t | None, optional
            System parameters. If `None`, the parameters stored in the instance
            are used.
        section_index : int_t, optional
            Index of the coordinate used to define the section.
        section_value : numeric_t, optional
            Value of the selected coordinate at which the section is taken.
        crossing : int_t, optional
            Crossing rule:
            - `-1` for downward crossings
            - `0` for all crossings
            - `1` for upward crossings
        periodic_section_coordinate : bool, optional
            If True, treats q[section_index] as a periodic coordinate on S¹ and
            performs crossing detection using modulo arithmetic.
            If False, uses standard Euclidean crossing detection.
        period : np.float64, optional
            Period of the angular coordinate when
            `periodic_section_coordinate=True`.
            Typically 2π for action-angle systems.

        Returns
        -------
        NDArray[np.float64]
            Poincaré-section points.

            - If `q` and `p` are 1D, returns an array of shape
              `(num_intersections, 2 * dof + 1)`, where the first column is the
              crossing time, the next `dof` columns are the coordinates, and the
              final `dof` columns are the momenta.
            - If `q` and `p` are 2D, returns an array of shape
              `(num_ic, num_intersections, 2 * dof + 1)`.

        Raises
        ------
        ValueError
            If `q` and `p` do not have the same shape.
            If `num_intersections` is negative.
            If `section_index` is outside `[0, dof)`.
            If `crossing` is not one of `-1`, `0`, or `1`.
            If the number of parameters does not match the expected number.
        TypeError
            If `num_intersections` or `section_index` is not an integer.
            If `section_value` is not a valid real number.
        """
        q = validate_initial_conditions(q, self.__degrees_of_freedom)
        p = validate_initial_conditions(p, self.__degrees_of_freedom)

        if q.shape != p.shape:
            raise ValueError("q and p must have the same shape")

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        else:
            parameters = validate_parameters(parameters, self.__number_of_parameters)

        validate_non_negative(num_intersections, "num_intersections", Integral)
        validate_non_negative(section_index, "section_index", Integral)

        if section_index >= self.__degrees_of_freedom:
            raise ValueError(
                "section_index must be in the range [0, degrees_of_freedom)"
            )

        if isinstance(section_value, bool) or not isinstance(section_value, Real):
            raise TypeError("section_value must be a valid real number")
        section_value = np.float64(section_value)

        if isinstance(crossing, bool) or not isinstance(crossing, Integral):
            raise TypeError("crossing must be an integer")
        if crossing not in (-1, 0, 1):
            raise ValueError("crossing must be -1, 0, or 1")

        # if self.__integrator in ["svy4", "vv2"]:
        #     generate_poincare_section = generate_poincare_section_sep
        #     ensemble_poincare_section = ensemble_poincare_section_sep
        # else:
        #     generate_poincare_section = generate_poincare_section_midpoint
        #     ensemble_poincare_section = ensemble_poincare_section_midpoint

        if q.ndim == 1:
            return generate_poincare_section(
                q=q,
                p=p,
                num_intersections=np.int64(num_intersections),
                parameters=parameters,
                system_func_1=self.__system_func_1,
                system_func_2=self.__system_func_2,
                time_step=self.__time_step,
                integrator=self.__integrator_func,
                section_index=int(section_index),
                section_value=section_value,
                crossing=int(crossing),
                periodic_section_coordinate=periodic_section_coordinate,
                period=np.float64(period),
                tol=self.__tol,
                max_iter=self.__max_iter,
            )

        return ensemble_poincare_section(
            q=q,
            p=p,
            num_intersections=np.int64(num_intersections),
            parameters=parameters,
            system_func_1=self.__system_func_1,
            system_func_2=self.__system_func_2,
            time_step=self.__time_step,
            integrator=self.__integrator_func,
            section_index=int(section_index),
            section_value=section_value,
            crossing=int(crossing),
            periodic_section_coordinate=periodic_section_coordinate,
            period=np.float64(period),
            tol=self.__tol,
            max_iter=self.__max_iter,
        )

    def lyapunov(
        self,
        q: numeric_like_t,
        p: numeric_like_t,
        total_time: numeric_t,
        parameters: numeric_like_t | None = None,
        num_exponents: int_t | None = None,
        return_history: bool = False,
        seed: int_t = 1312,
        log_base: numeric_t = np.e,
        qr_interval: int_t = 1,
        method: str = "QR",
    ) -> NDArray[np.float64] | np.float64:
        """
        Compute Lyapunov exponents for a Hamiltonian system.

        Parameters
        ----------
        q : numeric_like_t
            Initial generalized coordinates. Must define a 1D array of shape
            `(dof,)`.
        p : numeric_like_t
            Initial generalized momenta. Must define a 1D array of shape
            `(dof,)` and have the same shape as `q`.
        total_time : numeric_t
            Total integration time.
        parameters : numeric_like_t | None, optional
            System parameters. If `None`, the parameters stored in the instance
            are used.
        num_exponents : int_t | None, optional
            Number of Lyapunov exponents to compute. If `None`, the full
            spectrum of size `2 * degrees_of_freedom` is computed.
        return_history : bool, optional
            If `True`, return the time evolution of the exponents.
        seed : int_t, optional
            Random seed used to initialize the deviation vectors.
        log_base : numeric_t, optional
            Base of the logarithm used in the exponent calculation.
        qr_interval : int_t, optional
            Number of integration steps between successive QR
            reorthonormalizations.
        method : str, optional
            QR decomposition method:
            - `"QR"`: internal reduced modified Gram-Schmidt QR
            - `"QR_HH"`: `numpy.linalg.qr` based on Householder reflections

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
            If Hessian functions are missing.
            If `q` and `p` do not have the same shape.
            If `total_time` is not positive.
            If `num_exponents` is not in
            `[1, 2 * degrees_of_freedom]`.
            If `log_base == 1`.
            If `method` is not `"QR"` or `"QR_HH"`.
        TypeError
            If `method` is not a string.
            If `seed`, `qr_interval`, or `num_exponents` are not integers.
            If `total_time` or `log_base` is not a valid real number.
        """
        if self.__integrator in ["vv2", "svy4"] and (
            self.__hess_T is None or self.__hess_V is None
        ):
            raise ValueError(
                "Hessian functions are required to compute the Lyapunov exponents when using either the velocity-Verlet or the Yoshida integrators"
            )

        q = validate_initial_conditions(
            q, self.__degrees_of_freedom, allow_ensemble=False
        )
        p = validate_initial_conditions(
            p, self.__degrees_of_freedom, allow_ensemble=False
        )

        if q.shape != p.shape:
            raise ValueError("q and p must have the same shape")

        validate_positive(total_time, "total_time", Real)
        total_time = np.float64(total_time)

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        else:
            parameters = validate_parameters(parameters, self.__number_of_parameters)

        if num_exponents is None:
            num_exponents = 2 * self.__degrees_of_freedom
        else:
            validate_positive(num_exponents, "num_exponents", Integral)
            if num_exponents > 2 * self.__degrees_of_freedom:
                raise ValueError("num_exponents must be <= 2 * degrees_of_freedom")

        if not isinstance(return_history, bool):
            raise TypeError("return_history must be a boolean")

        validate_non_negative(seed, "seed", Integral)
        validate_positive(qr_interval, "qr_interval", Integral)

        validate_non_negative(log_base, "log_base", Real)
        if log_base == 1:
            raise ValueError("The logarithm function is not defined with base 1")
        log_base = np.float64(log_base)

        if not isinstance(method, str):
            raise TypeError("method must be a string")

        method = method.upper()
        if method not in ("QR", "QR_HH"):
            raise ValueError("method must be 'QR' or 'QR_HH'")

        if self.__integrator in ["vv2", "svy4"]:
            if num_exponents > 1:
                result = lyapunov_spectrum_sep(
                    q=q,
                    p=p,
                    total_time=total_time,
                    time_step=self.__time_step,
                    parameters=parameters,
                    grad_T=self.__system_func_1,
                    grad_V=self.__system_func_2,
                    hess_T=self.__system_func_3,
                    hess_V=self.__system_func_4,
                    num_exponents=int(num_exponents),
                    qr_interval=int(qr_interval),
                    return_history=return_history,
                    seed=int(seed),
                    log_base=log_base,
                    method=method,
                    integrator_traj_tan=self.__traj_tan_integrator_func,
                )
            else:
                result = largest_lyapunov_exponent_sep(
                    q=q,
                    p=p,
                    total_time=total_time,
                    time_step=self.__time_step,
                    parameters=parameters,
                    grad_T=self.__system_func_1,
                    grad_V=self.__system_func_2,
                    hess_T=self.__system_func_3,
                    hess_V=self.__system_func_4,
                    return_history=return_history,
                    seed=int(seed),
                    log_base=log_base,
                    integrator_traj_tan=self.__traj_tan_integrator_func,
                )

        else:
            if num_exponents > 1:
                result = lyapunov_spectrum_imp(
                    q=q,
                    p=p,
                    total_time=total_time,
                    time_step=self.__time_step,
                    parameters=parameters,
                    eom=self.__system_func_1,
                    hess_H=self.__system_func_2,
                    num_exponents=int(num_exponents),
                    qr_interval=int(qr_interval),
                    return_history=return_history,
                    seed=int(seed),
                    log_base=log_base,
                    method=method,
                    tol=self.__tol,
                    max_iter=self.__max_iter,
                    integrator_traj_tan=self.__traj_tan_integrator_func,
                )
            else:
                result = largest_lyapunov_exponent_imp(
                    q=q,
                    p=p,
                    total_time=total_time,
                    time_step=self.__time_step,
                    parameters=parameters,
                    eom=self.__system_func_1,
                    hess_H=self.__system_func_2,
                    return_history=return_history,
                    seed=int(seed),
                    log_base=log_base,
                    tol=self.__tol,
                    max_iter=self.__max_iter,
                    integrator_traj_tan=self.__traj_tan_integrator_func,
                )

        if return_history:
            return result

        if num_exponents == 1:
            return np.float64(result[0, 0])

        return result[0]

    def CLV(
        self,
        q: numeric_like_t,
        p: numeric_like_t,
        total_time: numeric_t,
        parameters: numeric_like_t | None = None,
        num_clvs: int_t | None = None,
        warmup_time: numeric_t = 0.0,
        tail_time: numeric_t = 0.0,
        qr_time_step: numeric_t | None = None,
        seed: int_t = 1312,
        poincare_section: bool = False,
        section_index: int_t | None = None,
        section_value: numeric_t | None = None,
        crossing: int_t | None = None,
        method: str = "QR",
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Compute covariant Lyapunov vectors (CLVs) for a Hamiltonian system.

        Parameters
        ----------
        q : numeric_like_t
            Initial generalized coordinates. Must define a 1D array of shape
            `(dof,)`.
        p : numeric_like_t
            Initial generalized momenta. Must define a 1D array of shape
            `(dof,)` and have the same shape as `q`.
        total_time : numeric_t
            Total integration time over which CLVs are computed.
        parameters : numeric_like_t | None, optional
            System parameters. If `None`, the parameters stored in the instance
            are used.
        num_clvs : int_t | None, optional
            Number of CLVs to compute. If `None`, computes the full set of size
            `2 * degrees_of_freedom`.
        warmup_time : numeric_t, optional
            Forward warmup time used to drive the orthonormal tangent basis
            toward the backward Lyapunov vectors.
        tail_time : numeric_t, optional
            Additional forward integration time after the storage window, used
            to initialize the backward recursion.
        qr_time_step : numeric_t | None, optional
            Time interval between successive QR factorizations. If `None`,
            defaults to the integrator time step.
        seed : int_t, optional
            Random seed used in the tangent-basis initialization and backward
            recursion.
        poincare_section : bool, optional
            If `True`, return the sampled trajectory restricted to a Poincaré
            section.
        section_index : int_t | None, optional
            Index of the coordinate defining the Poincaré section.
        section_value : numeric_t | None, optional
            Value of the section coordinate.
        crossing : int_t | None, optional
            Crossing rule:
            - `-1` for downward crossings
            - `0` for all crossings
            - `1` for upward crossings
        method : str, optional
            QR decomposition method:
            - `"QR"`: internal reduced modified Gram-Schmidt QR
            - `"QR_HH"`: `numpy.linalg.qr` based on Householder reflections

        Returns
        -------
        tuple[NDArray[np.float64], NDArray[np.float64]]
            - `clvs`: array of shape `(T, 2 * dof, num_clvs)`
            - `traj`: sampled trajectory array with columns `[t, q..., p...]`

        Raises
        ------
        ValueError
            If `q` and `p` do not have the same shape.
            If `total_time` is not positive.
            If `warmup_time` or `tail_time` is negative.
            If `num_clvs` is not in `[1, 2 * degrees_of_freedom]`.
            If `qr_time_step` is smaller than the integration time step.
            If Poincaré-section parameters are inconsistent.
            If `method` is not `"QR"` or `"QR_HH"`.
        TypeError
            If `poincare_section` is not boolean.
            If `method` is not a string.
            If time-like arguments are not valid real numbers.
        """
        if self.__hess_T is None or self.__hess_V is None:
            raise ValueError("Hessian functions are required to compute CLVs")

        q = validate_initial_conditions(
            q, self.__degrees_of_freedom, allow_ensemble=False
        )
        p = validate_initial_conditions(
            p, self.__degrees_of_freedom, allow_ensemble=False
        )

        if q.shape != p.shape:
            raise ValueError("q and p must have the same shape")

        validate_positive(total_time, "total_time", Real)
        validate_non_negative(warmup_time, "warmup_time", Real)
        validate_non_negative(tail_time, "tail_time", Real)
        validate_non_negative(seed, "seed", Integral)

        total_time = np.float64(total_time)
        warmup_time = np.float64(warmup_time)
        tail_time = np.float64(tail_time)

        if qr_time_step is None:
            qr_time_step = self.__time_step
        else:
            validate_positive(qr_time_step, "qr_time_step", Real)
            qr_time_step = np.float64(qr_time_step)
            if qr_time_step < self.__time_step:
                raise ValueError(
                    f"qr_time_step must be >= time_step. Got {qr_time_step}"
                )

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        else:
            parameters = validate_parameters(parameters, self.__number_of_parameters)

        if num_clvs is None:
            num_clvs = 2 * self.__degrees_of_freedom
        else:
            validate_positive(num_clvs, "num_clvs", Integral)
            if num_clvs > 2 * self.__degrees_of_freedom:
                raise ValueError(
                    f"num_clvs must be <= 2 * degrees_of_freedom. Got {num_clvs}"
                )

        if not isinstance(poincare_section, bool):
            raise TypeError("poincare_section must be a boolean")

        if not isinstance(method, str):
            raise TypeError("method must be a string")

        method = method.upper()
        if method not in ("QR", "QR_HH"):
            raise ValueError("method must be 'QR' or 'QR_HH'")

        if poincare_section:
            if section_index is None or section_value is None or crossing is None:
                raise ValueError(
                    "When poincare_section=True, section_index, section_value, and crossing must all be provided"
                )

            validate_non_negative(section_index, "section_index", Integral)
            if section_index >= self.__degrees_of_freedom:
                raise ValueError(
                    "section_index must be in the range [0, degrees_of_freedom)"
                )

            if isinstance(section_value, bool) or not isinstance(section_value, Real):
                raise TypeError("section_value must be a valid real number")
            section_value = np.float64(section_value)

            if isinstance(crossing, bool) or not isinstance(crossing, Integral):
                raise TypeError("crossing must be an integer")
            if crossing not in (-1, 0, 1):
                raise ValueError("crossing must be -1, 0, or 1")
        else:
            section_index = 0
            section_value = np.float64(0.0)
            crossing = 1

        if self.__integrator in ["vv2", "svy4"]:
            return compute_clvs_sep(
                q=q,
                p=p,
                total_time=total_time,
                time_step=self.__time_step,
                parameters=parameters,
                grad_T=self.__system_func_1,
                grad_V=self.__system_func_2,
                hess_T=self.__system_func_3,
                hess_V=self.__system_func_4,
                num_clvs=int(num_clvs),
                warmup_time=warmup_time,
                tail_time=tail_time,
                qr_time_step=qr_time_step,
                seed=int(seed),
                method=method,
                integrator_traj_tan=self.__traj_tan_integrator_func,
                poincare_section=poincare_section,
                section_index=int(section_index),
                section_value=section_value,
                crossing=int(crossing),
            )
        else:
            return compute_clvs_imp(
                q=q,
                p=p,
                total_time=total_time,
                time_step=self.__time_step,
                parameters=parameters,
                eom=self.__system_func_1,
                hess_H=self.__system_func_2,
                num_clvs=int(num_clvs),
                warmup_time=warmup_time,
                tail_time=tail_time,
                qr_time_step=qr_time_step,
                seed=int(seed),
                method=method,
                tol=self.__tol,
                max_iter=self.__max_iter,
                integrator_traj_tan=self.__traj_tan_integrator_func,
                poincare_section=poincare_section,
                section_index=int(section_index),
                section_value=section_value,
                crossing=int(crossing),
            )

    def CLV_angles(
        self,
        q: numeric_like_t,
        p: numeric_like_t,
        total_time: numeric_t,
        parameters: numeric_like_t | None = None,
        subspaces: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...] | None = None,
        pairs: tuple[tuple[int, int], ...] | None = None,
        warmup_time: numeric_t = 0.0,
        tail_time: numeric_t = 0.0,
        qr_time_step: numeric_t | None = None,
        seed: int_t = 1312,
        poincare_section: bool = False,
        section_index: int_t = 0,
        section_value: numeric_t = 0.0,
        crossing: int_t = 1,
        method: str = "QR",
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Compute CLV-based angle diagnostics for a Hamiltonian system.

        Parameters
        ----------
        q : numeric_like_t
            Initial generalized coordinates. Must define a 1D array of shape
            `(dof,)`.
        p : numeric_like_t
            Initial generalized momenta. Must define a 1D array of shape
            `(dof,)` and have the same shape as `q`.
        total_time : numeric_t
            Total integration time used for the angle diagnostics.
        parameters : numeric_like_t | None, optional
            System parameters. If `None`, the parameters stored in the instance
            are used.
        subspaces : tuple[tuple[tuple[int, ...], tuple[int, ...]], ...] | None, optional
            Requested subspace-angle diagnostics. Each entry is a pair `(A, B)`,
            where `A` and `B` are tuples of CLV indices defining two subspaces.
        pairs : tuple[tuple[int, int], ...] | None, optional
            Requested pairwise CLV angles. Each entry `(i, j)` returns the angle
            between CLV `i` and CLV `j`.
        warmup_time : numeric_t, optional
            Forward QR warmup duration passed to the CLV computation.
        tail_time : numeric_t, optional
            Backward-recursion convergence duration passed to the CLV computation.
        qr_time_step : numeric_t | None, optional
            Time spacing between successive QR factorizations. If `None`,
            defaults to the integrator time step.
        seed : int_t, optional
            Seed used in the CLV computation.
        poincare_section : bool, optional
            If `True`, return the trajectory restricted to a Poincaré section.
        section_index : int_t, optional
            Index of the coordinate defining the section.
        section_value : numeric_t, optional
            Value of the section coordinate.
        crossing : int_t, optional
            Crossing rule:
            - `-1` for downward crossings
            - `0` for all crossings
            - `1` for upward crossings
        method : str, optional
            QR decomposition method:
            - `"QR"`: internal reduced modified Gram-Schmidt QR
            - `"QR_HH"`: `numpy.linalg.qr` based on Householder reflections

        Returns
        -------
        tuple[NDArray[np.float64], NDArray[np.float64]]
            - `angles`: array of shape `(T, M)` containing the requested angle
              time series
            - `traj`: sampled trajectory array with columns `[t, q..., p...]`

        Raises
        ------
        ValueError
            If Hessian functions are missing.
            If `q` and `p` do not have the same shape.
            If `total_time` is not positive.
            If `warmup_time` or `tail_time` is negative.
            If `qr_time_step` is smaller than the integration time step.
            If both `subspaces` and `pairs` are missing or empty.
            If any subspace or pair specification is invalid.
            If Poincaré-section parameters are inconsistent.
            If `method` is not `"QR"` or `"QR_HH"`.
        TypeError
            If `poincare_section` is not boolean.
            If `method` is not a string.
        """
        if self.__hess_T is None or self.__hess_V is None:
            raise ValueError("Hessian functions are required to compute CLV angles")

        q = validate_initial_conditions(
            q, self.__degrees_of_freedom, allow_ensemble=False
        )
        p = validate_initial_conditions(
            p, self.__degrees_of_freedom, allow_ensemble=False
        )

        if q.shape != p.shape:
            raise ValueError("q and p must have the same shape")

        validate_positive(total_time, "total_time", Real)
        validate_non_negative(warmup_time, "warmup_time", Real)
        validate_non_negative(tail_time, "tail_time", Real)
        validate_non_negative(seed, "seed", Integral)

        total_time = np.float64(total_time)
        warmup_time = np.float64(warmup_time)
        tail_time = np.float64(tail_time)

        if qr_time_step is None:
            qr_time_step = self.__time_step
        else:
            validate_positive(qr_time_step, "qr_time_step", Real)
            qr_time_step = np.float64(qr_time_step)
            if qr_time_step < self.__time_step:
                raise ValueError(
                    f"qr_time_step must be >= time_step. Got {qr_time_step}"
                )

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        else:
            parameters = validate_parameters(parameters, self.__number_of_parameters)

        if not isinstance(method, str):
            raise TypeError("method must be a string")

        method = method.upper()
        if method not in ("QR", "QR_HH"):
            raise ValueError("method must be 'QR' or 'QR_HH'")

        pairs = validate_clv_pairs(pairs, 2 * self.__degrees_of_freedom)
        subspaces = validate_clv_subspaces(subspaces, 2 * self.__degrees_of_freedom)

        if not isinstance(poincare_section, bool):
            raise TypeError("poincare_section must be a boolean")

        if poincare_section:
            validate_non_negative(section_index, "section_index", Integral)
            if section_index >= self.__degrees_of_freedom:
                raise ValueError(
                    "section_index must be in the range [0, degrees_of_freedom)"
                )

            if isinstance(section_value, bool) or not isinstance(section_value, Real):
                raise TypeError("section_value must be a valid real number")
            section_value = np.float64(section_value)

            if isinstance(crossing, bool) or not isinstance(crossing, Integral):
                raise TypeError("crossing must be an integer")
            if crossing not in (-1, 0, 1):
                raise ValueError("crossing must be -1, 0, or 1")
        else:
            section_index = 0
            section_value = np.float64(0.0)
            crossing = 1

        if self.__integrator in ["vv2", "svy4"]:
            return clv_angles_sep(
                q=q,
                p=p,
                total_time=total_time,
                time_step=self.__time_step,
                parameters=parameters,
                grad_T=self.__system_func_1,
                grad_V=self.__system_func_2,
                hess_T=self.__system_func_3,
                hess_V=self.__system_func_4,
                warmup_time=warmup_time,
                tail_time=tail_time,
                qr_time_step=qr_time_step,
                seed=int(seed),
                method=method,
                integrator_traj_tan=self.__traj_tan_integrator_func,
                poincare_section=poincare_section,
                section_index=int(section_index),
                section_value=section_value,
                crossing=int(crossing),
                subspaces=subspaces,
                pairs=pairs,
            )
        else:
            return clv_angles_imp(
                q=q,
                p=p,
                total_time=total_time,
                time_step=self.__time_step,
                parameters=parameters,
                eom=self.__system_func_1,
                hess_H=self.__system_func_2,
                warmup_time=warmup_time,
                tail_time=tail_time,
                qr_time_step=qr_time_step,
                seed=int(seed),
                method=method,
                tol=self.__tol,
                max_iter=self.__max_iter,
                integrator_traj_tan=self.__traj_tan_integrator_func,
                poincare_section=poincare_section,
                section_index=int(section_index),
                section_value=section_value,
                crossing=int(crossing),
                subspaces=subspaces,
                pairs=pairs,
            )

    def SALI(
        self,
        q: numeric_like_t,
        p: numeric_like_t,
        total_time: numeric_t,
        parameters: numeric_like_t | None = None,
        return_history: bool = False,
        seed: int_t = 1312,
        threshold: numeric_t = 1e-16,
    ) -> NDArray[np.float64]:
        """
        Compute the Smaller Alignment Index (SALI).

        SALI distinguishes between chaotic and regular motion by evolving two
        deviation vectors and monitoring their alignment over time. In chaotic
        motion, SALI tends exponentially to zero, while in regular motion it
        remains bounded away from zero.

        Parameters
        ----------
        q : numeric_like_t
            Initial generalized coordinates. Must define a 1D array of shape
            `(dof,)`.
        p : numeric_like_t
            Initial generalized momenta. Must define a 1D array of shape
            `(dof,)` and have the same shape as `q`.
        total_time : numeric_t
            Total integration time.
        parameters : numeric_like_t | None, optional
            System parameters. If `None`, the parameters stored in the instance
            are used.
        return_history : bool, optional
            If `True`, return the time evolution of SALI.
        seed : int_t, optional
            Random seed used to initialize the deviation vectors.
        threshold : numeric_t, optional
            Early stopping threshold. Integration stops when `SALI <= threshold`.

        Returns
        -------
        NDArray[np.float64]
            - If `return_history=True`, returns an array of shape `(N, 2)` with
              columns `[time, SALI]`.
            - If `return_history=False`, returns an array of shape `(2,)`
              containing the final `[time, SALI]`.

        Raises
        ------
        ValueError
            If Hessian functions are missing.
            If `q` and `p` do not have the same shape.
            If `total_time` is not positive.
            If `threshold` is negative.
            If the number of parameters does not match the expected number.
        TypeError
            If `return_history` is not boolean.
            If `seed` is not an integer.
            If `total_time` or `threshold` is not a valid real number.
        """
        if self.__hess_T is None or self.__hess_V is None:
            raise ValueError("Hessian functions are required to compute SALI")

        q = validate_initial_conditions(
            q, self.__degrees_of_freedom, allow_ensemble=False
        )
        p = validate_initial_conditions(
            p, self.__degrees_of_freedom, allow_ensemble=False
        )

        if q.shape != p.shape:
            raise ValueError("q and p must have the same shape")

        validate_positive(total_time, "total_time", Real)
        total_time = np.float64(total_time)

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        else:
            parameters = validate_parameters(parameters, self.__number_of_parameters)

        if not isinstance(return_history, bool):
            raise TypeError("return_history must be a boolean")

        validate_non_negative(seed, "seed", Integral)
        validate_non_negative(threshold, "threshold", Real)
        threshold = np.float64(threshold)

        if self.__integrator in ["vv2", "svy4"]:
            result = sali_sep(
                q=q,
                p=p,
                total_time=total_time,
                time_step=self.__time_step,
                parameters=parameters,
                grad_T=self.__system_func_1,
                grad_V=self.__system_func_2,
                hess_T=self.__system_func_3,
                hess_V=self.__system_func_4,
                return_history=return_history,
                seed=int(seed),
                threshold=threshold,
                integrator_traj_tan=self.__traj_tan_integrator_func,
            )

        else:
            result = sali_imp(
                q=q,
                p=p,
                total_time=total_time,
                time_step=self.__time_step,
                parameters=parameters,
                eom=self.__system_func_1,
                hess_H=self.__system_func_2,
                return_history=return_history,
                seed=int(seed),
                threshold=threshold,
                tol=self.__tol,
                max_iter=self.__max_iter,
                integrator_traj_tan=self.__traj_tan_integrator_func,
            )

        if return_history:
            return result

        return result[0]

    def LDI(
        self,
        q: numeric_like_t,
        p: numeric_like_t,
        total_time: numeric_t,
        k: int_t,
        parameters: numeric_like_t | None = None,
        return_history: bool = False,
        seed: int_t = 13,
        threshold: numeric_t = 1e-16,
    ) -> NDArray[np.float64]:
        """
        Compute the Linear Dependence Index (LDI).

        LDI measures the linear dependence among `k` deviation vectors evolved
        along a trajectory. It is computed from the product of singular values
        of the deviation matrix. In chaotic motion, LDI tends rapidly to zero,
        while in regular motion it remains bounded away from zero.

        Parameters
        ----------
        q : numeric_like_t
            Initial generalized coordinates. Must define a 1D array of shape
            `(dof,)`.
        p : numeric_like_t
            Initial generalized momenta. Must define a 1D array of shape
            `(dof,)` and have the same shape as `q`.
        total_time : numeric_t
            Total integration time.
        k : int_t
            Number of deviation vectors to evolve.
        parameters : numeric_like_t | None, optional
            System parameters. If `None`, the parameters stored in the instance
            are used.
        return_history : bool, optional
            If `True`, return the time evolution of LDI.
        seed : int_t, optional
            Random seed used to initialize the deviation vectors.
        threshold : numeric_t, optional
            Early stopping threshold. Integration stops when `LDI <= threshold`.

        Returns
        -------
        NDArray[np.float64]
            - If `return_history=True`, returns an array of shape `(N, 2)` with
              columns `[time, LDI]`.
            - If `return_history=False`, returns an array of shape `(2,)`
              containing the final `[time, LDI]`.

        Raises
        ------
        ValueError
            If Hessian functions are missing.
            If `q` and `p` do not have the same shape.
            If `total_time` is not positive.
            If `k` is not in `[2, 2 * degrees_of_freedom]`.
            If `threshold` is negative.
            If the number of parameters does not match the expected number.
        TypeError
            If `return_history` is not boolean.
            If `k` or `seed` is not an integer.
            If `total_time` or `threshold` is not a valid real number.
        """
        if self.__hess_T is None or self.__hess_V is None:
            raise ValueError("Hessian functions are required to compute LDI")

        q = validate_initial_conditions(
            q, self.__degrees_of_freedom, allow_ensemble=False
        )
        p = validate_initial_conditions(
            p, self.__degrees_of_freedom, allow_ensemble=False
        )

        if q.shape != p.shape:
            raise ValueError("q and p must have the same shape")

        validate_positive(total_time, "total_time", Real)
        total_time = np.float64(total_time)

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        else:
            parameters = validate_parameters(parameters, self.__number_of_parameters)

        validate_positive(k, "k", Integral)
        if k < 2 or k > 2 * self.__degrees_of_freedom:
            raise ValueError("k must be in the range [2, 2 * degrees_of_freedom]")

        if not isinstance(return_history, bool):
            raise TypeError("return_history must be a boolean")

        validate_non_negative(seed, "seed", Integral)
        validate_non_negative(threshold, "threshold", Real)
        threshold = np.float64(threshold)

        if self.__integrator in ["vv2", "svy4"]:
            result = ldi_k_sep(
                q=q,
                p=p,
                total_time=total_time,
                time_step=self.__time_step,
                parameters=parameters,
                grad_T=self.__system_func_1,
                grad_V=self.__system_func_2,
                hess_T=self.__system_func_3,
                hess_V=self.__system_func_4,
                k=int(k),
                return_history=return_history,
                seed=int(seed),
                threshold=threshold,
                integrator_traj_tan=self.__traj_tan_integrator_func,
            )

        else:
            result = ldi_k_imp(
                q=q,
                p=p,
                total_time=total_time,
                time_step=self.__time_step,
                parameters=parameters,
                eom=self.__system_func_1,
                hess_H=self.__system_func_2,
                k=int(k),
                return_history=return_history,
                seed=int(seed),
                threshold=threshold,
                tol=self.__tol,
                max_iter=self.__max_iter,
                integrator_traj_tan=self.__traj_tan_integrator_func,
            )

        if return_history:
            return result

        return result[0]

    def GALI(
        self,
        q: numeric_like_t,
        p: numeric_like_t,
        total_time: numeric_t,
        k: int_t,
        parameters: numeric_like_t | None = None,
        return_history: bool = False,
        seed: int_t = 13,
        threshold: numeric_t = 1e-16,
        method: str = "QR",
    ) -> NDArray[np.float64]:
        """
        Compute the Generalized Alignment Index (GALI).

        GALI extends SALI by considering the evolution of `k` deviation
        vectors. It measures the volume spanned by the normalized deviation
        vectors. In chaotic motion, GALI tends rapidly to zero, while in
        regular motion it typically follows a slower decay or remains bounded.

        Parameters
        ----------
        q : numeric_like_t
            Initial generalized coordinates. Must define a 1D array of shape
            `(dof,)`.
        p : numeric_like_t
            Initial generalized momenta. Must define a 1D array of shape
            `(dof,)` and have the same shape as `q`.
        total_time : numeric_t
            Total integration time.
        k : int_t
            Number of deviation vectors to evolve.
        parameters : numeric_like_t | None, optional
            System parameters. If `None`, the parameters stored in the instance
            are used.
        return_history : bool, optional
            If `True`, return the time evolution of GALI.
        seed : int_t, optional
            Random seed used to initialize the deviation vectors.
        threshold : numeric_t, optional
            Early stopping threshold. Integration stops when `GALI <= threshold`.
        method : str, optional
            Method used to compute GALI:
            - `"DET"`: determinant of the Gram matrix
            - `"QR"`: product of diagonal entries from the internal QR routine
            - `"QR_HH"`: product of diagonal entries from `numpy.linalg.qr`

        Returns
        -------
        NDArray[np.float64]
            - If `return_history=True`, returns an array of shape `(N, 2)` with
              columns `[time, GALI]`.
            - If `return_history=False`, returns an array of shape `(2,)`
              containing the final `[time, GALI]`.

        Raises
        ------
        ValueError
            If Hessian functions are missing.
            If `q` and `p` do not have the same shape.
            If `total_time` is not positive.
            If `k` is not in `[2, 2 * degrees_of_freedom]`.
            If `threshold` is negative.
            If `method` is not `"DET"`, `"QR"`, or `"QR_HH"`.
            If the number of parameters does not match the expected number.
        TypeError
            If `return_history` is not boolean.
            If `k` or `seed` is not an integer.
            If `method` is not a string.
            If `total_time` or `threshold` is not a valid real number.
        """
        if self.__hess_T is None or self.__hess_V is None:
            raise ValueError("Hessian functions are required to compute GALI")

        q = validate_initial_conditions(
            q, self.__degrees_of_freedom, allow_ensemble=False
        )
        p = validate_initial_conditions(
            p, self.__degrees_of_freedom, allow_ensemble=False
        )

        if q.shape != p.shape:
            raise ValueError("q and p must have the same shape")

        validate_positive(total_time, "total_time", Real)
        total_time = np.float64(total_time)

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        else:
            parameters = validate_parameters(parameters, self.__number_of_parameters)

        validate_positive(k, "k", Integral)
        if k < 2 or k > 2 * self.__degrees_of_freedom:
            raise ValueError("k must be in the range [2, 2 * degrees_of_freedom]")

        if not isinstance(return_history, bool):
            raise TypeError("return_history must be a boolean")

        validate_non_negative(seed, "seed", Integral)
        validate_non_negative(threshold, "threshold", Real)
        threshold = np.float64(threshold)

        if not isinstance(method, str):
            raise TypeError("method must be a string")

        method = method.upper()
        if method not in ("DET", "QR", "QR_HH"):
            raise ValueError("method must be 'DET', 'QR', or 'QR_HH'")

        if self.__integrator in ["vv2", "svy4"]:
            result = gali_k_sep(
                q=q,
                p=p,
                total_time=total_time,
                time_step=self.__time_step,
                parameters=parameters,
                grad_T=self.__system_func_1,
                grad_V=self.__system_func_2,
                hess_T=self.__system_func_3,
                hess_V=self.__system_func_4,
                k=int(k),
                return_history=return_history,
                seed=int(seed),
                threshold=threshold,
                method=method,
                integrator_traj_tan=self.__traj_tan_integrator_func,
            )
        else:
            result = gali_k_imp(
                q=q,
                p=p,
                total_time=total_time,
                time_step=self.__time_step,
                parameters=parameters,
                eom=self.__system_func_1,
                hess_H=self.__system_func_2,
                k=int(k),
                return_history=return_history,
                seed=int(seed),
                threshold=threshold,
                method=method,
                tol=self.__tol,
                max_iter=self.__max_iter,
                integrator_traj_tan=self.__traj_tan_integrator_func,
            )

        if return_history:
            return result

        return result[0]

    def recurrence_time_entropy(
        self,
        q: numeric_like_t,
        p: numeric_like_t,
        num_intersections: int_t,
        parameters: numeric_like_t | None = None,
        section_index: int_t = 0,
        section_value: numeric_t = 0.0,
        crossing: int_t = 1,
        periodic_section_coordinate: bool = False,
        period: numeric_t = 2.0 * np.pi,
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
        Compute the recurrence time entropy (RTE) for a Hamiltonian system.

        Parameters
        ----------
        q : numeric_like_t
            Initial generalized coordinates. Must define a 1D array of shape
            `(dof,)`.
        p : numeric_like_t
            Initial generalized momenta. Must define a 1D array of shape
            `(dof,)` and have the same shape as `q`.
        num_intersections : int_t
            Number of Poincaré-section crossings used in the recurrence analysis.
        parameters : numeric_like_t | None, optional
            System parameters. If `None`, the parameters stored in the instance
            are used.
        section_index : int_t, optional
            Index of the coordinate used to define the Poincaré section.
        section_value : numeric_t, optional
            Value of the section coordinate.
        crossing : int_t, optional
            Crossing rule:
            - `-1` for downward crossings
            - `0` for all crossings
            - `1` for upward crossings
        periodic_section_coordinate : bool, optional
            If True, treats q[section_index] as a periodic coordinate on S¹ and
            performs crossing detection using modulo arithmetic.
            If False, uses standard Euclidean crossing detection.
        period : numeric_t, optional
            Period of the angular coordinate when
            `periodic_section_coordinate=True`.
            Typically 2π for action-angle systems.
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
            - the final Poincaré-section point without time
            - the recurrence matrix
            - the white-vertical-line distribution

        Raises
        ------
        ValueError
            If `q` and `p` do not have the same shape.
            If `num_intersections` is negative.
            If `section_index` is outside `[0, degrees_of_freedom)`.
            If `crossing` is not one of `-1`, `0`, or `1`.
            If the number of parameters does not match the expected number.
        TypeError
            If `section_value` is not a valid real number.
            If `section_index`, `crossing`, or `num_intersections` is not an integer.
        """
        q = validate_initial_conditions(
            q, self.__degrees_of_freedom, allow_ensemble=False
        )
        p = validate_initial_conditions(
            p, self.__degrees_of_freedom, allow_ensemble=False
        )

        if q.shape != p.shape:
            raise ValueError("q and p must have the same shape")

        validate_non_negative(num_intersections, "num_intersections", Integral)

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        else:
            parameters = validate_parameters(parameters, self.__number_of_parameters)

        validate_non_negative(section_index, "section_index", Integral)
        if section_index >= self.__degrees_of_freedom:
            raise ValueError(
                "section_index must be in the range [0, degrees_of_freedom)"
            )

        if isinstance(section_value, bool) or not isinstance(section_value, Real):
            raise TypeError("section_value must be a valid real number")
        section_value = np.float64(section_value)

        if isinstance(crossing, bool) or not isinstance(crossing, Integral):
            raise TypeError("crossing must be an integer")
        if crossing not in (-1, 0, 1):
            raise ValueError("crossing must be -1, 0, or 1")

        return recurrence_time_entropy_core(
            q=q,
            p=p,
            num_points=np.int64(num_intersections),
            parameters=parameters,
            system_func_1=self.__system_func_1,
            system_func_2=self.__system_func_2,
            time_step=self.__time_step,
            integrator=self.__integrator_func,
            section_index=int(section_index),
            section_value=section_value,
            crossing=int(crossing),
            periodic_section_coordinate=periodic_section_coordinate,
            period=np.float64(period),
            tol=self.__tol,
            max_iter=self.__max_iter,
            **kwargs,
        )

    def hurst_exponent(
        self,
        q: numeric_like_t,
        p: numeric_like_t,
        num_intersections: int_t,
        parameters: numeric_like_t | None = None,
        wmin: int_t = 2,
        section_index: int_t = 0,
        section_value: numeric_t = 0.0,
        crossing: int_t = 1,
        periodic_section_coordinate: bool = False,
        period: numeric_t = 2 * np.pi,
    ) -> NDArray[np.float64]:
        """
        Estimate the Hurst exponent from a Hamiltonian Poincaré section.

        Parameters
        ----------
        q : numeric_like_t
            Initial generalized coordinates. Must define a 1D array of shape
            `(dof,)`.
        p : numeric_like_t
            Initial generalized momenta. Must define a 1D array of shape
            `(dof,)` and have the same shape as `q`.
        num_intersections : int_t
            Number of Poincaré-section crossings used in the analysis.
        parameters : numeric_like_t | None, optional
            System parameters. If `None`, the parameters stored in the instance
            are used.
        wmin : int_t, optional
            Minimum window size used in the rescaled-range calculation.
        section_index : int_t, optional
            Index of the coordinate used to define the Poincaré section.
        section_value : numeric_t, optional
            Value of the section coordinate.
        crossing : int_t, optional
            Crossing rule:
            - `-1` for downward crossings
            - `0` for all crossings
            - `1` for upward crossings
        periodic_section_coordinate : bool, optional
            If True, treats q[section_index] as a periodic coordinate on S¹ and
            performs crossing detection using modulo arithmetic.
            If False, uses standard Euclidean crossing detection.
        period : numeric_t, optional
            Period of the angular coordinate when
            `periodic_section_coordinate=True`.
            Typically 2π for action-angle systems.

        Returns
        -------
        NDArray[np.float64]
            Estimated Hurst exponent values for the reduced Poincaré-section
            coordinates.

        Raises
        ------
        ValueError
            If `q` and `p` do not have the same shape.
            If `num_intersections` is negative.
            If `section_index` is outside `[0, degrees_of_freedom)`.
            If `crossing` is not one of `-1`, `0`, or `1`.
            If `wmin < 2` or `wmin >= num_intersections // 2`.
            If the number of parameters does not match the expected number.
        TypeError
            If `section_value` is not a valid real number.
            If `wmin`, `section_index`, `crossing`, or `num_intersections`
            is not an integer.
        """
        q = validate_initial_conditions(
            q, self.__degrees_of_freedom, allow_ensemble=False
        )
        p = validate_initial_conditions(
            p, self.__degrees_of_freedom, allow_ensemble=False
        )

        if q.shape != p.shape:
            raise ValueError("q and p must have the same shape")

        validate_non_negative(num_intersections, "num_intersections", Integral)

        if parameters is None and self.__parameters is not None:
            parameters = self.__parameters
        else:
            parameters = validate_parameters(parameters, self.__number_of_parameters)

        validate_non_negative(section_index, "section_index", Integral)
        if section_index >= self.__degrees_of_freedom:
            raise ValueError(
                "section_index must be in the range [0, degrees_of_freedom)"
            )

        if isinstance(section_value, bool) or not isinstance(section_value, Real):
            raise TypeError("section_value must be a valid real number")
        section_value = np.float64(section_value)

        if isinstance(crossing, bool) or not isinstance(crossing, Integral):
            raise TypeError("crossing must be an integer")
        if crossing not in (-1, 0, 1):
            raise ValueError("crossing must be -1, 0, or 1")

        validate_positive(wmin, "wmin", Integral)
        if wmin < 2 or wmin >= num_intersections // 2:
            raise ValueError(
                f"`wmin` must be an integer >= 2 and < num_intersections // 2. Got {wmin}."
            )

        return hurst_exponent_wrapped(
            q=q,
            p=p,
            num_points=np.int64(num_intersections),
            parameters=parameters,
            system_func_1=self.__system_func_1,
            system_func_2=self.__system_func_2,
            time_step=self.__time_step,
            integrator=self.__integrator_func,
            section_index=int(section_index),
            section_value=section_value,
            crossing=int(crossing),
            periodic_section_coordinate=periodic_section_coordinate,
            period=np.float64(period),
            wmin=int(wmin),
            tol=self.__tol,
            max_iter=self.__max_iter,
        )
