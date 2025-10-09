# chaotic_indicators.py

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

from typing import Callable

import numpy as np
from numba import njit
from numpy.typing import NDArray

from pynamicalsys.common.recurrence_quantification_analysis import (
    RTEConfig,
    recurrence_matrix,
    white_vertline_distr,
)
from pynamicalsys.common.time_series_metrics import hurst_exponent
from pynamicalsys.common.utils import qr, wedge_norm
from pynamicalsys.hamiltonian_systems.trajectory_analysis import (
    generate_poincare_section,
)


@njit
def lyapunov_spectrum(
    q: NDArray[np.float64],
    p: NDArray[np.float64],
    total_time: float,
    time_step: float,
    parameters: NDArray[np.float64],
    grad_T: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    grad_V: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    hess_T: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    hess_V: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    num_exponents: int,
    qr_interval: int,
    return_history: bool,
    seed: int,
    log_base: float,
    QR: Callable[
        [NDArray[np.float64]], tuple[NDArray[np.float64], NDArray[np.float64]]
    ],
    integrator_traj_tan: Callable,
) -> NDArray[np.float64]:
    """
    Compute the full Lyapunov spectrum of a Hamiltonian system.

    Parameters
    ----------
    q : NDArray[np.float64], shape (dof,)
        Initial generalized coordinates.
    p : NDArray[np.float64], shape (dof,)
        Initial generalized momenta.
    total_time : float
        Total integration time.
    time_step : float
        Integration step size.
    parameters : NDArray[np.float64]
        Additional system parameters.
    grad_T : Callable
        Gradient of kinetic energy with respect to momenta.
    grad_V : Callable
        Gradient of potential energy with respect to coordinates.
    hess_T : Callable
        Hessian of kinetic energy with respect to momenta.
    hess_V : Callable
        Hessian of potential energy with respect to coordinates.
    num_exponents : int
        Number of Lyapunov exponents to compute.
    qr_interval : int
        Interval (in steps) between QR re-orthonormalizations.
    return_history : bool
        If True, return time evolution of exponents; if False, return only final values.
    seed : int
        Random seed for deviation vector initialization.
    log_base : float
        Base of the logarithm used for normalization.
    QR : Callable
        Function for orthonormalization (returns Q, R).
    integrator_traj_tan : Callable
        Symplectic integrator for the main trajectory AND tangent vectors.

    Returns
    -------
    spectrum : NDArray[np.float64], shape (num_steps/qr_interval, num_exponents+1) or (1, num_exponents)
        - If `return_history=True`: time and instantaneous Lyapunov exponents.
        - If `return_history=False`: final averaged Lyapunov spectrum.
    """
    num_steps = round(total_time / time_step)
    dof = len(q)
    neq = 2 * dof

    np.random.seed(seed)
    dv = -1 + 2 * np.random.rand(neq, num_exponents)
    dv, _ = QR(dv)

    exponents = np.zeros(num_exponents, dtype=np.float64)
    history = np.zeros((round(num_steps / qr_interval), num_exponents + 1))
    count = 0
    for i in range(num_steps):
        time = (i + 1) * time_step
        # Evolve trajectory and tangent vectors
        q, p, dv = integrator_traj_tan(
            q, p, dv, time_step, grad_T, grad_V, hess_T, hess_V, parameters
        )

        if i % qr_interval == 0:
            count += 1
            # Orthonormalize the deviation vectors
            dv, R = QR(dv)

            # Acculate the log
            exponents += np.log(np.abs(np.diag(R)))

            if return_history:
                result = np.zeros(num_exponents + 1)
                result[0] = time
                for j in range(num_exponents):
                    result[j + 1] = exponents[j] / time
                history[count - 1, :] = result

    if return_history:
        history = history / np.log(log_base)
        return history
    else:
        spectrum = np.zeros((1, num_exponents))
        spectrum[0, :] = exponents / (total_time * np.log(log_base))
        return spectrum


@njit
def maximum_lyapunov_exponent(
    q: NDArray[np.float64],
    p: NDArray[np.float64],
    total_time: float,
    time_step: float,
    parameters: NDArray[np.float64],
    grad_T: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    grad_V: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    hess_T: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    hess_V: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    return_history: bool,
    seed: int,
    log_base: float,
    integrator_traj_tan: Callable,
) -> NDArray[np.float64]:
    """
    Compute the maximum Lyapunov exponent (MLE).

    Parameters
    ----------
    q : NDArray[np.float64], shape (dof,)
        Initial coordinates.
    p : NDArray[np.float64], shape (dof,)
        Initial momenta.
    total_time : float
        Total integration time.
    time_step : float
        Integration step size.
    parameters : NDArray[np.float64]
        System parameters.
    grad_T, grad_V, hess_T, hess_V : Callable
        Gradient and Hessian functions of kinetic/potential energies.
    return_history : bool
        If True, return time series of MLE estimates.
    seed : int
        Random seed for initial deviation vector.
    log_base : float
        Base of logarithm.
    integrator_traj_tan : Callable
        Symplectic trajectory and tangent integrator.

    Returns
    -------
    mle : NDArray[np.float64], shape (num_steps, 2) or (1, 1)
        - If `return_history=True`: time vs. running MLE.
        - If `return_history=False`: final MLE value.
    """
    num_steps = round(total_time / time_step)
    dof = len(q)

    np.random.seed(seed)
    dv = np.random.uniform(-1, 1, 2 * dof)
    norm = np.linalg.norm(dv)
    dv /= norm
    dv = dv.reshape(2 * dof, 1)

    lyapunov_exponent = 0
    history = np.zeros((num_steps, 2))
    for i in range(num_steps):
        time = (i + 1) * time_step
        # Evolve trajectory
        q, p, dv = integrator_traj_tan(
            q, p, dv, time_step, grad_T, grad_V, hess_T, hess_V, parameters
        )

        # Norm of the deviation vector
        norm = np.linalg.norm(dv[:, 0])

        # Acculate the log
        lyapunov_exponent += np.log(norm)

        # Renormalize the deviation vector
        dv /= norm

        if return_history:
            history[i, 0] = time
            history[i, 1] = lyapunov_exponent / time

    if return_history:
        history = history / np.log(log_base)
        return history
    else:
        result = np.zeros((1, 1))
        result[0, 0] = lyapunov_exponent / time
        return result


@njit
def SALI(
    q: NDArray[np.float64],
    p: NDArray[np.float64],
    total_time: float,
    time_step: float,
    parameters: NDArray[np.float64],
    grad_T: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    grad_V: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    hess_T: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    hess_V: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    return_history: bool,
    seed: int,
    integrator_traj_tan: Callable,
    threshold: float,
) -> list[list[float]]:
    """
    Compute the Smaller Alignment Index (SALI).

    Parameters
    ----------
    q, p : NDArray[np.float64], shape (dof,)
        Initial conditions.
    total_time : float
        Total integration time.
    time_step : float
        Integration step size.
    parameters : NDArray[np.float64]
        System parameters.
    grad_T, grad_V, hess_T, hess_V : Callable
        Gradient and Hessian functions of the Hamiltonian.
    return_history : bool
        If True, return time evolution of SALI.
    seed : int
        Random seed for deviation vectors.
    integrator_traj_tan : Callable
        Symplectic trajectory and tangent integrator.
    threshold : float
        Early termination threshold for SALI.

    Returns
    -------
    sali : list of [time, value]
        Time evolution of SALI (or final value if `return_history=False`).
    """
    num_steps = round(total_time / time_step)
    dof = len(q)
    neq = 2 * dof

    np.random.seed(seed)
    dv = -1 + 2 * np.random.rand(neq, 2)
    dv, _ = qr(dv)

    history = []
    for i in range(num_steps):
        time = (i + 1) * time_step
        # Evolve trajectory and tangent vectors
        q, p, dv = integrator_traj_tan(
            q, p, dv, time_step, grad_T, grad_V, hess_T, hess_V, parameters
        )

        # Normalize deviation vectors
        for j in range(2):
            norm = np.linalg.norm(dv[:, j])
            dv[:, j] /= norm

        pai = np.linalg.norm(dv[:, 0] + dv[:, 1])
        aai = np.linalg.norm(dv[:, 0] - dv[:, 1])

        sali = min(pai, aai)

        if return_history:
            result = [time, sali]
            history.append(result)

        if sali <= threshold:
            break

    if return_history:
        return history
    else:
        return [[time, sali]]


def LDI(
    q: NDArray[np.float64],
    p: NDArray[np.float64],
    total_time: float,
    time_step: float,
    parameters: NDArray[np.float64],
    grad_T: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    grad_V: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    hess_T: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    hess_V: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    k: int,
    return_history: bool,
    seed: int,
    integrator_traj_tan: Callable,
    threshold: float,
) -> list[list[float]]:
    """
    Compute the Linear Dependence Index (LDI).

    Parameters
    ----------
    q, p : NDArray[np.float64], shape (dof,)
        Initial conditions.
    total_time : float
        Total integration time.
    time_step : float
        Integration step size.
    parameters : NDArray[np.float64]
        System parameters.
    grad_T, grad_V, hess_T, hess_V : Callable
        Gradient and Hessian functions.
    k : int
        Number of deviation vectors.
    return_history : bool
        If True, return LDI time series.
    seed : int
        Random seed for initialization.
    integrator_traj_tan : Callable
        Symplectic trajectory and tangent integrator.
    threshold : float
        Early termination threshold.

    Returns
    -------
    ldi : list of [time, value]
        LDI evolution (or final value).
    """
    num_steps = round(total_time / time_step)
    dof = len(q)
    neq = 2 * dof

    np.random.seed(seed)
    dv = -1 + 2 * np.random.rand(neq, k)
    dv, _ = qr(dv)

    history = []
    for i in range(num_steps):
        time = (i + 1) * time_step
        # Evolve trajectory and tangent vectors
        q, p, dv = integrator_traj_tan(
            q, p, dv, time_step, grad_T, grad_V, hess_T, hess_V, parameters
        )

        # Normalize deviation vectors
        for j in range(k):
            norm = np.linalg.norm(dv[:, j])
            dv[:, j] /= norm

        # Calculate the singular values
        S = np.linalg.svd(dv, full_matrices=False, compute_uv=False)
        ldi = np.exp(np.sum(np.log(S)))  # LDI is the product of all singular values

        if return_history:
            result = [time, ldi]
            history.append(result)

        # Early termination
        if ldi <= threshold:
            break

    if return_history:
        return history
    else:
        return [[time, ldi]]


def GALI(
    q: NDArray[np.float64],
    p: NDArray[np.float64],
    total_time: float,
    time_step: float,
    parameters: NDArray[np.float64],
    grad_T: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    grad_V: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    hess_T: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    hess_V: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    k: int,
    return_history: bool,
    seed: int,
    integrator_traj_tan: Callable,
    threshold: float,
) -> list[list[float]]:
    """
    Compute the Generalized Alignment Index (GALI).

    Parameters
    ----------
    q, p : NDArray[np.float64], shape (dof,)
        Initial conditions.
    total_time : float
        Total integration time.
    time_step : float
        Integration step size.
    parameters : NDArray[np.float64]
        System parameters.
    grad_T, grad_V, hess_T, hess_V : Callable
        Gradient and Hessian functions.
    k : int
        Number of deviation vectors.
    return_history : bool
        If True, return GALI time series.
    seed : int
        Random seed for initialization.
    integrator_traj_tan : Callable
        Symplectic trajectory and tangent integrator.
    threshold : float
        Early termination threshold.

    Returns
    -------
    gali : list of [time, value]
        GALI evolution (or final value).
    """
    num_steps = round(total_time / time_step)
    dof = len(q)
    neq = 2 * dof

    np.random.seed(seed)
    dv = -1 + 2 * np.random.rand(neq, k)
    dv, _ = qr(dv)

    history = []
    for i in range(num_steps):
        time = (i + 1) * time_step
        # Evolve trajectory and tangent vectors
        q, p, dv = integrator_traj_tan(
            q, p, dv, time_step, grad_T, grad_V, hess_T, hess_V, parameters
        )

        # Normalize deviation vectors
        for j in range(k):
            norm = np.linalg.norm(dv[:, j])
            dv[:, j] /= norm

        # Calculate GALI
        gali = wedge_norm(dv)

        if return_history:
            result = [time, gali]
            history.append(result)

        # Early termination
        if gali <= threshold:
            break

    if return_history:
        return history
    else:
        return [[time, gali]]


def recurrence_time_entropy(
    q,
    p,
    num_points,
    parameters,
    grad_T,
    grad_V,
    time_step,
    integrator,
    section_index,
    section_value,
    crossing,
    **kwargs,
):

    # Configuration handling
    config = RTEConfig(**kwargs)

    # Metric setup
    metric_map = {"supremum": np.inf, "euclidean": 2, "manhattan": 1}

    try:
        ord = metric_map[config.std_metric.lower()]
    except KeyError:
        raise ValueError(
            f"Invalid std_metric: {config.std_metric}. Must be {list(metric_map.keys())}"
        )

    # Generate the Poincaré section or stroboscopic map
    points = generate_poincare_section(
        q,
        p,
        num_points,
        parameters,
        grad_T,
        grad_V,
        time_step,
        integrator,
        section_index,
        section_value,
        crossing,
    )
    data = points[:, 1:]  # Remove time
    data = np.delete(data, section_index, axis=1)

    # Threshold calculation
    if config.threshold_std:
        std = np.std(data, axis=0)
        eps = config.threshold * np.linalg.norm(std, ord=ord)
        if eps <= 0:
            eps = 0.1
    else:
        eps = config.threshold

    # Recurrence matrix calculation
    recmat = recurrence_matrix(data, float(eps), metric=config.metric)

    # White line distribution
    P = white_vertline_distr(recmat, wmin=config.lmin)
    P = P[P > 0]  # Remove zeros
    P /= P.sum()  # Normalize

    # Entropy calculation
    rte = -np.sum(P * np.log(P))

    # Prepare output
    result = [rte]
    if config.return_final_state:
        result.append(points[-1, 1:])
    if config.return_recmat:
        result.append(recmat)
    if config.return_p:
        result.append(P)

    return result[0] if len(result) == 1 else tuple(result)


def hurst_exponent_wrapped(
    q: NDArray[np.float64],
    p: NDArray[np.float64],
    num_points: int,
    parameters: NDArray[np.float64],
    grad_T: Callable,
    grad_V: Callable,
    time_step: float,
    integrator: Callable,
    section_index: int,
    section_value: float,
    crossing: int,
    wmin: int = 2,
) -> NDArray[np.float64]:

    q = q.copy()
    p = p.copy()

    # Generate the Poincaré section or stroboscopic map
    points = generate_poincare_section(
        q,
        p,
        num_points,
        parameters,
        grad_T,
        grad_V,
        time_step,
        integrator,
        section_index,
        section_value,
        crossing,
    )
    data = points[:, 1:]  # Remove time
    data = np.delete(data, section_index, axis=1)

    return hurst_exponent(data, wmin=wmin)
