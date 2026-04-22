from typing import Any

import numpy as np
from numpy.typing import NDArray

from pynamicalsys.common.recurrence_quantification_analysis import (
    RTEConfig,
    build_recurrence_matrix,
    calculate_threshold,
    white_vertline_distr,
)
from pynamicalsys.common.types import int_t, map_t
from pynamicalsys.discrete_time.trajectory import generate_trajectory, iterate_mapping


rte_return_t = (
    float
    | tuple[float, ...]
    | tuple[float, NDArray[np.float64]]
    | tuple[float, NDArray[np.uint8]]
    | tuple[float, NDArray[np.float64], NDArray[np.uint8]]
    | tuple[float, NDArray[np.float64], NDArray[np.float64]]
    | tuple[float, NDArray[np.uint8], NDArray[np.float64]]
    | tuple[float, NDArray[np.float64], NDArray[np.uint8], NDArray[np.float64]]
)


def RTE(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: int_t,
    mapping: map_t,
    transient_time: int_t | None = None,
    **kwargs: Any,
) -> rte_return_t:
    """
    Compute the recurrence time entropy (RTE) of a trajectory.

    Parameters
    ----------
    u : NDArray[np.float64]
        Initial condition of shape `(system_dimension,)`.
    parameters : NDArray[np.float64]
        System parameters passed to `mapping`.
    total_time : int_t
        Total number of iterations used in the computation.
    mapping : map_t
        System mapping function.
    transient_time : int_t | None, optional
        Number of initial iterations discarded before the computation.
    **kwargs : Any
        Keyword arguments forwarded to `RTEConfig`.

    Returns
    -------
    float or tuple
        - If no optional outputs are requested, returns the scalar RTE value.
        - Otherwise returns a tuple whose first entry is the RTE value, followed
          by the requested outputs in this order:
            1. final state, if `return_final_state=True`
            2. recurrence matrix, if `return_recmat=True`
            3. white-vertical-line distribution, if `return_p=True`

    Notes
    -----
    This is a low-level helper. Input validation is expected to be handled by
    the wrapper.
    """
    u = u.copy()
    config = RTEConfig(**kwargs)

    effective_time = total_time
    if transient_time is not None:
        u = iterate_mapping(
            u=u,
            parameters=parameters,
            total_time=transient_time,
            mapping=mapping,
        )
        effective_time -= transient_time

    time_series = generate_trajectory(
        u=u,
        parameters=parameters,
        total_time=effective_time,
        mapping=mapping,
    )

    eps = calculate_threshold(time_series, config)
    recmat = build_recurrence_matrix(time_series, float(eps), metric=config.metric)

    p = white_vertline_distr(recmat, wmin=config.lmin)
    p = p[p > 0]
    p = p / p.sum()

    rte = float(-np.sum(p * np.log(p)))

    result: list[Any] = [rte]

    if config.return_final_state:
        result.append(time_series[-1, :])

    if config.return_recmat:
        result.append(recmat)

    if config.return_p:
        result.append(p)

    if len(result) == 1:
        return rte

    return tuple(result)


def finite_time_RTE(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: int_t,
    finite_time: int_t,
    mapping: map_t,
    return_points: bool = False,
    **kwargs: Any,
) -> NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute finite-time recurrence time entropy over consecutive non-overlapping windows.

    Parameters
    ----------
    u : NDArray[np.float64]
        Initial condition of shape ``(system_dimension,)``.
    parameters : NDArray[np.float64]
        System parameters passed to ``mapping``.
    total_time : int_t
        Total number of iterations used in the computation.
    finite_time : int_t
        Length of each non-overlapping window.
    mapping : map_t
        System mapping function.
    return_points : bool, optional
        If True, also return the final phase-space point of each window.
    **kwargs : Any
        Additional keyword arguments forwarded to ``RTE``.

    Returns
    -------
    NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]]
        - If ``return_points=False``, returns an array of shape ``(num_windows,)``
          containing the finite-time RTE values.
        - If ``return_points=True``, returns:
            - ``rte_values``: array of shape ``(num_windows,)``
            - ``phase_space_points``: array of shape
              ``(num_windows, system_dimension)`` containing the final point of
              each window

    Notes
    -----
    Input validation is expected to be handled by the wrapper.
    """
    u = u.copy()

    num_windows = total_time // finite_time
    rte_values = np.empty(num_windows, dtype=np.float64)
    phase_space_points = np.empty((num_windows, u.shape[0]), dtype=np.float64)

    for i in range(num_windows):
        phase_space_points[i] = u
        result = RTE(
            u=u,
            parameters=parameters,
            total_time=finite_time,
            mapping=mapping,
            transient_time=None,
            return_final_state=True,
            **kwargs,
        )

        if not isinstance(result, tuple):
            raise TypeError("RTE(return_final_state=True) must return a tuple")

        rte_values[i] = float(result[0])
        u = np.asarray(result[1], dtype=np.float64).copy()

    if return_points:
        return rte_values, phase_space_points

    return rte_values
