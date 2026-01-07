# validators.py

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

import numpy as np
from numbers import Integral, Real
from numpy.typing import NDArray
from typing import Optional, Sequence, Tuple, List


def validate_initial_conditions(
    u, system_dimension, allow_ensemble=True
) -> NDArray[np.float64]:
    """
    Validate and format the initial condition(s).

    Parameters
    ----------
    u : scalar, 1D or 2D array
        Initial condition(s).
    system_dimension : int
        Expected number of variables in the system.
    allow_ensemble : bool, optional
        Whether 2D array of ICs (ensemble) is allowed. Default is True.

    Returns
    -------
    u : np.ndarray
        Validated and contiguous copy of initial conditions.

    Raises
    ------
    ValueError
        If `u` is not a scalar, 1D, or 2D array, or if its shape does not match
        the expected system dimension.
        If `u` is a 1D array but its length does not match the system dimension,
        or if `u` is a 2D array but does not match the expected shape for an ensemble.
    TypeError
        If `u` is not a scalar or array-like type.

    """
    if np.isscalar(u):
        u = np.array([u], dtype=np.float64)
    else:
        u = np.asarray(u, dtype=np.float64)
        if u.ndim not in (1, 2):
            raise ValueError("Initial condition must be 1D or 2D array")

    u = np.ascontiguousarray(u).copy()

    if u.ndim == 1:
        if len(u) != system_dimension:
            raise ValueError(
                f"1D initial condition must have length {system_dimension}"
            )
    elif u.ndim == 2:
        if not allow_ensemble:
            raise ValueError(
                "Ensemble of initial conditions not allowed in this context"
            )
        if u.shape[1] != system_dimension:
            raise ValueError(
                f"Each initial condition must have length {system_dimension}"
            )
    return u


def validate_parameters(parameters, number_of_parameters) -> NDArray[np.float64]:
    """
    Validate and standardize parameter vector.

    Parameters
    ----------
    parameters : scalar, 1D array-like, or None
        The parameter values to validate. `None` is allowed if number_of_parameters == 0.
    number_of_parameters : int
        The required number of parameters (defined by the system).

    Returns
    -------
    parameters : np.ndarray
        Validated 1D parameter array (empty if no parameters are required and input is None).

    Raises
    ------
    ValueError
        If `parameters` is not None and does not match the expected number of parameters.
        If `parameters` is None but the system expects parameters.
        If `parameters` is a scalar or array-like but not 1D.
    TypeError
        If `parameters` is not a scalar or array-like type.
    """
    if parameters is None:
        if number_of_parameters != 0:
            raise ValueError(
                f"This system expects {number_of_parameters} parameter(s), but got None."
            )
    else:
        if np.isscalar(parameters):
            parameters = np.array([parameters], dtype=np.float64)
        else:
            parameters = np.asarray(parameters, dtype=np.float64)
            if parameters.ndim != 1:
                raise ValueError(
                    f"`parameters` must be a 1D array or scalar. Got shape {parameters.shape}."
                )

        if number_of_parameters == 0:
            if parameters is not None and parameters.size != 0:
                raise ValueError("This system does not expect any parameters.")
            return np.array([0], dtype=np.float64)

        if parameters.size != number_of_parameters:
            raise ValueError(
                f"Expected {number_of_parameters} parameter(s), but got {parameters.size}."
            )

    return parameters


def validate_non_negative(value, name, type_=Integral) -> None:
    """Ensure value is non-negative of specified type.

    Parameters
    ----------
    value : Any
        The value to validate.
    name : str
        The name of the value for error messages.
    type_ : type, optional
        The expected type of the value (default is int).
    Raises
    ------
    TypeError
        If value is not of the expected type.
    ValueError
        If value is negative.
    """
    if not isinstance(value, type_):
        raise TypeError(f"{name} must be of type {type_.__name__}")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def validate_positive(value, name, type_=Integral) -> None:
    """Ensure value is >= 1 and of specified type.

    Parameters
    ----------
    value : Any
        The value to validate.
    name : str
        The name of the value for error messages.
    type_ : type, optional
        The expected type of the value (default is int).
    Raises
    ------
    TypeError
        If value is not of the expected type.
    ValueError
        If value is less than 1.
    """
    if not isinstance(value, type_):
        raise TypeError(f"{name} must be of type {type_.__name__}")
    if value < 1:
        raise ValueError(f"{name} must be greater than or equal to 1")


def validate_transient_time(transient_time, total_time, type_=Integral) -> None:
    """Ensure transient_time is valid relative to total_time.
    Parameters
    ----------
    transient_time : int
        The transient time to validate.
    total_time : int
        The total time of the simulation.
    Raises
    ------
    TypeError
        If transient_time is not int.
    ValueError
        If transient_time is negative.
        If transient_time is greater than or equal to total_time.
    """

    if transient_time is not None:
        validate_non_negative(transient_time, "transient_time", type_=type_)

        if transient_time >= total_time:
            raise ValueError("transient_time must be less than total_time")


def validate_axis(axis, system_dimension):
    """
    Validate that axis is a non-negative integer within system dimension bounds.

    Parameters
    ----------
    axis : int
        Axis index to validate.
    system_dimension : int
        The number of dimensions in the system.

    Raises
    ------
    TypeError
        If axis is not an integer.
    ValueError
        If axis is not in [0, system_dimension - 1].
    """
    if not isinstance(axis, Integral):
        raise TypeError("axis must be an integer")
    if axis < 0 or axis >= system_dimension:
        raise ValueError(f"axis must be in the range [0, {system_dimension - 1}]")


def validate_clv_subspaces(
    subspaces,
    system_dimension: int,
):
    """
    Validate CLV subspace specifications.

    Accepts either:
        - a single subspace split: ([...], [...])
        - a sequence of subspace splits: [([...], [...]), ...]

    Returns
    -------
    tuple of ((tuple[int], tuple[int]), ...)
        Canonicalized and validated subspace specifications.
    """

    if subspaces is None:
        return None

    # --- Normalize: allow a single (A, B) without wrapping ---
    if (
        isinstance(subspaces, (list, tuple))
        and len(subspaces) == 2
        and isinstance(subspaces[0], (list, tuple))
        and isinstance(subspaces[1], (list, tuple))
        and not (
            len(subspaces) > 0
            and isinstance(subspaces[0], (list, tuple))
            and len(subspaces) > 0
            and isinstance(subspaces[0][0], (list, tuple))
        )
    ):
        subspaces = [subspaces]

    validated = []

    for A, B in subspaces:
        A = tuple(int(i) for i in A)
        B = tuple(int(i) for i in B)

        if len(A) == 0 or len(B) == 0:
            raise ValueError(
                f"Invalid subspace split {A}, {B}: subspaces must be non-empty."
            )

        if set(A) & set(B):
            raise ValueError(
                f"Invalid subspace split {A}, {B}: overlapping CLV indices."
            )

        for i in (*A, *B):
            if i < 0 or i >= system_dimension:
                raise ValueError(
                    f"Invalid CLV index {i} for system_dimension={system_dimension}."
                )

        validated.append((A, B))

    return tuple(validated)


def validate_clv_pairs(
    pairs,
    system_dimension: int,
) -> Optional[Tuple[Tuple[int, int], ...]]:
    """
    Validate pairwise CLV angle specifications.

    Accepts either:
        - a single pair (i, j)
        - a sequence of pairs [(i, j), (k, l), ...]

    Returns
    -------
    tuple of (int, int)
        Canonicalized and validated CLV index pairs.
    """

    if pairs is None:
        return None

    # --- Normalize: allow a single (i, j) without wrapping ---
    if (
        isinstance(pairs, (list, tuple))
        and len(pairs) == 2
        and all(isinstance(x, (int, np.integer)) for x in pairs)
    ):
        pairs = [pairs]

    validated = []

    for i, j in pairs:
        i = int(i)
        j = int(j)

        if i == j:
            raise ValueError(f"Invalid CLV pair ({i}, {j}): indices must be distinct.")

        if i < 0 or j < 0 or i >= system_dimension or j >= system_dimension:
            raise ValueError(
                f"Invalid CLV pair ({i}, {j}) for system_dimension={system_dimension}."
            )

        # canonical ordering
        validated.append((i, j) if i < j else (j, i))

    return tuple(validated)
