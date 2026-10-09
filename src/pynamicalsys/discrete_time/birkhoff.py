# birkhoff.py

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


import warnings
import weakref
from functools import lru_cache
from typing import Callable

import numpy as np
from numba import njit
from numba.core.registry import CPUDispatcher
from numpy.typing import NDArray

from pynamicalsys.common.types import int_t, map_t

# Forms in which an observable can be evaluated
OBSERVABLE_COMPILED = "compiled"
OBSERVABLE_PYTHON = "python"
OBSERVABLE_LEGACY = "legacy"

# Observables compiled by `resolve_observable`, so that repeated calls with the
# same function do not compile it again. A value of None marks a function that
# Numba cannot compile.
_compiled_observables: "weakref.WeakKeyDictionary[Callable, Callable | None]" = (
    weakref.WeakKeyDictionary()
)


@njit
def wba_weights(window_size: int_t) -> NDArray[np.float64]:
    """
    Compute the normalized weights of a weighted Birkhoff average.

    The weights are samples of the exponential bump function

        `g(t) = exp(-1 / (t (1 - t)))` for `0 < t < 1`, and `g(t) = 0` otherwise,

    at `t_n = n / N`, for `n = 0, ..., N - 1`, normalized so that they sum to
    one (Sander and Meiss, 2020, Eqs. 6-7).

    Parameters
    ----------
    window_size : int_t
        Number of weights `N`. Must be at least 2.

    Returns
    -------
    NDArray[np.float64]
        Weights of shape `(N,)`. The first weight is zero.
    """
    weights = np.zeros(window_size, dtype=np.float64)
    for n in range(1, window_size):
        t = n / window_size
        weights[n] = np.exp(-1.0 / (t * (1.0 - t)))

    return weights / weights.sum()


@njit
def two_window_wba(values: NDArray[np.float64], window_size: int_t) -> tuple[float, float]:
    """
    Compute the weighted Birkhoff averages of a series over two consecutive windows.

    Parameters
    ----------
    values : NDArray[np.float64]
        Series of shape `(2 N,)`.
    window_size : int_t
        Length `N` of each window.

    Returns
    -------
    tuple[float, float]
        Weighted Birkhoff averages over `values[:N]` and `values[N:2N]`.
    """
    weights = wba_weights(window_size)
    wb0 = 0.0
    wb1 = 0.0
    for n in range(window_size):
        wb0 += weights[n] * values[n]
        wb1 += weights[n] * values[window_size + n]

    return wb0, wb1


@njit
def orbit_states(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    number_of_states: int_t,
    mapping: map_t,
) -> NDArray[np.float64]:
    """
    Store the states `x_0, ..., x_{M-1}` of an orbit, starting at `x_0 = u`.

    Parameters
    ----------
    u : NDArray[np.float64]
        Initial condition of shape `(d,)`.
    parameters : NDArray[np.float64]
        Parameter array passed to `mapping`.
    number_of_states : int_t
        Number of states `M`.
    mapping : map_t
        Function defining the time evolution of the system.

    Returns
    -------
    NDArray[np.float64]
        States of shape `(M, d)`.
    """
    states = np.empty((number_of_states, u.shape[0]), dtype=np.float64)
    state = u.copy()
    for n in range(number_of_states):
        states[n] = state
        state = mapping(state, parameters)

    return states


@njit
def _evaluate_compiled(
    states: NDArray[np.float64], func: Callable
) -> NDArray[np.float64]:
    values = np.empty(states.shape[0], dtype=np.float64)
    for n in range(states.shape[0]):
        values[n] = func(states[n])

    return values


@lru_cache(maxsize=None)
def default_observable(period: float) -> Callable:
    """
    Return the default observable `cos(2 pi u[0] / period)`, compiled with Numba.

    The first coordinate is assumed to be an angle with the given period, so
    the observable is `cos(2 pi u[0])` for period one and `cos(u[0])` for
    period `2 pi`. Compiled observables are cached by period.

    Parameters
    ----------
    period : float
        Period of the angle `u[0]`. Must be positive.

    Returns
    -------
    Callable
        Compiled observable of a single state.
    """
    frequency = 2.0 * np.pi / period

    @njit
    def observable(u):
        return np.cos(frequency * u[0])

    return observable


def evaluate_observable(
    states: NDArray[np.float64], func: Callable, form: str
) -> NDArray[np.float64]:
    """
    Evaluate an observable on each state of an orbit.

    Parameters
    ----------
    states : NDArray[np.float64]
        States of shape `(M, d)`.
    func : Callable
        Observable, as returned by `resolve_observable`.
    form : str
        Form of the observable, as returned by `resolve_observable`.

    Returns
    -------
    NDArray[np.float64]
        Values of shape `(M,)`.
    """
    if form == OBSERVABLE_COMPILED:
        return _evaluate_compiled(states, func)
    if form == OBSERVABLE_PYTHON:
        return np.array([func(state) for state in states], dtype=np.float64)

    return np.asarray(func(states), dtype=np.float64)


def _returns_real_number(func: Callable, u: NDArray[np.float64]) -> bool:
    try:
        value = func(u.copy())
    except Exception:
        return False

    return np.ndim(value) == 0 and np.isrealobj(value)


def resolve_observable(
    func: Callable, u: NDArray[np.float64], allow_legacy: bool, stacklevel: int = 3
) -> tuple[Callable, str]:
    """
    Validate an observable and decide how to evaluate it.

    An observable takes one state of shape `(d,)` and returns a real number.
    It is compiled with Numba when possible, so that it can be evaluated in
    compiled code, and is evaluated as a plain Python function otherwise.

    The former form, a function of a 2D array of states of shape `(n, d)`
    returning a 1D array of shape `(n,)`, is still accepted with a
    `DeprecationWarning` when `allow_legacy` is True.

    Parameters
    ----------
    func : Callable
        Observable provided by the user.
    u : NDArray[np.float64]
        A state of shape `(d,)` used to test the observable.
    allow_legacy : bool
        Whether to accept the former form.
    stacklevel : int, optional
        Stack level of the deprecation warning. The default is 3, which
        points to the caller of the wrapper method.

    Returns
    -------
    tuple[Callable, str]
        Observable to evaluate and its form, one of `OBSERVABLE_COMPILED`,
        `OBSERVABLE_PYTHON`, or `OBSERVABLE_LEGACY`.

    Raises
    ------
    TypeError
        If `func` is not callable.
    ValueError
        If `func` does not return a real number for a single state.
    """
    if not callable(func):
        raise TypeError("func must be callable")

    if _returns_real_number(func, u):
        if isinstance(func, CPUDispatcher):
            return func, OBSERVABLE_COMPILED

        try:
            compiled = _compiled_observables[func]
        except (KeyError, TypeError):
            compiled = njit(func)
            if not _returns_real_number(compiled, u):
                compiled = None
            try:
                _compiled_observables[func] = compiled
            except TypeError:
                pass

        if compiled is not None:
            return compiled, OBSERVABLE_COMPILED
        return func, OBSERVABLE_PYTHON

    if allow_legacy:
        try:
            test_output = func(u[np.newaxis, :].copy())
        except Exception:
            test_output = None

        if (
            isinstance(test_output, np.ndarray)
            and test_output.ndim == 1
            and test_output.shape[0] == 1
        ):
            warnings.warn(
                "Passing an observable that acts on an array of states of shape "
                "(n, d) is deprecated and will stop working in a future version. "
                "Pass a function of a single state of shape (d,) that returns a "
                "real number instead, for example "
                "`lambda u: np.cos(2 * np.pi * u[0])`.",
                DeprecationWarning,
                stacklevel=stacklevel,
            )
            return func, OBSERVABLE_LEGACY

    raise ValueError(
        "func must take a single state of shape (d,) and return a real number, "
        "for example `lambda u: np.cos(2 * np.pi * u[0])`."
    )


def dig(
    u: NDArray[np.float64],
    parameters: NDArray[np.float64],
    total_time: int_t,
    mapping: map_t,
    func: Callable,
    form: str,
    transient_time: int_t | None = None,
) -> float:
    """
    Compute the `dig` indicator from weighted Birkhoff averages.

    The `dig` indicator measures the agreement between two weighted Birkhoff
    averages computed over two consecutive windows of the same trajectory,
    each of length `total_time`. It is defined as

        `dig = -log10(|WB_0 - WB_1|)`

    where `WB_0` and `WB_1` are the weighted Birkhoff averages of the
    observable over the states `x_0, ..., x_{N-1}` and `x_N, ..., x_{2N-1}`,
    respectively.

    Parameters
    ----------
    u : NDArray[np.float64]
        Initial condition of shape `(d,)`, where `d` is the system dimension.
    parameters : NDArray[np.float64]
        Parameter array passed to `mapping`.
    total_time : int_t
        Length `N` of each window.
    mapping : map_t
        Function defining the time evolution of the system.
    func : Callable
        Observable, as returned by `resolve_observable`.
    form : str
        Form of the observable, as returned by `resolve_observable`.
    transient_time : int_t | None, optional
        Number of initial iterations discarded before computing the weighted
        Birkhoff averages. If None, no transient is discarded.

    Returns
    -------
    float
        Value of the `dig` indicator.

    Notes
    -----
    This function assumes that input validation is handled by the wrapper, and
    in particular that `total_time >= 2`.
    """
    u = u.copy()

    if transient_time is not None:
        for _ in range(transient_time):
            u = mapping(u, parameters)

    states = orbit_states(u, parameters, 2 * total_time, mapping)
    values = evaluate_observable(states, func, form)
    wb0, wb1 = two_window_wba(values, total_time)

    with np.errstate(divide="ignore"):
        return float(-np.log10(abs(wb0 - wb1)))
