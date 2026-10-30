"""
The optimized discrete kernels against straightforward reference versions.

The references below are the earlier implementations, which allocate at every
step or reduce the ensemble in parallel. The optimized kernels must give the
same bits on two-dimensional maps, and agree to round-off in higher
dimensions, where the order of the sums in `J @ v` may differ from BLAS.
"""

import warnings

import numba
import numpy as np
import pytest
from numba import njit

from pynamicalsys import DiscreteDynamicalSystem as DDS
from pynamicalsys.common.linalg import qr
from pynamicalsys.discrete_time.models import (
    standard_map,
    standard_map_jacobian,
    symplectic_map_4D,
    symplectic_map_4D_jacobian,
    unbounded_standard_map,
)
from pynamicalsys.discrete_time.gali import gali_k
from pynamicalsys.discrete_time.lyapunov import lyapunov_qr
from pynamicalsys.discrete_time.sali import sali
from pynamicalsys.discrete_time.transport import (
    average_vs_time,
    cumulative_average_vs_time,
    mean_squared_displacement,
    root_mean_squared,
)

STANDARD = (standard_map, standard_map_jacobian, np.array([1.5]))
SYMPLECTIC_4D = (
    symplectic_map_4D,
    symplectic_map_4D_jacobian,
    np.array([0.5, 0.1, 0.001]),
)
U_2D = [np.array([0.05, 0.05]), np.array([0.5, 0.25])]
U_4D = np.array([0.5, 0.1, 0.3, 0.2])


@njit
def reference_lyapunov_qr(u, parameters, total_time, mapping, jacobian, k):
    np.random.seed(1312)
    v = np.ascontiguousarray(np.random.rand(len(u), k))
    v, _ = qr(v)
    exponents = np.zeros(k)
    for _ in range(total_time):
        u = np.ascontiguousarray(mapping(u, parameters))
        J = np.ascontiguousarray(jacobian(u, parameters, mapping))
        for i in range(k):
            v[:, i] = J @ np.ascontiguousarray(v[:, i])
        v, R = qr(v)
        exponents += np.log(np.abs(np.diag(R)))
    return exponents / total_time


@njit
def reference_sali(u, parameters, total_time, mapping, jacobian):
    np.random.seed(1312)
    v = np.ascontiguousarray(np.random.rand(len(u), 2))
    v, _ = qr(v)
    history = np.empty(total_time)
    for n in range(total_time):
        u = mapping(u, parameters)
        J = np.ascontiguousarray(jacobian(u, parameters, mapping))
        for i in range(2):
            v[:, i] = J @ np.ascontiguousarray(v[:, i])
            v[:, i] /= np.linalg.norm(v[:, i])
        history[n] = min(
            np.linalg.norm(v[:, 0] + v[:, 1]), np.linalg.norm(v[:, 0] - v[:, 1])
        )
    return history


@njit
def reference_gali_qr(u, parameters, total_time, mapping, jacobian, k):
    np.random.seed(1312)
    v = np.random.rand(len(u), k)
    v, _ = np.linalg.qr(v)
    history = np.empty(total_time)
    for n in range(total_time):
        u = mapping(u, parameters)
        J = np.ascontiguousarray(jacobian(u, parameters, mapping))
        for i in range(k):
            v[:, i] = J @ np.ascontiguousarray(v[:, i])
            v[:, i] = v[:, i] / np.linalg.norm(v[:, i])
        v = np.ascontiguousarray(v)
        _, R = qr(v)
        history[n] = np.exp(np.sum(np.log(np.abs(np.diag(R)))))
    return history


@njit
def reference_transport(u, parameters, sample_times, mapping, axis):
    # Average, cumulative average, RMS, and MSD, reduced sequentially.
    states = u.copy()
    sums = np.zeros(u.shape[0])
    squares = np.zeros(u.shape[0])
    initial = u[:, axis].copy()
    out = np.empty((4, sample_times.shape[0]))
    time = 0
    for k in range(sample_times.shape[0]):
        while time < sample_times[k]:
            for i in range(u.shape[0]):
                states[i] = mapping(states[i], parameters)
            sums += states[:, axis]
            squares += states[:, axis] ** 2
            time += 1
        st = sample_times[k]
        out[0, k] = np.mean(states[:, axis])
        out[1, k] = np.mean(sums / st)
        out[2, k] = np.sqrt(np.mean(squares / st))
        out[3, k] = np.mean((states[:, axis] - initial) ** 2)
    return out


@pytest.mark.parametrize("u", U_2D)
def test_lyapunov_qr_is_bit_identical_in_2d(u):
    mapping, jacobian, parameters = STANDARD
    result, _ = lyapunov_qr(u.copy(), parameters, 5_000, mapping, jacobian, 2)
    expected = reference_lyapunov_qr(u.copy(), parameters, 5_000, mapping, jacobian, 2)
    np.testing.assert_array_equal(result.ravel(), expected)


@pytest.mark.parametrize("u", U_2D)
def test_sali_is_bit_identical_in_2d(u):
    mapping, jacobian, parameters = STANDARD
    sample_times = np.arange(1, 5_001, dtype=np.int64)
    result, _ = sali(
        u.copy(), parameters, 5_000, mapping, jacobian, sample_times,
        return_history=True, tol=0.0,
    )
    expected = reference_sali(u.copy(), parameters, 5_000, mapping, jacobian)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("u", U_2D)
def test_gali_qr_is_bit_identical_in_2d(u):
    mapping, jacobian, parameters = STANDARD
    sample_times = np.arange(1, 5_001, dtype=np.int64)
    result, _ = gali_k(
        u.copy(), parameters, 5_000, mapping, jacobian, 2, sample_times,
        method="QR", return_history=True, tol=0.0, seed=1312,
    )
    expected = reference_gali_qr(u.copy(), parameters, 5_000, mapping, jacobian, 2)
    np.testing.assert_array_equal(result, expected)


def test_tangent_kernels_agree_to_round_off_in_4d():
    mapping, jacobian, parameters = SYMPLECTIC_4D
    sample_times = np.arange(1, 2_001, dtype=np.int64)

    result, _ = lyapunov_qr(U_4D.copy(), parameters, 2_000, mapping, jacobian, 4)
    expected = reference_lyapunov_qr(U_4D.copy(), parameters, 2_000, mapping, jacobian, 4)
    np.testing.assert_allclose(result.ravel(), expected, rtol=0, atol=1e-14)

    result, _ = sali(
        U_4D.copy(), parameters, 2_000, mapping, jacobian, sample_times,
        return_history=True, tol=0.0,
    )
    expected = reference_sali(U_4D.copy(), parameters, 2_000, mapping, jacobian)
    np.testing.assert_allclose(result, expected, rtol=1e-10)

    result, _ = gali_k(
        U_4D.copy(), parameters, 2_000, mapping, jacobian, 3, sample_times,
        method="QR", return_history=True, tol=0.0, seed=1312,
    )
    expected = reference_gali_qr(U_4D.copy(), parameters, 2_000, mapping, jacobian, 3)
    np.testing.assert_allclose(result, expected, rtol=1e-8)


TRANSPORT = [
    average_vs_time,
    cumulative_average_vs_time,
    root_mean_squared,
    mean_squared_displacement,
]


@pytest.mark.parametrize("axis", [0, 1])
def test_transport_matches_a_sequential_reduction(axis):
    u = np.random.default_rng(0).uniform(size=(37, 2))
    parameters = np.array([5.0])
    sample_times = np.array([1, 2, 17, 300, 301, 1_000], dtype=np.int64)
    expected = reference_transport(
        u, parameters, sample_times, unbounded_standard_map, axis
    )
    for row, kernel in enumerate(TRANSPORT):
        result = kernel(
            u, parameters, 1_000, unbounded_standard_map, sample_times, axis
        )
        np.testing.assert_array_equal(result, expected[row])


def test_transport_does_not_depend_on_the_number_of_threads():
    u = np.random.default_rng(1).uniform(size=(500, 2))
    parameters = np.array([5.0])
    threads = numba.get_num_threads()
    try:
        numba.set_num_threads(1)
        serial = [
            kernel(u, parameters, 300, unbounded_standard_map, None, 1)
            for kernel in TRANSPORT
        ]
        numba.set_num_threads(threads)
        parallel = [
            kernel(u, parameters, 300, unbounded_standard_map, None, 1)
            for kernel in TRANSPORT
        ]
    finally:
        numba.set_num_threads(threads)
    for a, b in zip(serial, parallel):
        np.testing.assert_array_equal(a, b)


def test_transport_is_unchanged_across_chunks():
    # With 2**17 initial conditions, a chunk holds 2**20 / 2**17 = 8 sample
    # times, so the 25 samples below are processed in four chunks.
    u = np.random.default_rng(2).uniform(size=(2**17, 2))
    parameters = np.array([5.0])
    sample_times = np.arange(1, 50, 2, dtype=np.int64)
    expected = reference_transport(
        u, parameters, sample_times, unbounded_standard_map, 1
    )
    for row, kernel in enumerate(TRANSPORT):
        result = kernel(u, parameters, 50, unbounded_standard_map, sample_times, 1)
        np.testing.assert_array_equal(result, expected[row])


def test_bifurcation_diagram_records_the_requested_coordinate():
    ds = DDS(model="henon map")
    ds.set_parameters([1.4, 0.3])
    values, results, last = ds.bifurcation_diagram(
        [0.1, 0.1], 0, np.linspace(1.0, 1.4, 5), 300, transient_time=200,
        continuation=True, return_last_state=True, observable_index=1,
    )
    state = np.array([0.1, 0.1])
    for i, a in enumerate(values):
        ds.set_parameters([a, 0.3])
        trajectory = ds.trajectory(state, 300, transient_time=200)
        np.testing.assert_array_equal(results[i], trajectory[:, 1])
        state = trajectory[-1]
    np.testing.assert_array_equal(last, state)


def test_dig_and_finite_time_lyapunov_with_a_transient():
    ds = DDS(model="standard map")
    ds.set_parameters(1.5)
    u = [0.5, 0.25]
    start = ds.trajectory(u, 1_234)[-1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert ds.dig(u, 500, transient_time=1_234) == ds.dig(start, 500)
    np.testing.assert_array_equal(
        ds.finite_time_lyapunov(u, 1_234 + 1_000, 100, transient_time=1_234),
        ds.finite_time_lyapunov(start, 1_000, 100),
    )
