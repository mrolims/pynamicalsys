"""
Seeding of the invariant manifolds of a saddle.

The seeds of each branch span one fundamental domain along the eigendirection,
at distances delta * s**(i / n) from the saddle, where s is the stretching
factor over one period. With a small delta the map is linear around the saddle,
so after one period seed i sits at distance delta * s**(1 + i / n) along the
same direction, and the distances can be checked against that closed form.
"""

import numpy as np
import pytest

from pynamicalsys import DiscreteDynamicalSystem as DDS

SADDLE = np.array([0.0, 0.0])
DELTA = 1e-8
N_POINTS = 20
ITER_TIME = 3


def distance_to_saddle(points):
    # The standard map is taken modulo 1 in both coordinates, so points on the
    # negative side of the saddle at the origin appear near 1.
    difference = (points - SADDLE + 0.5) % 1.0 - 0.5
    return np.linalg.norm(difference, axis=1)


@pytest.fixture
def system():
    ds = DDS(model="standard map")
    ds.set_parameters([1.5])
    return ds


@pytest.mark.parametrize("stability", ["unstable", "stable"])
def test_seeds_span_one_fundamental_domain(system, stability):
    eigenvalues = system.classify_stability(SADDLE, 1)["eigenvalues"]
    stretching = (
        abs(eigenvalues[0]) if stability == "unstable" else 1 / abs(eigenvalues[1])
    )

    branches = system.manifold(
        SADDLE, 1, delta=DELTA, n_points=N_POINTS, iter_time=ITER_TIME,
        stability=stability,
    )

    expected = DELTA * stretching ** (1 + np.arange(N_POINTS) / N_POINTS)
    for branch in branches:
        # Rows are grouped by seed, with iterates 1, ..., ITER_TIME of each.
        first_images = branch[::ITER_TIME]
        distances = distance_to_saddle(first_images)
        np.testing.assert_allclose(distances, expected, rtol=1e-6)


def test_the_saddle_itself_is_not_a_seed(system):
    for branch in system.manifold(SADDLE, 1, n_points=50, iter_time=5):
        distances = distance_to_saddle(branch)
        assert distances.min() > 1e-4


def test_kernel_requires_a_backward_map_for_the_stable_manifold():
    # The kernel needs the backward map only for the stable manifold, and must
    # say so when it is missing instead of failing inside the compiled code.
    from pynamicalsys.discrete_time.manifolds import calculate_manifolds
    from pynamicalsys.discrete_time.models import henon_map, henon_map_jacobian

    saddle = np.array([0.6313544770895048, 0.18940634312685142])
    parameters = np.array([1.4, 0.3])
    with pytest.raises(ValueError, match="backward"):
        calculate_manifolds(
            saddle, parameters, henon_map, None, henon_map_jacobian, 1, stability="stable"
        )
    # The unstable manifold does not use it.
    branches = calculate_manifolds(
        saddle, parameters, henon_map, None, henon_map_jacobian, 1, stability="unstable"
    )
    assert len(branches) == 2


def henon_map(u, parameters):
    a, b = parameters
    return np.array([1.0 - a * u[0] ** 2 + u[1], b * u[0]])


def henon_map_jacobian(u, parameters, *args):
    a, b = parameters
    return np.array([[-2.0 * a * u[0], 1.0], [b, 0.0]])


HENON_SADDLE = [0.6313544770895048, 0.18940634312685142]


@pytest.mark.parametrize("caught", [ValueError, RuntimeError])
def test_missing_backward_map_raises_a_value_error_that_is_also_a_runtime_error(caught):
    # ValueError, as for the other missing or invalid arguments; RuntimeError,
    # raised until now, so that existing code catching it keeps working.
    ds = DDS(mapping=henon_map, jacobian=henon_map_jacobian, system_dimension=2, parameters=[1.4, 0.3])
    with pytest.raises(caught, match="backward"):
        ds.manifold(HENON_SADDLE, 1, stability="stable")


# n_points and iter_time are an integer, or a tuple or list of two integers.


@pytest.mark.parametrize("name", ["n_points", "iter_time"])
@pytest.mark.parametrize("value", [(2.7, 3), (True, 5), True, ("5", 5), 5.0])
def test_counts_must_be_integers(system, name, value):
    with pytest.raises(TypeError, match=name):
        system.manifold(SADDLE, 1, **{name: value})


@pytest.mark.parametrize("name", ["n_points", "iter_time"])
def test_counts_accept_a_list(system, name):
    from_list = system.manifold(SADDLE, 1, **{name: [5, 6]})
    from_tuple = system.manifold(SADDLE, 1, **{name: (5, 6)})
    for a, b in zip(from_list, from_tuple):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("name", ["n_points", "iter_time"])
def test_whole_number_floats_are_deprecated(system, name):
    with pytest.warns(DeprecationWarning, match=f"{name}.*future version"):
        from_floats = system.manifold(SADDLE, 1, **{name: (5.0, 6.0)})
    from_ints = system.manifold(SADDLE, 1, **{name: (5, 6)})
    for a, b in zip(from_floats, from_ints):
        np.testing.assert_array_equal(a, b)
