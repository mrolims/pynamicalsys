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
