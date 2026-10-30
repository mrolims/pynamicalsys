"""
Validation of the arguments of a custom DiscreteDynamicalSystem.
"""

import pytest
from numba import njit

from pynamicalsys import DiscreteDynamicalSystem as DDS


@njit
def identity(u, parameters):
    return u


@pytest.mark.parametrize("dimension", [0, -1])
def test_system_dimension_must_be_positive(dimension):
    # A map with no state variables has nothing to iterate, so it is rejected
    # when the system is built instead of failing, or not, in later methods.
    with pytest.raises(ValueError, match="system_dimension"):
        DDS(mapping=identity, system_dimension=dimension, number_of_parameters=0)


def test_system_dimension_must_be_an_integer():
    with pytest.raises(TypeError, match="system_dimension"):
        DDS(mapping=identity, system_dimension=2.0, number_of_parameters=0)


def test_one_dimensional_custom_system_is_accepted():
    system = DDS(mapping=identity, system_dimension=1, number_of_parameters=0)
    assert len(system.trajectory([0.3], 2)) == 2
