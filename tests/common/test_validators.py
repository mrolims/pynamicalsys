"""
The shared validators reject booleans where a number is expected. `bool` is a
subclass of `int`, so `True` would otherwise pass as 1.
"""

from numbers import Integral, Real

import pytest

from pynamicalsys import DiscreteDynamicalSystem as DDS
from pynamicalsys.common.validators import validate_non_negative, validate_positive


@pytest.mark.parametrize("validator", [validate_positive, validate_non_negative])
@pytest.mark.parametrize("type_", [Integral, Real])
@pytest.mark.parametrize("value", [True, False])
def test_booleans_are_rejected(validator, type_, value):
    with pytest.raises(TypeError, match="x must be of type"):
        validator(value, "x", type_)


def test_a_public_method_rejects_a_boolean_positive_argument():
    ds = DDS(model="henon map")
    ds.set_parameters([1.4, 0.3])
    with pytest.raises(TypeError, match="finite_time"):
        ds.finite_time_lyapunov([0.1, 0.1], 100, True)
