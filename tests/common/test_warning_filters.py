"""
Creating a dynamical system leaves the warning filters of the session alone.
The constructors of DiscreteDynamicalSystem and ContinuousDynamicalSystem used
to add a filter that silenced NumbaExperimentalFeatureWarning for the whole
program.
"""

import warnings

import pytest

from pynamicalsys import ContinuousDynamicalSystem as CDS
from pynamicalsys import DiscreteDynamicalSystem as DDS
from pynamicalsys import HamiltonianSystem as HS


@pytest.mark.parametrize(
    "create",
    [
        lambda: DDS(model="henon map"),
        lambda: CDS(model="lorenz system"),
        lambda: HS(model="henon heiles"),
    ],
)
def test_creating_a_system_does_not_change_the_warning_filters(create):
    with warnings.catch_warnings():
        before = list(warnings.filters)
        create()
        assert warnings.filters == before
