"""
Shared test configuration.

pytest imports this file automatically for every test in this directory and
below. You never import it yourself, and anything defined here as a fixture is
available to any test that names it as an argument.
"""

import numpy as np
import pytest


@pytest.fixture
def rng():
    """
    A seeded random number generator.

    Every test that asks for `rng` gets a generator seeded identically, so a
    failure is always reproducible. Using `default_rng` rather than the global
    `np.random` state keeps tests independent of each other and of any seeding
    the library itself performs.
    """
    return np.random.default_rng(20260731)
