"""
Stability of periodic orbits: classify_stability, eigenvalues_and_eigenvectors,
and manifold, which builds on them.

None of them emits Numba's NumbaExperimentalFeatureWarning. It used to appear
because the Jacobian, a compiled function, was called from Python with the
compiled map as an argument, which Numba treats as an experimental
first-class function. Numba warns only when a function is first compiled, so
the calls run in a new Python process.
"""

import pathlib
import subprocess
import sys

SOURCE = pathlib.Path(__file__).resolve().parents[2] / "src"

# The kernel is called directly: creating a DiscreteDynamicalSystem would
# install its own warning filter and hide the warning.
SCRIPT = """
import warnings
import numpy as np
from numba.core.errors import NumbaExperimentalFeatureWarning
warnings.simplefilter("error", NumbaExperimentalFeatureWarning)

from pynamicalsys.discrete_time.models import henon_map, henon_map_jacobian
from pynamicalsys.discrete_time.stability import eigenvalues_and_eigenvectors

saddle = np.array([0.6313544770895048, 0.18940634312685142])
eigenvalues_and_eigenvectors(saddle, np.array([1.4, 0.3]), henon_map, henon_map_jacobian, 1)
"""


def test_eigenvalues_emit_no_numba_experimental_warning():
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(SOURCE), "PATH": ""},
    )
    assert result.returncode == 0, result.stderr[-2000:]
