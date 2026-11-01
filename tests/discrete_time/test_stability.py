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

# The kernel is called directly, as the public methods do, without any
# other code that could install a warning filter first.
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


def test_a_map_without_a_jacobian_uses_finite_differences():
    # A system always has a Jacobian: without one, finite differences are used.
    import numpy as np
    from numba import njit

    from pynamicalsys import DiscreteDynamicalSystem as DDS

    @njit
    def henon(u, parameters):
        a, b = parameters
        return np.array([1.0 - a * u[0] ** 2 + u[1], b * u[0]])

    saddle = [0.6313544770895048, 0.18940634312685142]
    custom = DDS(mapping=henon, system_dimension=2, parameters=[1.4, 0.3])
    builtin = DDS(model="henon map")
    builtin.set_parameters([1.4, 0.3])

    stability = custom.classify_stability(saddle, 1)
    assert stability["classification"] == builtin.classify_stability(saddle, 1)["classification"]
    np.testing.assert_allclose(
        custom.eigenvalues_and_eigenvectors(saddle, 1)[0],
        builtin.eigenvalues_and_eigenvectors(saddle, 1)[0],
        rtol=1e-6,
    )
