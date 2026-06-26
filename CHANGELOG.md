# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `common.types`:
  - Added `system_func_t` type alias for the polymorphic per-step callables
    (gradients, Hessians, equations of motion) consumed by Hamiltonian
    integrators, since their argument count and meaning depend on which
    integrator they are paired with.

- `hamiltonian_systems` module:
  - Added support for the **implicit midpoint method**, a symplectic
    integrator applicable to general (including non-separable) Hamiltonian
    systems `H(q, p)`, specified via the full equations of motion `eom`
    and the Hessian of `H` with respect to the combined state `(q, p)`,
    `hess_H`.
  - `fixed_step.py`:
    - Added `implicit_midpoint_step`, advancing the trajectory `(q, p)`
      via a Newton-solved implicit midpoint update.
  - `tangent.py`:
    - Added `implicit_midpoint_step_traj_tan`, jointly advancing the
      trajectory and a set of tangent (deviation) vectors in a single
      implicit midpoint step, by reusing the Jacobian from the converged
      midpoint Newton solve to update the deviation vectors via the
      discrete-map monodromy without forming the full monodromy matrix.
    - Added `advance_block_imp`, mirroring `advance_block` (now
      `advance_block_sep`) for the implicit midpoint stepper.
  - `poincare.py`:
    - Added `generate_poincare_section_midpoint` and
      `ensemble_poincare_section_midpoint` for Poincaré-section generation
      under the implicit midpoint integrator.
    - Added `generate_poincare_section_from_traj_imp`, extracting
      Poincaré-section crossings from precomputed trajectories using the
      full equations of motion for the crossing-velocity check, required
      for non-separable systems.
  - `lyapunov.py`, `clv.py`, `sali.py`, `ldi.py`, `gali.py`:
    - Added `_imp`-suffixed variants of the Lyapunov spectrum, largest
      Lyapunov exponent, CLV, CLV-angle, SALI, LDI, and GALI computations,
      supporting the implicit midpoint integrator via `eom`/`hess_H` and
      Newton-solver `tol`/`max_iter` controls.

- `HamiltonianSystem` class:
  - Added support for constructing general (possibly non-separable)
    Hamiltonian systems via new `eom` and `hess_H` constructor parameters,
    integrated with the implicit midpoint method.
  - Added `"imp"` as a selectable integrator in `integrator()`, with `tol`
    and `max_iter` parameters controlling the underlying Newton solve.
  - `lyapunov`, `CLV`, `CLV_angles`, `SALI`, `LDI`, and `GALI` now dispatch
    to the implicit midpoint backend automatically when `"imp"` is the
    active integrator.

- `BasinMetrics` class:
  - Added `uncertainty_fraction_mapping`, a Monte Carlo estimator for the
    uncertainty fraction and uncertainty exponent of escape basins in
    discrete-time dynamical systems.
  - Supports estimating the scaling law
    `f(ε) ~ ε^α` by sampling random initial conditions and perturbed
    neighborhoods, classifying points as uncertain when a user-defined
    fraction of perturbations reach different exits.
  - Supports parallel evaluation over ε values through `joblib`, with
    reproducible random sampling controlled by a user-provided seed.

### Changed

- `common.types`:
  - Widened `symplectic_step_t` and `symplectic_tangent_step_t` to
    variadic `Callable` aliases, since the explicit symplectic steppers
    (velocity Verlet, fourth-order Yoshida) and the implicit midpoint
    stepper genuinely differ in argument count and cannot share one fixed
    positional signature.

- `hamiltonian_systems` module:
  - Renamed the separable-only implementations to `_sep`-suffixed names
    for symmetry with their new `_imp` counterparts, including
    `velocity_verlet_2nd_step`/`yoshida_4th_step`'s gradient/Hessian
    parameters (now typed `system_func_t`), `generate_poincare_section`
    (now split into `generate_poincare_section_sep` and
    `generate_poincare_section_midpoint`), `advance_block` (now
    `advance_block_sep`), `lyapunov_spectrum`/`largest_lyapunov_exponent`,
    `compute_clvs`/`clv_angles`, `sali`, `ldi_k`, and `gali_k`.
  - `generate_trajectory` and `ensemble_trajectories` (`trajectory.py`)
    now take generic `system_func_1`/`system_func_2` parameters in place
    of `grad_T`/`grad_V`, since these slots hold different callables
    depending on the active integrator.
  - `recurrence_time_entropy` (`rte.py`) and `hurst_exponent_wrapped`
    (`hurst.py`) gained a `pss_func` parameter selecting which
    Poincaré-section generator to call, and `tol`/`max_iter` parameters
    forwarded to it.

- `HamiltonianSystem` class:
  - Updated the class docstring to describe both the separable
    (`grad_T`/`grad_V`/`hess_T`/`hess_V`) and general/non-separable
    (`eom`/`hess_H`) ways of specifying a system, including the required
    `(qdot, pdot)` return order for `eom`. The class previously documented
    only separable Hamiltonians.
  - `poincare_section`, `recurrence_time_entropy`, and `hurst_exponent`
    now select the separable or implicit-midpoint Poincaré-section
    generator based on the active integrator, rather than always using
    the separable one.
  - `lyapunov`'s Hessian-requirement check is now scoped to the
    velocity-Verlet and Yoshida integrators, since the implicit midpoint
    integrator uses `hess_H` instead of `hess_T`/`hess_V`.

[Unreleased]: https://github.com/mrolims/pynamicalsys/compare/v1.5.4...HEAD

## [v1.5.4] - 2026-05-28

### Added

- `DiscreteDynamicalSystem` class:
  - Added `return_last_state` to the `SALI`, `LDI`, and `GALI` methods.
  - Added `method` option to `GALI` with the following implementations:
    - `"DET"`: computes `GALI_k` from the Gram matrix determinant.
    - `"QR"`: computes `GALI_k` from the diagonal of the triangular factor returned by the internal QR routine.
    - `"QR_HH"`: computes `GALI_k` from the diagonal of the triangular factor returned by `numpy.linalg.qr`.

- `ContinuousDynamicalSystem` class:
  - Added `method` option to `GALI` with the following implementations:
    - `"DET"`: computes `GALI_k` from the Gram matrix determinant.
    - `"QR"`: computes `GALI_k` from the diagonal of the triangular factor returned by the internal QR routine.
    - `"QR_HH"`: computes `GALI_k` from the diagonal of the triangular factor returned by `numpy.linalg.qr`.

- `HamiltonianSystem` class:
  - Added `method` option to `GALI` with the following implementations:
    - `"DET"`: computes `GALI_k` from the Gram matrix determinant.
    - `"QR"`: computes `GALI_k` from the diagonal of the triangular factor returned by the internal QR routine.
    - `"QR_HH"`: computes `GALI_k` from the diagonal of the triangular factor returned by `numpy.linalg.qr`.

- `discrete_time` module:
  - Added dedicated low-level modules:
    - `trajectory.py`
    - `bifurcation.py`
    - `birkhoff.py`
    - `hurst.py`
    - `rte.py`
    - `periodic_orbits.py`
    - `stability.py`
    - `manifolds.py`
    - `rotation.py`
    - `escape.py`
    - `transport.py`
    - `averages.py`
    - `symmetry.py`
    - `sali.py`
    - `ldi.py`
    - `gali.py`
    - `clv.py`

- `continuous_time` module:
  - Added dedicated low-level modules:
    - `fixed_step.py`
    - `adaptive_step.py`
    - `step_methods.py`
    - `variational.py`
    - `step.py`
    - `trajectory.py`
    - `poincare.py`
    - `stroboscopic.py`
    - `maxima_map.py`
    - `basins.py`
    - `lyapunov.py`
    - `sali.py`
    - `ldi.py`
    - `gali.py`
    - `clv.py`
    - `rte.py`
    - `hurst.py`

- `hamiltonian_systems` module:
  - Added dedicated low-level modules:
    - `coefficients.py`
    - `fixed_step.py`
    - `tangent.py`
    - `trajectory.py`
    - `poincare.py`
    - `lyapunov.py`
    - `sali.py`
    - `ldi.py`
    - `gali.py`
    - `clv.py`
    - `rte.py`
    - `hurst.py`

- `common.types`:
  - Added `observable_t` type alias for weighted-Birkhoff observable functions.
  - Added `flow_t` and `flow_jacobian_t` type aliases for continuous-time vector fields and their Jacobians.
  - Added `grad_t` and `hess_t` type aliases for Hamiltonian gradients and Hessians.
  - Added `symplectic_step_t` and `symplectic_tangent_step_t` type aliases for Hamiltonian fixed-step integrators and their tangent-map variants.

### Changed

- Refactored low-level discrete-time analysis routines by splitting the old monolithic modules into dedicated files for improved project organization, readability, and maintainability.
- Refactored low-level continuous-time analysis routines by splitting the old monolithic modules into dedicated files for improved project organization, readability, and maintainability.
- Refactored low-level Hamiltonian-system analysis and integration routines by splitting the old monolithic modules into dedicated files for improved project organization, readability, and maintainability.

- `DiscreteDynamicalSystem` class:
  - Updated wrappers across the discrete-time analysis API with improved type annotations, return annotations, argument validation, and docstrings.
  - Standardized handling of `sample_times` in wrappers that support sampled outputs by explicitly validating user input and constructing internal sampling arrays only when needed.
  - Updated `dig` observable validation so the observable must be callable and return a 1D NumPy array with one value per input state.
  - Improved validation and normalization of wrapper inputs for periodic-orbit, manifold, transport, escape, Hurst, and recurrence diagnostics.
  - The Lyapunov exponent API now exposes the analytical QR procedure (after Eckmann-Ruelle) explicitly through the `method` argument:
    - `method="ER"` now selects the analytical QR-based approach (available only for 2D systems).
    - `method="QR"` now consistently uses the modifed Gram-Schmidt QR decomposition, independent of system dimension.
    - `method=QR_HH` uses Householder QR decomposition via `np.linalg.qr`.
  - The previous implicit behavior where method="QR" automatically switched to the Eckmann–Ruelle algorithm for 2D systems has been removed. Existing code relying on this optimization should now explicitly set `ds.lyapunov(..., method="ER")`.

- `ContinuousDynamicalSystem` class:
  - Reorganized imports and internal plumbing to use the new continuous-time module layout.
  - Updated constructor, integrator configuration, and wrapper methods with improved type annotations, return annotations, argument validation, and docstrings.
  - Standardized validation of continuous-time arguments by using dedicated time validation helpers and more explicit normalization of parameters and wrapper inputs.
  - Updated trajectory-related wrappers to reflect adaptive-step behavior more faithfully when ensemble trajectories do not all share the same stored length.
  - Updated Lyapunov-related return annotations and docstrings so the scalar `num_exponents=1` case is documented and handled consistently.

- `HamiltonianSystem` class:
  - Reorganized imports and internal plumbing to use the new Hamiltonian-system module layout.
  - Updated constructor, integrator configuration, and wrapper methods with improved type annotations, return annotations, argument validation, and docstrings.
  - Standardized validation of Hamiltonian wrapper inputs for coordinates, momenta, parameters, section arguments, CLV configuration, and time-like arguments.
  - Updated Lyapunov-, CLV-, SALI-, LDI-, GALI-, RTE-, Hurst-, trajectory-, and Poincaré-related wrappers to use the refactored low-level modules and more explicit scalar/array return semantics.

- `continuous_time.validators`:
  - Refactored `validate_times()` to return validated `np.float64` values and to enforce stricter type and range checks for continuous-time arguments.

- `hamiltonian_systems.validators`:
  - Refactored `validate_times()` and `validate_initial_conditions()` with stricter typing, shape checks, and `np.float64` normalization for Hamiltonian-system inputs.

- `common.validators`:
  - Refactored `validate_clv_subspaces()` and `validate_clv_pairs()` to validate indices against `num_clvs` instead of the full system dimension.
  - Improved normalization of single subspace and single pair inputs into canonical tuple-based representations.

- `common.utils`:
  - Refactored the internal QR routine to a simpler reduced modified Gram-Schmidt implementation with manual inner products for better Numba compatibility and lower overhead.
  - Simplified `householder_qr()` implementation.
  - Temporarily kept shared CLV helper routines in `utils.py` because they are still used by the continuous and Hamiltonian classes during the refactor.
  - Added typed `qr_truncate()` helper documentation and return annotations for CLV-related truncation logic.

### Fixed

- `DiscreteDynamicalSystem` class:
  - Fixed `SALI`, `LDI`, and `GALI` wrappers so that scalar outputs are returned consistently when `return_history=False`, while preserving the final state when `return_last_state=True`.
  - Fixed `GALI` computation by using stable QR-based volume evaluation.
  - Fixed `CLV_angles` validation so subspace and pair indices are checked against the number of computed CLVs rather than the ambient phase-space dimension.
  - Removed a stray debug `print(iter_time)` from `manifold()`.
  - Added validation to prevent the use of `method="ER"` for systems with dimension greater than 2, raising a clear error instead of silently falling back to another implementation.

- `ContinuousDynamicalSystem` class:
  - Fixed wrapper regressions introduced during the continuous-time refactor in trajectory, reduced-map, Lyapunov, CLV, SALI, LDI, GALI, RTE, and Hurst-related methods.
  - Fixed `trajectory()` return handling for ensembles evolved with adaptive integrators, where different initial conditions may produce trajectories with different numbers of stored time steps.
  - Fixed `lyapunov()` so `method="QR_HH"` works correctly with the refactored low-level implementation.
  - Fixed `lyapunov()` so `num_exponents=1` returns a scalar instead of a length-1 array when `return_history=False`.
  - Fixed `GALI()` wrapper so the selected low-level computation method is validated and passed through correctly.

- `HamiltonianSystem` class:
  - Fixed wrapper validation bugs in trajectory, Poincaré-section, Lyapunov, CLV, CLV-angle, SALI, LDI, GALI, RTE, and Hurst-related methods introduced during the Hamiltonian refactor.
  - Fixed wrapper shape checks so coordinate and momentum arrays are validated by full shape compatibility rather than only by matching dimensionality.
  - Fixed `lyapunov()` so `num_exponents=1` returns a scalar instead of a length-1 array when `return_history=False`.
  - Fixed `GALI()` wrapper so the selected low-level computation method is validated and passed through correctly.
  - Fixed section-index validation in Hamiltonian reduced-map wrappers so section coordinates are checked against the number of degrees of freedom rather than the full phase-space dimension where appropriate.

- `discrete_time` module:
  - Fixed low-level `SALI`, `LDI`, and `GALI` implementations to return both the computed result and the final state consistently.
  - Fixed history allocation and sampling logic in low-level `SALI`, `LDI`, and `GALI` implementations for `return_history=True`.
  - Fixed the weighted-Birkhoff `dig` implementation to live in its own dedicated module while preserving wrapper behavior.
  - Fixed dtype consistency in sampled transport routines to avoid Numba typing errors from mixing `int32` and `int64` sampling arrays.

- `continuous_time` module:
  - Fixed QR-related low-level Lyapunov and GALI computations to handle Householder-based QR consistently under Numba.
  - Fixed array contiguity issues in low-level QR-based continuous-time routines to avoid reshape/type failures after `numpy.linalg.qr`.

- `hamiltonian_systems` module:
  - Fixed low-level Lyapunov and GALI computations to support both internal QR and Householder-based QR through method dispatch instead of passing unsupported callable QR objects into Numba-compiled routines.
  - Fixed low-level trajectory and reduced-map helpers with clearer typing, stricter validation, and dedicated Poincaré-section extraction from stored trajectories.
  - Fixed Hamiltonian recurrence-time-entropy returns by replacing dynamically assembled list-based returns with explicit tuple branches compatible with static typing.

- `common.validators`:
  - Fixed `validate_positive()` so zero is rejected correctly.

[v1.5.4]: https://github.com/mrolims/pynamicalsys/compare/v1.5.3...v1.5.4

## [v1.5.3] - 2026-04-08

### Fixed

- Incorrect tangent drift update in symplectic (Verlet/Yoshida) integrators (used δq instead of δp in δq update)
- QR re-orthonormalization scheduling in Lyapunov spectrum (`(i + 1) % qr_interval`)
- History allocation using `round` instead of integer division
- Incorrect normalization of history array (time column was being scaled)

[v1.5.3]: https://github.com/mrolims/pynamicalsys/compare/v1.5.2...v1.5.3

## [v1.5.2] - 2026-03-23

### Fixed

- `DiscreteDynamicalSystem` class:
  - Fixed parameter handling in `DiscreteDynamicalSystem.bifurcation_diagram()` when the system has no stored default parameters:
    - user-provided parameters are now validated and the scanned parameter is inserted at `param_index`;
    - if no parameters are provided, a default parameter array is initialized so the scanned parameter can be assigned correctly.

### Changed

- Minor formatting cleanup in `logistic_map()`.

[v1.5.2]: https://github.com/mrolims/pynamicalsys/compare/v1.5.1...v1.5.2

## [v1.5.1] - 2026-03-16

### Added

- Support for fixed recurrence rate threshold selection via `fixed_rr=True`. When enabled, the recurrence threshold is automatically chosen such that the recurrence matrix achieves the desired recurrence rate (`threshold` interpreted as the target RR).

- Support for callable distance metrics for recurrence matrix computation and recurrence-rate threshold estimation. Custom metrics must have signature `metric(x, y) -> float` and must be Numba-compatible (decorated with `@numba.njit`).

- New `return_eps` option in `TimeSeriesMetrics.recurrence_matrix()` to return the threshold value used to construct the recurrence matrix.

### Modifed

- Internal recurrence matrix computation has been refactored for improved performance and consistency between built-in metrics (supremum, euclidean, manhattan) and callable metrics.

- Threshold selection is now centralized in a new `calculate_threshold()` function, which handles
  - direct thresholds.
  - standard-deviation–scaled thresholds (`threshold_mode="std"`).
  - fixed recurrence rate thresholds (`threshold_mode="rr"`).

[v1.5.1]: https://github.com/mrolims/pynamicalsys/compare/v1.5.0...v1.5.1

## [v1.5.0] - 2026-01-07

### Added

- Added two new methods, `set_parameters` and `get_parameters`, to each of the three main classes to improve parameter management. Parameters can now be set once via `set_parameters` and stored internally, allowing subsequent method calls to use the stored values without requiring parameters to be passed explicitly each time. Existing workflows remain fully backward compatible: methods that accept parameters directly continue to work as before.

- Added **Covariant Lyapunov Vector (CLV) angle diagnostics** to all three core system classes:
  - The new methods `CLV` and `CLV_angles` allow computation of the **CLVs**, **angles between arbitrary CLV subspaces** and **pairwise CLV angles**, with full user control over:
    - which subspaces are compared,
    - which CLV pairs are measured.
  - Support for **Poincaré-section–restricted CLV angles** in Hamiltonian systems, enabling angle analysis directly on the section.
  - Angle diagnostics are based on **minimum principal angles between covariant subspaces**, providing a geometrically meaningful measure of hyperbolicity and near-tangencies in high-dimensional systems.

### Modified

- The parameter validator now accepts empty lists for systems that take no parameters.

[v1.5.0]: https://github.com/mrolims/pynamicalsys/compare/v1.4.9...v1.5.0

## [v1.4.9] - 2025-11-18

### Fixed

- Fix missing dependency:
  - Added `ipython (>=8.13,<9.0.0)` to `pyproject.toml` dependencies, since it is required by the `ContinuousDynamicalSystem` class.

[v1.4.9]: https://github.com/mrolims/pynamicalsys/compare/v1.4.1...v1.4.9

## [v1.4.6] - 2025-10-09

### Modified

- `HamiltonianSystem` class:
  - Fixed the integration of the tangent vectors that was leading to numerical instability for long integration times.

- Refactored `recurrence_time_entropy` methods and `white_vertline_distr` function to handle the minimum line length parameter more consistently.

[v1.4.6]: https://github.com/mrolims/pynamicalsys/compare/v1.4.1...v1.4.6

## [v1.4.5] - 2025-09-17

### Modified

- `DiscreteDynamicalSystem` class:
  - Fixed problems in the `finite_hurst_exponent`

- `ContinuousDynamicalSystem` and `HamiltonianSystem` classes:
  - Fixed the output of the `recurrence_time_entropy` method when `return_final_state=True`.

[v1.4.5]: https://github.com/mrolims/pynamicalsys/compare/v1.4.1...v1.4.5

## [v1.4.1] - 2025-09-15

### Added

- `HamiltonianSystem` class for simulating and analyzing continuous-time Hamiltonian systems.
  - Support for symplectic integration:
    - 2nd-order velocity–Verlet
    - 4th-order Yoshida
  - Trajectory computation and ensemble trajectories.
  - Poincaré section generation (single and ensemble).
  - Chaos indicators:
    - Lyapunov spectrum and maximum Lyapunov exponent.
    - Smaller Alignment Index (SALI).
    - Generalized Alignment Index (GALI).
    - Linear Dependence Index (LDI).
    - Recurrence time entropy (RTE).
    - Hurst exponent.

- `ContinuousDynamicalSystem` class:
  - `poincare_section` method: return the Poincaré section of a given initial condition or of an ensemble of initial conditions.
  - `stroboscopic_map` method: return the stroboscopic map of a given initial condition or of an ensemble of initial conditions.
  - `maxima_map` method: return the maxima map of a given initial condition or of an ensemble of inital conditions.
  - `basin_of_attraction` method: given an ensemble of initial conditions, detect and label the attractors in the system.
  - `recurrence_time_entropy` method: calculates the recurrence time entropy for a given initial condition using the Poincaré section, stroboscopic map, or maxima map to construct the recurrence matrix.
  - `hurst_exponent` method: calculates the Hurst exponent for a given initial condition using the Poincaré section, stroboscopic map, or maxima map.

- `TimeSeriesMetrics`:
  - `hurst_exponent` method.

### Modified

- `DiscreteDynamicalSystem` class:
  - Unified the Hurst exponent calculation into a single function.

- `ContinuousDynamicalSystem` class:
  - `lyapunov` method now uses a specific function to compute only the maximum Lyapunov exponent when `num_exponents=1`.

[v1.4.1]: https://github.com/mrolims/pynamicalsys/compare/v1.3.1...v1.4.1

## [v1.3.1] - 2025-08-24

### Modified

- Removed `cache=True` from the low level methods that was leading to cache compilation errors.

[v1.3.1]: https://github.com/mrolims/pynamicalsys/compare/v1.3.0...v1.3.1

## [v1.3.0] - 2025-08-23

### Added

- `DiscreteDynamicalSystem` class:
  - `step` method: returns the next state of the system.
  - `GALI` method: computes the generalized alignment index (GALI).

- `ContinuousDynamicalSystem` class:
  - `GALI` method that computes the generalized alignment index (GALI).

### Modified

- `DiscreteDynamicalSystem` class:
  - Improved performance when checking sampling points by avoiding repeated searches in sample_times.
  - Refactored the `lyapunov` method to allow computing only a subset of the Lyapunov spectrum.

- `ContinuousDynamicalSystem` class:
  - Unified integration step logic (previously duplicated across methods like trajectory and lyapunov_exponents) into a single step function.
  - Refactored the `lyapunov` method to allow computing only a subset of the Lyapunov spectrum.

[v1.3.0]: https://github.com/mrolims/pynamicalsys/compare/v1.2.2...v1.3.0

## [v1.2.2] - 2025-06-29

### Added

- `ContinuousDynamicalSystem` class for simulating and analyzing continuous nonlinear dynamical systems:
  - Integration using the 4th order Runge-Kutta method with fixed time step.
  - Integration using the adaptive 4th/5th order Runge-Kutta method with adaptive time step.
  - Trajectory computation.
  - Lyapunov exponents calculation.
  - The smaller aligment index (SALI) and linear dependence index (LDI) for chaos detection.

[v1.2.2]: https://github.com/mrolims/pynamicalsys/compare/v1.0.0...v1.2.2

## v1.0.0 - 2025-06-16

### Added

- `DiscreteDynamicalSystem` class for simulating and analyzing discrete nonlinear dynamical systems:
  - Trajectory computation.
  - Chaotic indicators.
  - Fixed points, periodic orbits, and manifolds.
  - Statistical analysis of ensemble of trajetories.
  - Escape basin quantification.
- Initial release of the package
- First version of documentation
- Basic tests

- `BasinMetrics` class to compute basin metris such as basin entropy and boundary dimension.

- `TimeSeriesMetrics` class to compute metrics related to time series analysis.

- `PlotStyler` utility class to globally configure and apply consistent styling for Matplotlib plots.

<!-- Dummy heading to avoid ending on a transition -->

##
