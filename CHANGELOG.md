# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Discrete-time systems can be given a **frequency map**, the advance of an angle variable in one iteration, in turns, with the new `frequency_map` argument of `DiscreteDynamicalSystem` or the new `set_frequency_map` method. The `unbounded standard map`, `standard nontwist map`, `extended standard nontwist map`, and `leonel map` models include one.
- `rotation_number` computes the rotation number from the frequency map with the **weighted Birkhoff average** (Sander and Meiss, 2020), which converges far faster on regular orbits and returns NaN for chaotic orbits. `threshold` sets the number of digits in which the averages over two consecutive windows of `window_size` iterations must agree (9 by default), `method="average"` gives the plain time average, and `transient_time` discards iterations first. The rotation number is no longer reduced modulo one.
- New method `classify_orbit` for discrete-time systems, which classifies an orbit as chaotic, in an island chain, or on a rotational invariant circle from weighted Birkhoff averages (Sander and Meiss, 2020), and returns the rotation number and the number of converged digits. An orbit whose rotation number has not converged to `omega_threshold` digits is reported as undetermined; a longer `window_size` usually classifies it. It takes an observable and `transient_time` as `dig` does.
- `sample_times` stores results only at the requested times in `trajectory` of all three classes, and, with `return_history=True`, in `lyapunov`, `SALI`, `LDI`, and `GALI` of `ContinuousDynamicalSystem` and `HamiltonianSystem`. Discrete times are iterations counted from the initial condition. With an adaptive integrator, the integration lands exactly on each time; with a fixed step, and in `HamiltonianSystem`, each time is rounded to the nearest step. Every trajectory of an ensemble is stored at the same times.
- `lyapunov`, `SALI`, `LDI`, and `GALI` of `ContinuousDynamicalSystem` and `HamiltonianSystem` accept `return_last_state`, which also returns the state at the time the integration stopped (in `HamiltonianSystem`, as `(result, q_last, p_last)`), for parameter continuation without repeating the transient.
- `ContinuousDynamicalSystem.lyapunov` accepts `qr_interval`, the number of steps between reorthonormalizations (1 by default). Values greater than 1 require a fixed-step integrator.
- `DiscreteDynamicalSystem.trajectory` accepts `concatenate`. With `concatenate=False`, an ensemble is returned with shape `(num_initial_conditions, sample_size, system_dimension)`.
- `DiscreteDynamicalSystem.recurrence_times` accepts `periods`, the wrapping period of each coordinate (`np.inf` for coordinates that do not wrap), so returns across the boundary of a periodic coordinate are detected.
- `DiscreteDynamicalSystem.lyapunov` and `finite_time_lyapunov` accept `seed`, as the other methods that draw random numbers do.
- `BasinMetrics.basin_entropy` uses boxes of `nx` by `ny` grid points when they are given, along the first and second axes of the basin. Both were accepted before but ignored.
- `PlotStyler` accepts Matplotlib's short names `lw`, `ms`, `mew`, and `mec`, and `set_tick_padding` accepts an array or list of Axes, such as the one returned by `plt.subplots`.

### Changed

- The observable of `dig` is a function of a **single state** that returns a number, for example `lambda u: np.cos(2 * np.pi * u[0])`. The default observable, `cos(2 pi u[0] / period)`, takes the period of `u[0]` from the new `period` argument (1 by default).
- `dig` takes `window_size`, the length of each of the two windows it compares, instead of `total_time`, so the same call iterates twice as long as before and usually returns a higher value. Pass half the former value to reproduce previous results.
- In `DiscreteDynamicalSystem.recurrence_times`, `total_time` counts iterations from the initial condition and includes `transient_time`; add `transient_time` to `total_time` to reproduce previous results. The recurrence times are returned as integers.
- In `lyapunov`, `SALI`, `LDI`, and `GALI` of `DiscreteDynamicalSystem`, `sample_times` are iterations counted from the initial condition and must lie in `(transient_time, total_time]`; add `transient_time` to them to reproduce previous results. When the indicator falls below `threshold`, the history now ends at the stop instead of being padded with zeros.
- `SALI`, `LDI`, and `GALI` of `ContinuousDynamicalSystem` and `HamiltonianSystem` stop when the indicator falls strictly below `threshold`, as documented and as in `DiscreteDynamicalSystem`, so `threshold=0` never stops early.
- Every method that draws random numbers, including `BasinMetrics.uncertainty_fraction_mapping`, uses the default seed 13121989 (it was 1312 or 13 depending on the class and method, and a fixed 1312 in the discrete `lyapunov` and `finite_time_lyapunov`), so results computed with the default seed change. Pass `seed` to reproduce earlier results.
- `DiscreteDynamicalSystem.finite_time_lyapunov` carries the deviation vectors from one window to the next instead of restarting them, so the mean of the windows equals the exponent from `lyapunov` over the same iterations and the methods agree after the first window. Results change, most for short windows, where each window used to include the alignment of new vectors; over short windows, the exponents of a window need not be in decreasing order.
- Poincaré sections interpolate the crossing with a cubic Hermite polynomial, which uses the vector field at both ends of the step, instead of linearly: `poincare_section` of `ContinuousDynamicalSystem` and `HamiltonianSystem`, the sections of `HamiltonianSystem.CLV` and `CLV_angles`, and `recurrence_time_entropy`, `hurst_exponent`, and `basin_of_attraction` with `map_type="PS"`. Section points are far more accurate, for example 1e-4 instead of 5e-2 on the Lorenz system with `rk4` and `time_step=0.01`.
- `DiscreteDynamicalSystem.manifold` spaces the seed points geometrically over one fundamental domain, so the output no longer repeats the periodic orbit and its points no longer cluster. The manifolds have the same shape but differ point by point.
- The step-size control of `rk45` uses the exponent of the Dormand-Prince 5(4) pair, so `rk45` results may differ slightly.
- Many methods are faster, notably `rk45`, `lyapunov`, `SALI`, and `GALI` in all three classes, the ensemble averages of `DiscreteDynamicalSystem`, and ensembles in `ContinuousDynamicalSystem.trajectory` and `HamiltonianSystem.poincare_section`, which now run in threads; `max_workers` of `poincare_section` now sets the number of threads. The Lyapunov exponents and alignment indices, and the CLVs of continuous and Hamiltonian systems, may differ at the level of round-off (in `DiscreteDynamicalSystem`, only on maps of more than two dimensions), and, with `rk45`, by more on chaotic orbits.
- `ContinuousDynamicalSystem` emits a `RuntimeWarning` when the result of `evolve_system`, `trajectory`, a reduced map, `lyapunov`, `SALI`, `LDI`, or `GALI` contains infinite or NaN values, which usually means the solution diverged.
- `lyapunov`, `SALI`, `LDI`, and `GALI` emit a `UserWarning` when `sample_times` is passed without `return_history=True`, where it has no effect.
- `PlotStyler.apply_style` no longer clears the current figure, which also removes the warnings it emitted after figures with shared logarithmic axes. Call `plt.clf()` yourself if you relied on it.
- `DiscreteDynamicalSystem.manifold` raises a `ValueError` for a stable manifold without a backward map. The error is also a `RuntimeError`, as raised before.
- Python 3.14 and 3.15 are supported. On Python 3.15, pip installs Numba 0.68 or newer.

### Deprecated

- Calling `rotation_number` on a system without a frequency map, and passing `mod` when a frequency map is set. Set a frequency map instead.
- Passing to `dig` an observable that acts on an array of states. Pass a function of a single state instead.
- The `total_time` argument of `dig` and `rotation_number`. Use `window_size` instead.
- The `tol` argument of `SALI`, `LDI`, and `GALI` of `DiscreteDynamicalSystem`. Use `threshold` instead.
- The `endpoint` argument of `lyapunov`, `SALI`, `LDI`, and `GALI` of `ContinuousDynamicalSystem`, which is ignored.
- Calling `DiscreteDynamicalSystem.trajectory` with an ensemble without `concatenate`. In a future version the default will be `concatenate=False`; pass `concatenate=True` to keep the concatenated array.
- Whole-number floats in the `n_points` and `iter_time` of `DiscreteDynamicalSystem.manifold`. Pass integers instead.

### Fixed

- `ContinuousDynamicalSystem.poincare_section` with `rk45` gave wrong crossing times after a rejected step, and never counted a step that landed exactly on the section, hanging if no other crossing followed; `HamiltonianSystem.poincare_section` counted such a step twice.
- `HamiltonianSystem.CLV` and `CLV_angles` with `poincare_section=True` always used the upward crossings of `q[0] = 0`, ignoring `section_index`, `section_value`, and `crossing`, and gave crossing times that ignored `warmup_time` and, with a `qr_time_step` that is not a multiple of the time step, used the wrong interval between samples; the warmup and tail were then also shorter than requested.
- `ContinuousDynamicalSystem.CLV` and `CLV_angles` with `warmup_time` stored wrong, sometimes decreasing, times, and could practically never end with `rk4` and a `warmup_time` or `tail_time` that is not a multiple of the time step. Round-off could also drop a QR block. The samples now run from `transient_time + warmup_time` to `total_time`, and a `ValueError` is raised when there is no time left.
- `ContinuousDynamicalSystem.stroboscopic_map` interpolated the state between steps, with errors of order one at usual step sizes, and failed with a `ZeroDivisionError` for steps longer than `sampling_time` or for `sampling_time=0`. It now lands exactly on every sampling time and requires a positive `sampling_time`, also in `basin_of_attraction`, `recurrence_time_entropy`, and `hurst_exponent` with `map_type="SM"`.
- `DiscreteDynamicalSystem.finite_time_lyapunov` replaced `method="QR"` with `"ER"` on two-dimensional maps and rejected `"ER"` by name, limited `finite_time` to `total_time // 2` regardless of `transient_time`, and did not check `num_exponents`. `finite_time` may now be as long as `total_time - transient_time`.
- `DiscreteDynamicalSystem.lyapunov` and `finite_time_lyapunov` accepted `num_exponents=0` and returned empty arrays. It must now be a positive integer.
- `DiscreteDynamicalSystem.survival_probability` was always 0 at `max_time`. An escape time equal to `max_time`, as `escape_analysis` returns for trajectories that do not escape, now counts as not escaped.
- The `lyapunov` methods and `finite_time_lyapunov` accepted `log_base=0` and returned zeros, and `BasinMetrics.basin_entropy` accepted `log_base=1` and returned infinite or NaN values. Both now raise a `ValueError`.
- `ContinuousDynamicalSystem.lyapunov` with `return_history=True` and a `log_base` other than `e` also rescaled the time column.
- `ContinuousDynamicalSystem.evolve_system`, `lyapunov`, `SALI`, `LDI`, and `GALI` integrated one step past `total_time`, so results differ slightly.
- A `transient_time` shorter than the first integration step was overshot in `ContinuousDynamicalSystem`, which mainly affected non-autonomous systems.
- With a fixed step, round-off could leave the integration slightly short of `total_time`, and `trajectory` then returned the final state twice.
- With `rk45`, a diverging solution made the integration run forever; it now raises a `RuntimeError`. Ensembles with `rk45` also failed for vector fields written for a single state.
- `ContinuousDynamicalSystem.integrator` ignored names with uppercase letters, such as `"RK45"`, and accepted `time_step=0` and `atol=0`, which made the integration run forever. A name that is not a string now raises a `TypeError` instead of a `ValueError`.
- Modifying the dictionary returned by `integrator_info` of `ContinuousDynamicalSystem` or `HamiltonianSystem` changed it for every instance.
- `HamiltonianSystem.lyapunov` left out the last steps when `qr_interval` did not divide the number of steps, and ignored `qr_interval` for `num_exponents=1`.
- With the `imp` integrator, `lyapunov`, `SALI`, `LDI`, `GALI`, and `CLV` of `HamiltonianSystem` continued after a Newton iteration that did not converge; they now raise a `RuntimeError`.
- `HamiltonianSystem.poincare_section` with an ensemble failed in scripts without an `if __name__ == "__main__":` guard on macOS and Windows.
- The ensemble averages of `DiscreteDynamicalSystem` (`average_in_time`, `cumulative_average`, `root_mean_squared`, and `mean_squared_displacement`) changed at the level of round-off with the number of threads. `cumulative_average` and `root_mean_squared` also returned NaN for a sample time of 0, which now raises a `ValueError`.
- `DiscreteDynamicalSystem.bifurcation_diagram` failed for systems with more than one parameter when `parameters` was passed while parameters were stored, or omitted while none were.
- `DiscreteDynamicalSystem.find_periodic_orbit` added cache files in every Python session and, after many sessions, could fail with `ReferenceError: underlying object has vanished`.
- `DiscreteDynamicalSystem.manifold` truncated fractional `n_points` and `iter_time`, accepted booleans and strings, and refused a list for `n_points`.
- `DiscreteDynamicalSystem.recurrence_times` accepted `eps=0`, and `DiscreteDynamicalSystem` accepted `system_dimension=0`.
- Arguments that must be positive, such as `total_time` or `time_step`, accepted `True` as 1.
- The **recurrence time entropy** left out white vertical lines ending in the last column of the recurrence matrix, so results may differ from those of v1.4.6 to v1.7.1.
- `TimeSeriesMetrics.recurrence_time_entropy` and `recurrence_matrix` with `compute_white_vert_distr=True` started the white-line distribution at `lmin + 1` instead of `lmin`.
- The recurrence methods warned that `threshold_std` is deprecated even when it was not passed.
- `PlotStyler(tick_direction_in=False)` had no effect.

[Unreleased]: https://github.com/mrolims/pynamicalsys/compare/v1.7.0...HEAD

## [v1.7.0] - 2026-09-16

### Added

- `find_periodic_orbit` can now refine a single initial guess with **Newton's method**, selected by passing a 1D initial guess instead of a grid. Unlike the grid and symmetry-line searches, which are limited to two dimensions, it works in **any number of dimensions**, so one-dimensional maps such as the logistic map and four-dimensional ones such as the 4D symplectic map are now supported, and it is far more accurate at elliptic orbits, where the searches stall. Two options apply: `periods`, the wrapping period of each coordinate for maps on a torus, and `prime_period`, which rejects a solution whose period is a proper divisor of the one requested (off by default).
- All built-in dynamical system classes now include the governing equations in the `.info` property, which returns a dictionary describing the current model. The equation is available in LaTeX form via `.info["equation"]` and in plain-text Unicode form via `.info["equation_readable"]`, along with a short descriptive note.
- New optional install `pip install pynamicalsys[notebook]`, which adds IPython so that `.info["equation"]` renders as typeset mathematics in a notebook.

### Changed

- Choosing a recurrence threshold from a **fixed recurrence rate** now uses far less memory and is considerably faster, so long series that previously ran out of memory now work. The threshold returned is unchanged.
- IPython is no longer required, and any version now works. Without it, `.info["equation"]` returns the LaTeX source as a plain string instead of a rendered equation; install `pynamicalsys[notebook]` to restore rendered output.
- The minimum supported Python version is now correctly declared as 3.10. It was already required, but the package advertised 3.8, so installs on 3.8 or 3.9 appeared to succeed and then failed on import; `pip` now reports the incompatibility up front.

### Fixed

- Recurrence matrices built with `metric="euclidean"` used the **Manhattan** distance and `metric="manhattan"` used the **Euclidean** distance, wherever a recurrence matrix is built, which affects recurrence plots, recurrence time entropy and `TimeSeriesMetrics`. Results obtained with either of these two metrics will change; the default `"supremum"` metric was always correct and is unaffected.
- The Jacobian of the **Lozi map** had the wrong sign on its `dx'/dx` entry (the map is `x' = 1 - a|x| + y`, so the derivative is `-a sign(x)`). Any quantity computed from the linearised dynamics of this model was affected, including eigenvalues and eigenvectors of periodic orbits, stability classification, invariant manifolds, covariant Lyapunov vectors, and the finite-time SALI, LDI and GALI indicators. The averaged Lyapunov exponents were largely unaffected.
- The backward mapping of the **4D symplectic map** was not the inverse of the forward mapping, with an O(1) round-trip error. Computations that iterate the map backwards, such as the stable branches of invariant manifolds, gave incorrect results for this model.
- Modifying the dictionary returned by `.info` no longer alters the built-in model definition for the rest of the session.
- `HamiltonianSystem.info` now raises `ValueError` with an explanatory message for systems built from custom functions, as documented, instead of an unhelpful `KeyError`.

[v1.7.0]: https://github.com/mrolims/pynamicalsys/compare/v1.6.0...v1.7.0

## [v1.6.0] - 2026-07-01

### Added

- Support for **general (non-separable) Hamiltonian systems** `H(q, p)` via a new `"imp"` (implicit midpoint) integrator. Specify the system via `eom` and `hess_H` instead of `grad_T`/`grad_V`. All analysis methods (`lyapunov`, `CLV`, `CLV_angles`, `SALI`, `LDI`, `GALI`, `poincare_section`, `recurrence_time_entropy`, `hurst_exponent`) dispatch to the implicit midpoint backend automatically when `"imp"` is selected.
- Support for **periodic/angular section coordinates** in Poincaré sections via `periodic_section_coordinate` and `period` parameters. Useful for action-angle systems where the section coordinate grows unbounded rather than oscillating. Available on `poincare_section`, `recurrence_time_entropy`, and `hurst_exponent`.
- `poincare_section` now accepts a `max_workers` parameter controlling the number of parallel workers used when computing an ensemble of initial conditions (default: all available CPUs).
- The implicit midpoint integrator now raises `RuntimeError` on non-convergence instead of silently returning corrupted results.
- `BasinMetrics`: added `uncertainty_fraction_mapping` for Monte Carlo estimation of the uncertainty fraction and uncertainty exponent of escape basins, with parallel evaluation and reproducible random sampling via a user-provided seed.

[v1.6.0]: https://github.com/mrolims/pynamicalsys/compare/v1.5.4...v1.6.0

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
