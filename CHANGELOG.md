# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
