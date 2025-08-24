# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [v1.3.1] - 2025-08-24

### Modified

- Removed `cache=True` from the low level methods that was leading to cache compilation errors.

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
  - The smallest aligment index (SALI) and linear dependence index (LDI) for chaos detection.

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
