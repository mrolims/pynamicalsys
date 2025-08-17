# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `DiscreteDynamicalSystem` class:
  - step method that returns the next step of the system.

### Modified

- `DiscreteDynamicalSystem` class:
    - Optimized methods to check sampling points without repeatedly searching through sample_times, improving performance for large sampling lists.

- `ContinuousDynamicalSystem` class:
    - Centralized the integration step logic previously duplicated across functions like trajectory, lyapunov_exponents, etc., into a new step() function.

[Unreleased]: https://github.com/mrolims/pynamicalsys/compare/v1.2.2...HEAD

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
