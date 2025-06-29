pynamicalsys: A Python toolkit for the analysis of dynamical systems
====================================================================

   *A fast, flexible, and user-friendly toolkit for analyzing chaotic systems and dynamical behaviors in Python.*

Welcome to **pynamicalsys**'s documentation! This is the official documentation for **pynamicalsys**, a Python toolkit for the analysis of dynamical systems. Here, you will find everything you need to get started, from installation instructions to detailed API references.

Overview
--------

**pynamicalsys** is designed to provide a fast, flexible, and user-friendly environment for analyzing **nonlinear dynamical systems**. It is intended for students, researchers, educators, and enthusiasts who want to explore the world of chaos and dynamical systems. Beyond standard tools like trajectory generation and Lyapunov exponents calculation, **pynamicalsys** includes advanced features such as

- **Linear dependence index** for chaos detection.
- **Recurrence plots** and recurrence time statistics.
- Chaos indicators based on **weighted Birkhoff averages**.
- Statistical measures of **diffusion and transport** in dynamical systems.
- Computation of **periodic orbits**, their **stability** and their **manifolds**.
- Basin metric for **quantifying** the structure of **basins of attraction**.
- **Plot styling** for consistent and customizable visualizations.

**pynamicalsys** is built on top of NumPy and Numba, ensuring high performance and efficiency. Thanks to Numba accelerated computation, **pynamicalsys** offers speedups up to **130x** compared to the original Python implementation of the algorithms. This makes it suitable for large-scale simulations and analyses.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: Tutorial

   dds_tutorial
   cds_tutorial

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api/dds
   api/cds
   api/basin_metrics
   api/time_series_metrics
   api/plot_styler

.. toctree::
   :maxdepth: 1
   :caption: Community

   citation
   contact
   contributing
   code_of_conduct
   contributors
   changelog
   acknowledgments
   disclaimer
