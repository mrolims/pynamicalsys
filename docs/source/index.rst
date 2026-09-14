pynamicalsys: A Python toolkit for dynamical systems
====================================================

.. only:: development_docs

   .. warning::

      **Development documentation**

      This documentation describes the development version of **pynamicalsys**
      and may include changes that have not yet been released. For the latest
      released version, see the
      `stable documentation <https://pynamicalsys.readthedocs.io/en/stable/>`_.

**pynamicalsys** is a Python library for simulating and analyzing nonlinear
dynamical systems, designed for students, researchers, educators, and
enthusiasts exploring chaos and dynamical behavior. It brings numerical
simulation and analysis together so you can connect mathematical descriptions
with the behavior they produce: follow a trajectory, change a parameter, and
investigate how regular motion gives way to chaos.

Whether you are learning the subject, preparing a classroom demonstration, or
investigating a research problem, you can start with built-in models and use
the same tools with your own equations. You can also analyze time-series data
you already have, using recurrence statistics and other measures to explore
its structure.

Where to start
--------------

New to the package? Follow the :doc:`installation` guide, then try the
:doc:`quickstart`. For a more detailed introduction, choose the tutorial that
matches your system:

- **Discrete maps:** systems that advance one iteration at a time, such as the
  logistic or Hénon map. Start with the :doc:`discrete-system tutorial <dds_tutorial>`.
- **Continuous systems:** systems described by differential equations, such as
  the Lorenz system or a driven oscillator. Start with the
  :doc:`continuous-system tutorial <cds_tutorial>`.
- **Hamiltonian systems:** dynamics described by positions, momenta, and a
  Hamiltonian, with symplectic integration methods. Start with the
  :doc:`Hamiltonian-system tutorial <hs_tutorial>`.
- **Existing data:** use :doc:`TimeSeriesMetrics <api/time_series_metrics>` for
  recurrence analysis and Hurst exponents, or
  :doc:`BasinMetrics <api/basin_metrics>` to quantify basin structure.

A first simulation
------------------

The logistic map evolves a single variable according to
:math:`x_{n+1} = r x_n(1-x_n)`. This example starts at :math:`x_0 = 0.2` and
plots 100 iterations with :math:`r = 3.8`:

.. code-block:: python

   import matplotlib.pyplot as plt
   from pynamicalsys import DiscreteDynamicalSystem

   system = DiscreteDynamicalSystem(model="logistic map")
   trajectory = system.trajectory(0.2, 100, parameters=[3.8])

   plt.plot(range(1, len(trajectory) + 1), trajectory, ".-", markersize=3)
   plt.xlabel("Iteration n")
   plt.ylabel("x")
   plt.show()

For this one-dimensional map, ``trajectory`` is a one-dimensional array. Its
first entry is :math:`x_1`, the state after one iteration; the initial condition
:math:`x_0` is not included. The :doc:`quickstart` also introduces continuous
and Hamiltonian systems.

What you can explore
--------------------

**Trajectories and phase-space structure**
   Generate trajectories from one or many initial conditions, construct
   bifurcation diagrams, and sample continuous dynamics with Poincaré sections
   and stroboscopic maps.

**Regular and chaotic motion**
   Compute Lyapunov exponents, covariant Lyapunov vectors, and alignment and
   linear-dependence indicators. For discrete maps, weighted Birkhoff averages
   provide another way to investigate the dynamics.

**Recurrence, transport, and escape**
   Study recurrence plots and recurrence-time statistics, estimate Hurst
   exponents, analyze diffusion and transport in maps, and measure escape
   times and survival probabilities.

**Periodic orbits and basins**
   Locate periodic orbits of maps, examine their stability and manifolds, and
   quantify basin structure with entropy and uncertainty measures.

Available methods depend on the system class. The API references below list
supported inputs, options, and outputs. Plotting examples use Matplotlib;
:doc:`PlotStyler <api/plot_styler>` provides optional, consistent plot styling.

Numerical routines use NumPy and Numba. The first call to a calculation may
include compilation time; later calls with the same input types can reuse the
compiled code. See the benchmark notebook below for performance comparisons.

Publication and reproducibility
-------------------------------

The package and its applications are described in:

M. Rolim Sales et al., *pynamicalsys: A Python toolkit for the analysis of
dynamical systems*,
`Chaos, Solitons and Fractals 201, 117269 (2025) <https://doi.org/10.1016/j.chaos.2025.117269>`_.

The companion notebooks contain the paper's numerical experiments, figures,
and performance comparisons:

- `Reproduce the paper's results <https://github.com/mrolims/pynamicalsys/blob/main/paper/paper.ipynb>`_
- `Explore the benchmarks <https://github.com/mrolims/pynamicalsys/blob/main/paper/benchmarks.ipynb>`_

See :doc:`citation` for citation formats.

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
   hs_tutorial

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api/dds
   api/cds
   api/hs
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
