Alignment indices
-----------------

The Smaller Alignment Index (SALI), Generalized Alignment Index (GALI), and Linear Dependence Index (LDI) follow normalized deviation vectors as they evolve under the tangent dynamics. They provide closely related ways to distinguish regular and chaotic motion. SALI was introduced by `Skokos (2001) <https://doi.org/10.1088/0305-4470/34/47/309>`_, GALI by `Skokos, Bountis, and Antonopoulos (2007) <https://doi.org/10.1016/j.physd.2007.04.004>`_, and LDI by `Antonopoulos and Bountis (2006) <https://arxiv.org/abs/0711.0360>`_.

Consider the four-dimensional Rössler system

.. math::

    \begin{aligned}
        \dot{x} &= -y-z,\\
        \dot{y} &= x+ay+w,\\
        \dot{z} &= b+xz,\\
        \dot{w} &= -cz+dw.
    \end{aligned}

The parameters :math:`a=0.25`, :math:`b=3`, :math:`c=0.5`, and :math:`d=0.05` produce a hyperchaotic trajectory. Use a fixed-step integrator so that all three indicators are sampled at the same time interval:

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from pynamicalsys import ContinuousDynamicalSystem as cds, PlotStyler

    system = cds(model="4d rossler system")
    system.set_parameters([0.25, 3.0, 0.5, 0.05])
    system.integrator("rk4", time_step=0.01)

    initial_state = [-20.0, 0.0, 0.0, 15.0]
    transient_time = 1_000.0
    total_time = 2_000.0
    threshold = 1e-16
    seed = 1312

Smaller Alignment Index
~~~~~~~~~~~~~~~~~~~~~~~

Let :math:`\hat{\mathbf{v}}_1(t)` and :math:`\hat{\mathbf{v}}_2(t)` be two normalized deviation vectors. SALI is defined as

.. math::

    \mathrm{SALI}(t)=\min\left\{\left\|\hat{\mathbf{v}}_1(t)+\hat{\mathbf{v}}_2(t)\right\|,\left\|\hat{\mathbf{v}}_1(t)-\hat{\mathbf{v}}_2(t)\right\|\right\}.

The two norms detect alignment in the same or opposite directions. Along a chaotic trajectory, both vectors approach the most unstable tangent direction. For Lyapunov exponents :math:`\lambda_1\geq\lambda_2\geq\cdots`, the asymptotic decay is

.. math::

    \mathrm{SALI}(t)\propto\exp\left[-(\lambda_1-\lambda_2)t\right].

For a typical chaotic autonomous flow with one positive exponent, :math:`\lambda_2=0` is the neutral exponent along the flow direction. In a hyperchaotic flow, :math:`\lambda_2` can be positive and must be retained in the decay rate. For regular trajectories, SALI generally remains bounded away from zero when the invariant object has at least two independent tangent directions.

Compute the history with :py:meth:`SALI <pynamicalsys.core.continuous_dynamical_systems.ContinuousDynamicalSystem.SALI>`:

.. code-block:: python

    sali_history = system.SALI(
        initial_state,
        total_time,
        transient_time=transient_time,
        return_history=True,
        threshold=threshold,
        seed=seed,
    )

With ``return_history=True``, the first column contains time and the second contains SALI. The calculation stops when the indicator falls below ``threshold``.

.. code-block:: python

    ps = PlotStyler()
    ps.apply_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(
        sali_history[:, 0],
        np.maximum(sali_history[:, 1], threshold),
        color="darkviolet",
    )
    ax.set_xlabel("Time $t$")
    ax.set_ylabel(r"$\mathrm{SALI}$")
    fig.tight_layout()
    plt.show()

.. figure:: images/continuous_rossler4d_sali.png
    :align: center
    :width: 80%

    Evolution of SALI for the hyperchaotic four-dimensional Rössler system.

Generalized Alignment Index
~~~~~~~~~~~~~~~~~~~~~~~~~~~

GALI extends the alignment idea to :math:`k` normalized deviation vectors. If :math:`\mathbf{V}(t)\in\mathbb{R}^{d\times k}` contains these vectors as columns, the index is the volume of the :math:`k`-dimensional parallelepiped that they span:

.. math::

    \mathrm{GALI}_k(t)=\left\|\hat{\mathbf{v}}_1(t)\wedge\hat{\mathbf{v}}_2(t)\wedge\cdots\wedge\hat{\mathbf{v}}_k(t)\right\|.

For a chaotic trajectory, the decay is governed by the first :math:`k` Lyapunov exponents:

.. math::

    \mathrm{GALI}_k(t)\propto\exp\left\{-\left[(k-1)\lambda_1-\sum_{i=2}^{k}\lambda_i\right]t\right\}.

For regular trajectories, whether :math:`\mathrm{GALI}_k` remains nonzero or decays algebraically depends on :math:`k` and the dimension of the invariant torus. This behavior and the decay laws were established by `Skokos, Bountis, and Antonopoulos (2007) <https://doi.org/10.1016/j.physd.2007.04.004>`_.

The :py:meth:`GALI <pynamicalsys.core.continuous_dynamical_systems.ContinuousDynamicalSystem.GALI>` method provides three ways to compute the spanned volume:

``method="DET"``
    Forms the Gram matrix :math:`\mathbf{G}=\mathbf{V}^T\mathbf{V}` and computes :math:`\mathrm{GALI}_k=\sqrt{\det(\mathbf{G})}`. This is the most direct expression, but it becomes sensitive to rounding when the vectors are nearly aligned.

``method="QR"``
    Uses the project's modified Gram-Schmidt decomposition :math:`\mathbf{V}=\mathbf{Q}\mathbf{R}` and computes :math:`\mathrm{GALI}_k=\prod_i|R_{ii}|`. This is the default method.

``method="QR_HH"``
    Uses a Householder QR decomposition and the same product :math:`\prod_i|R_{ii}|`. It is generally the most numerically stable option, with a higher computational cost.

Compare the three methods for :math:`k=2` using the same initial deviation vectors:

.. code-block:: python

    gali_methods = ["QR", "QR_HH", "DET"]
    gali_method_histories = []
    for method in gali_methods:
        gali_method_histories.append(
            system.GALI(
                initial_state,
                total_time,
                k=2,
                transient_time=transient_time,
                return_history=True,
                method=method,
                threshold=threshold,
                seed=seed,
            )
        )

    colors = ["darkviolet", "darkgreen", "darkorange"]

    ps = PlotStyler()
    ps.apply_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    for history, method, color in zip(gali_method_histories, gali_methods, colors):
        ax.semilogy(
            history[:, 0],
            np.maximum(history[:, 1], threshold),
            color=color,
            label=method,
        )
    ax.set_xlabel("Time $t$")
    ax.set_ylabel(r"$\mathrm{GALI}_2$")
    ax.legend(frameon=False)
    fig.tight_layout()
    plt.show()

.. figure:: images/continuous_rossler4d_gali_methods.png
    :align: center
    :width: 80%

    Comparison of the modified Gram-Schmidt QR, Householder QR, and Gram determinant formulations of :math:`\mathrm{GALI}_2`.

The determinant formulation commonly loses resolution first because :math:`\det(\mathbf{V}^T\mathbf{V})=\mathrm{GALI}_2^2`. When :math:`\mathrm{GALI}_2` is approximately :math:`10^{-8}`, the determinant is already approximately :math:`10^{-16}` and rounding errors can dominate. The QR formulations operate directly on the volume through the diagonal entries of :math:`\mathbf{R}`.

GALI for different values of k
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compute :math:`\mathrm{GALI}_2`, :math:`\mathrm{GALI}_3`, and :math:`\mathrm{GALI}_4` for the same trajectory:

.. code-block:: python

    k_values = np.array([2, 3, 4])
    gali_histories = []
    for k in k_values:
        gali_histories.append(
            system.GALI(
                initial_state,
                total_time,
                k=k,
                transient_time=transient_time,
                return_history=True,
                method="QR_HH",
                threshold=threshold,
                seed=seed,
            )
        )

    colors = ["darkviolet", "darkgreen", "darkorange"]

    ps = PlotStyler()
    ps.apply_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    for history, k, color in zip(gali_histories, k_values, colors):
        ax.semilogy(
            history[:, 0],
            np.maximum(history[:, 1], threshold),
            color=color,
            label=rf"$\mathrm{{GALI}}_{k}$",
        )
    ax.set_xlabel("Time $t$")
    ax.set_ylabel(r"$\mathrm{GALI}_k$")
    ax.legend(frameon=False)
    fig.tight_layout()
    plt.show()

.. figure:: images/continuous_rossler4d_gali.png
    :align: center
    :width: 80%

    Evolution of :math:`\mathrm{GALI}_2`, :math:`\mathrm{GALI}_3`, and :math:`\mathrm{GALI}_4` for the hyperchaotic four-dimensional Rössler system.

Linear Dependence Index
~~~~~~~~~~~~~~~~~~~~~~~

LDI computes the same volume from the singular values of the normalized deviation matrix. If

.. math::

    \mathbf{V}(t)=\mathbf{U}(t)\mathbf{\Sigma}(t)\mathbf{W}^T(t),

then the definition introduced by `Antonopoulos and Bountis (2006) <https://arxiv.org/abs/0711.0360>`_ is

.. math::

    \mathrm{LDI}_k(t)=\prod_{i=1}^{k}\sigma_i(t).

Compute :math:`\mathrm{LDI}_2`, :math:`\mathrm{LDI}_3`, and :math:`\mathrm{LDI}_4` with :py:meth:`LDI <pynamicalsys.core.continuous_dynamical_systems.ContinuousDynamicalSystem.LDI>`:

.. code-block:: python

    ldi_histories = []
    for k in k_values:
        ldi_histories.append(
            system.LDI(
                initial_state,
                total_time,
                k=k,
                transient_time=transient_time,
                return_history=True,
                threshold=threshold,
                seed=seed,
            )
        )

    ps = PlotStyler()
    ps.apply_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    for history, k, color in zip(ldi_histories, k_values, colors):
        ax.semilogy(
            history[:, 0],
            np.maximum(history[:, 1], threshold),
            color=color,
            label=rf"$\mathrm{{LDI}}_{k}$",
        )
    ax.set_xlabel("Time $t$")
    ax.set_ylabel(r"$\mathrm{LDI}_k$")
    ax.legend(frameon=False)
    fig.tight_layout()
    plt.show()

.. figure:: images/continuous_rossler4d_ldi.png
    :align: center
    :width: 80%

    Evolution of :math:`\mathrm{LDI}_2`, :math:`\mathrm{LDI}_3`, and :math:`\mathrm{LDI}_4` for the hyperchaotic four-dimensional Rössler system.

Correspondence between GALI and LDI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GALI and LDI measure the same volume. Since

.. math::

    \mathbf{V}^T\mathbf{V}=\mathbf{W}\mathbf{\Sigma}^T\mathbf{\Sigma}\mathbf{W}^T,

the determinant of the Gram matrix is :math:`\prod_{i=1}^{k}\sigma_i^2`. Therefore

.. math::

    \mathrm{GALI}_k(t)=\sqrt{\det\left(\mathbf{V}^T(t)\mathbf{V}(t)\right)}=\prod_{i=1}^{k}\sigma_i(t)=\mathrm{LDI}_k(t).

This exact identity and the common decay rate for discrete and continuous chaotic systems are derived by `Sales, Leonel, and Antonopoulos (2026) <https://doi.org/10.1016/j.chaos.2026.117884>`_. The indicators are mathematically identical when they use the same normalized deviation vectors, while their numerical implementations differ. GALI uses a determinant or QR factorization and LDI uses a singular value decomposition.

Benchmarking the formulations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compare SALI, the three formulations of :math:`\mathrm{GALI}_2`, and :math:`\mathrm{LDI}_2` along a regular periodic Duffing trajectory over the same fixed integration time. Setting ``threshold=0.0`` disables early stopping so every formulation performs the complete workload. A short preliminary call excludes just-in-time compilation from the measurements. Absolute times depend on the processor and software environment, so the benchmark should be run on the machine where the methods will be used:

.. code-block:: python

    from time import perf_counter

    benchmark_system = cds(model="duffing")
    benchmark_system.set_parameters([0.2, 1.0, 1.0, 3.0, 1.1])
    benchmark_system.integrator("rk4", time_step=0.01)

    benchmark_state = [-2.0, 0.0]
    benchmark_transient_time = 50.0
    benchmark_time = 100.0
    benchmark_repeats = 20
    benchmark_labels = ["SALI"] + [f"GALI {method.replace('_', '-')}" for method in gali_methods] + ["LDI SVD"]
    benchmark_times = np.empty((len(benchmark_labels), benchmark_repeats))

    benchmark_system.SALI(benchmark_state, 0.01, threshold=0.0, seed=seed)
    for method in gali_methods:
        benchmark_system.GALI(benchmark_state, 0.01, k=2, method=method, threshold=0.0, seed=seed)
    benchmark_system.LDI(benchmark_state, 0.01, k=2, threshold=0.0, seed=seed)

    for repeat in range(benchmark_repeats):
        start = perf_counter()
        benchmark_system.SALI(
            benchmark_state,
            benchmark_time,
            transient_time=benchmark_transient_time,
            threshold=0.0,
            seed=seed,
        )
        benchmark_times[0, repeat] = perf_counter() - start

        for index, method in enumerate(gali_methods):
            start = perf_counter()
            benchmark_system.GALI(
                benchmark_state,
                benchmark_time,
                k=2,
                transient_time=benchmark_transient_time,
                method=method,
                threshold=0.0,
                seed=seed,
            )
            benchmark_times[index + 1, repeat] = perf_counter() - start

        start = perf_counter()
        benchmark_system.LDI(
            benchmark_state,
            benchmark_time,
            k=2,
            transient_time=benchmark_transient_time,
            threshold=0.0,
            seed=seed,
        )
        benchmark_times[-1, repeat] = perf_counter() - start

    mean_times = np.mean(benchmark_times, axis=1)
    std_times = np.std(benchmark_times, axis=1, ddof=1)

    ps = PlotStyler()
    ps.apply_style()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(
        benchmark_labels,
        mean_times,
        yerr=std_times,
        color="darkviolet",
        edgecolor="black",
        linewidth=1.0,
        capsize=4,
    )
    ax.set_ylabel("Wall time (s)")
    ax.tick_params(axis="x", labelrotation=15)
    fig.tight_layout()
    plt.show()

.. figure:: images/continuous_alignment_indices_benchmark.png
    :align: center
    :width: 100%

    Wall times for SALI, the three formulations of :math:`\mathrm{GALI}_2`, and SVD-based :math:`\mathrm{LDI}_2` along a regular periodic trajectory. Bars show the mean of 20 fixed-workload runs and error bars show one standard deviation.

All three public methods return a two-element array containing the final time and indicator value by default. With ``return_history=True``, they return a two-dimensional array whose columns contain time and the indicator history. The ``parameters``, ``transient_time``, ``seed``, ``threshold``, and ``endpoint`` arguments follow the same pattern for SALI, GALI, and LDI.

References
~~~~~~~~~~

- C. Skokos, `Alignment indices: a new, simple method for determining the ordered or chaotic nature of orbits <https://doi.org/10.1088/0305-4470/34/47/309>`_, Journal of Physics A: Mathematical and General 34, 10029-10043 (2001).
- C. Skokos, T. C. Bountis, and C. Antonopoulos, `Geometrical properties of local dynamics in Hamiltonian systems: The Generalized Alignment Index (GALI) method <https://doi.org/10.1016/j.physd.2007.04.004>`_, Physica D 231, 30-54 (2007).
- C. Antonopoulos and T. Bountis, `Detecting order and chaos by the Linear Dependence Index (LDI) method <https://arxiv.org/abs/0711.0360>`_, ROMAI Journal 2, 1-13 (2006).
- M. R. Sales, E. D. Leonel, and C. G. Antonopoulos, `On the behavior of Linear Dependence, Smaller, and Generalized Alignment Indices in discrete and continuous chaotic systems <https://doi.org/10.1016/j.chaos.2026.117884>`_, Chaos, Solitons and Fractals 205, 117884 (2026).
