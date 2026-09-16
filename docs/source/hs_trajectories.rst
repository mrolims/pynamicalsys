Generating trajectories
-----------------------

The :py:meth:`trajectory <pynamicalsys.core.hamiltonian_systems.HamiltonianSystem.trajectory>` method of the :py:class:`HamiltonianSystem <pynamicalsys.core.hamiltonian_systems.HamiltonianSystem>` class integrates Hamilton's equations from an initial condition given by the coordinates :math:`\mathbf{q}` and the momenta :math:`\mathbf{p}`, which are passed as separate arguments. It returns a NumPy array of shape ``(N, d + 1)``, where ``d`` is the number of state variables (twice the number of degrees of freedom). The first column holds the time samples, and the remaining columns hold the state in the order :math:`(\mathbf{q}, \mathbf{p})`. For the two-degree-of-freedom Hénon-Heiles system these columns are :math:`x`, :math:`y`, :math:`p_x`, and :math:`p_y`.

The integrators are symplectic, so they do not conserve the energy exactly but keep its error bounded over long times, without the secular drift of a general-purpose method.

Choosing the integrator
~~~~~~~~~~~~~~~~~~~~~~~

The :py:meth:`integrator <pynamicalsys.core.hamiltonian_systems.HamiltonianSystem.integrator>` method selects one of three symplectic schemes. The second-order velocity Verlet (``vv2``) and the fourth-order composition of `Yoshida (1990) <https://doi.org/10.1016/0375-9601(90)90092-3>`_ (``svy4``) apply to separable Hamiltonians. The implicit midpoint method (``imp``) applies to general Hamiltonians. All three take a time step, which defaults to :math:`10^{-2}`:

.. code-block:: python

    from pynamicalsys import HamiltonianSystem

    system = HamiltonianSystem(model="henon heiles")
    system.integrator("svy4", time_step=0.01)

The implicit midpoint method solves a nonlinear equation at every step, so it also accepts a tolerance and a maximum number of iterations for the internal root finder:

.. code-block:: python

    system.integrator("imp", time_step=0.01, tol=1e-9, max_iter=1000)

A trajectory of the Hénon-Heiles system
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Hénon-Heiles system was introduced by `Hénon and Heiles (1964) <https://doi.org/10.1086/109234>`_ and has the Hamiltonian

.. math::

    H(x,y,p_x,p_y) = \frac{p_x^2+p_y^2}{2} + \frac{x^2+y^2}{2} + x^2y - \frac{y^3}{3}.

Fix the total energy at :math:`E=1/8` and place the initial condition on that energy surface. With :math:`x=0`, :math:`y=0.1`, and :math:`p_y=0`, the remaining momentum follows from :math:`E=\tfrac{1}{2}(p_x^2+p_y^2)+V(x,y)`:

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from pynamicalsys import HamiltonianSystem, PlotStyler

    system = HamiltonianSystem(model="henon heiles")
    system.integrator("svy4", time_step=0.01)

    energy = 1 / 8
    x, y, py = 0.0, 0.1, 0.0
    potential = (x**2 + y**2) / 2 + x**2 * y - y**3 / 3
    px = np.sqrt(2 * (energy - potential) - py**2)

    q = [x, y]
    p = [px, py]
    total_time = 1_000.0

    trajectory = system.trajectory(q, p, total_time)

The columns of ``trajectory`` are time, :math:`x`, :math:`y`, :math:`p_x`, and :math:`p_y`. Plotting the trajectory in the :math:`(x,y)` configuration plane shows the region it explores:

.. code-block:: python

    ps = PlotStyler()
    ps.apply_style()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(trajectory[:, 1], trajectory[:, 2], color="darkviolet", lw=0.3)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    fig.tight_layout()
    plt.show()

.. figure:: images/henon_heiles_trajectory.png
    :align: center
    :width: 80%

    A regular trajectory of the Hénon-Heiles system at energy :math:`E=1/8`, shown in the configuration plane.

The trajectory fills a bounded region with a smooth outer boundary and an empty center, the signature of quasiperiodic motion on an invariant torus. A chaotic trajectory at the same energy would instead spread across the accessible region.

Energy conservation
~~~~~~~~~~~~~~~~~~~

The quality of a symplectic integrator is seen in the energy error rather than in the trajectory alone. Define the relative energy error

.. math::

    E_r(t) = \frac{\left|E(t)-E_0\right|}{E_0},

and evaluate the Hénon-Heiles Hamiltonian along the trajectory produced by each integrator. Because Hénon-Heiles is separable, all three integrators apply:

.. code-block:: python

    def henon_heiles_energy(trajectory):
        x, y, px, py = (trajectory[:, i] for i in range(1, 5))
        return (px**2 + py**2) / 2 + (x**2 + y**2) / 2 + x**2 * y - y**3 / 3

    integrators = {"vv2": "VV2", "imp": "IMP", "svy4": "SVY4"}
    histories = {}
    for name in integrators:
        if name == "imp":
            system.integrator(name, time_step=0.01, tol=1e-12, max_iter=100)
        else:
            system.integrator(name, time_step=0.01)
        history = system.trajectory(q, p, total_time)
        error = np.abs(henon_heiles_energy(history) - energy) / energy
        histories[name] = (history[:, 0], error)

.. code-block:: python

    ps = PlotStyler()
    ps.apply_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, label in integrators.items():
        time, error = histories[name]
        ax.semilogy(time, error, lw=0.8, label=label)
    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$E_r = |E(t) - E_0| / E_0$")
    ax.legend()
    fig.tight_layout()
    plt.show()

.. figure:: images/henon_heiles_energy_check.png
    :align: center
    :width: 100%

    Relative energy error of the three integrators along the Hénon-Heiles trajectory. The error stays bounded for all three, with no secular growth.

The error remains bounded for every integrator, which is the defining property of symplectic integration. The fourth-order method ``svy4`` is several orders of magnitude more accurate than the second-order ``vv2`` and ``imp`` methods at the same time step. The explicit ``vv2`` and ``svy4`` methods are available here only because the Hénon-Heiles Hamiltonian is separable; a general Hamiltonian, such as the Walker-Ford system in the :doc:`creation guide <hs_creating_hs>`, must use ``imp``.

References
~~~~~~~~~~

.. container:: references-list

    - M\. Hénon and C\. Heiles, `The applicability of the third integral of motion: Some numerical experiments <https://doi.org/10.1086/109234>`_, The Astronomical Journal 69, 73 (1964).
    - H\. Yoshida, `Construction of higher order symplectic integrators <https://doi.org/10.1016/0375-9601(90)90092-3>`_, Physics Letters A 150, 262-268 (1990).
