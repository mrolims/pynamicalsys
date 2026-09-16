Creating a Hamiltonian system
-----------------------------

The :py:class:`HamiltonianSystem <pynamicalsys.core.hamiltonian_systems.HamiltonianSystem>` class represents systems written in generalized coordinates :math:`\mathbf{q}` and momenta :math:`\mathbf{p}`. You can select a built-in model or provide functions for a custom separable or general Hamiltonian.

Using a built-in model
~~~~~~~~~~~~~~~~~~~~~~

Import the class as ``hs`` and call :py:meth:`available_models <pynamicalsys.core.hamiltonian_systems.HamiltonianSystem.available_models>` to see the built-in models:

.. code-block:: python

    from pynamicalsys import HamiltonianSystem as hs

    for model in hs.available_models():
        print(model)

.. code-block:: text

    henon heiles

The built-in Hénon-Heiles model has two degrees of freedom and Hamiltonian

.. math::

    H(x,y,p_x,p_y) = \frac{p_x^2+p_y^2}{2} + \frac{x^2+y^2}{2} + x^2y - \frac{y^3}{3}.

Create the system and inspect its metadata with the ``info`` property:

.. code-block:: python

    system = hs(model="henon heiles")
    print(system.info["degrees of freedom"])
    print(system.info["parameters"])

.. code-block:: text

    2
    []

The coordinate and momentum vectors are ordered as ``q = [x, y]`` and ``p = [px, py]``. This model has no adjustable parameters.

Choosing a custom representation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A separable Hamiltonian has the form :math:`H(\mathbf{q},\mathbf{p}) = T(\mathbf{p}) + V(\mathbf{q})`. Define ``grad_T`` and ``grad_V`` to use the explicit ``svy4`` or ``vv2`` integrator. Add ``hess_T`` and ``hess_V`` when tangent-space calculations are needed.

A general Hamiltonian can depend on coordinates and momenta in a way that cannot be separated. Define ``eom`` and ``hess_H`` for this representation. The class then uses the implicit midpoint integrator, ``imp``.

The available integrators can be listed directly:

.. code-block:: python

    for integrator in hs.available_integrators():
        print(integrator)

.. code-block:: text

    svy4
    vv2
    imp

Creating Hénon-Heiles as a separable system
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Hénon-Heiles Hamiltonian separates into kinetic and potential energies:

.. math::

    H(\mathbf{q},\mathbf{p}) = T(\mathbf{p}) + V(\mathbf{q}) = \frac{p_x^2+p_y^2}{2} + \frac{x^2+y^2}{2} + x^2y - \frac{y^3}{3}.

The gradients are

.. math::

    \nabla_{\mathbf{q}}V = \begin{pmatrix}x+2xy \\ y+x^2-y^2\end{pmatrix}, \qquad \nabla_{\mathbf{p}}T = \begin{pmatrix}p_x \\ p_y\end{pmatrix}.

The corresponding Hessians are

.. math::

    \nabla_{\mathbf{q}}^2V = \begin{pmatrix}1+2y & 2x \\ 2x & 1-2y\end{pmatrix}, \qquad \nabla_{\mathbf{p}}^2T = \begin{pmatrix}1 & 0 \\ 0 & 1\end{pmatrix}.

Define one function for each gradient and Hessian. Every function receives the relevant state vector and a parameter array, even though Hénon-Heiles has no adjustable parameters:

.. code-block:: python

    import numpy as np
    from numba import njit
    from pynamicalsys import HamiltonianSystem as hs

    @njit
    def henon_heiles_grad_V(q, parameters):
        x, y = q
        return np.array([x + 2.0 * x * y, y + x**2 - y**2])

    @njit
    def henon_heiles_grad_T(p, parameters):
        px, py = p
        return np.array([px, py])

    @njit
    def henon_heiles_hess_V(q, parameters):
        x, y = q
        return np.array(
            [
                [1.0 + 2.0 * y, 2.0 * x],
                [2.0 * x, 1.0 - 2.0 * y],
            ]
        )

    @njit
    def henon_heiles_hess_T(p, parameters):
        return np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )

The ``@njit`` decorator compiles the functions for use by the package's Numba-accelerated numerical routines. Keep the functions limited to operations supported by Numba.

Create the custom system with two degrees of freedom and no parameters:

.. code-block:: python

    system = hs(
        grad_T=henon_heiles_grad_T,
        grad_V=henon_heiles_grad_V,
        hess_T=henon_heiles_hess_T,
        hess_V=henon_heiles_hess_V,
        degrees_of_freedom=2,
        parameters=[],
    )

The separable representation selects ``svy4`` by default. Use :py:meth:`integrator <pynamicalsys.core.hamiltonian_systems.HamiltonianSystem.integrator>` to choose ``vv2`` or change the integration step. The gradient functions are sufficient for trajectories, while the Hessians are required for tangent-space calculations.

Creating a general Hamiltonian system
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a general Hamiltonian, provide the equations of motion and the full Hessian with respect to :math:`\mathbf{z}=(\mathbf{q},\mathbf{p})`. Consider the two-resonance Hamiltonian introduced by `Walker and Ford <https://doi.org/10.1103/PhysRev.188.416>`_:

.. math::

    \begin{aligned}
        H(\boldsymbol{\theta},\mathbf{J}) ={}& J_1+J_2-J_1^2-3J_1J_2+J_2^2 \\
        &+\alpha J_1J_2\cos\phi_{22}+\beta J_1J_2^{3/2}\cos\phi_{23},
    \end{aligned}

where :math:`\boldsymbol{\theta}=(\theta_1,\theta_2)` are the angle variables, :math:`\mathbf{J}=(J_1,J_2)` are the action variables, :math:`\phi_{22}=2\theta_1-2\theta_2`, and :math:`\phi_{23}=2\theta_1-3\theta_2`. In the ``HamiltonianSystem`` interface, pass the angles as ``q`` and the actions as ``p``. The parameters are :math:`\alpha` and :math:`\beta`.

The terms containing :math:`J_i\cos\phi` couple the action and angle variables, so this Hamiltonian cannot be written as :math:`T(\mathbf{J})+V(\boldsymbol{\theta})`. The ``svy4`` and ``vv2`` methods split the evolution into separate kinetic and potential steps and therefore cannot integrate this system. It must use the implicit midpoint method, ``imp``.

Hamilton's equations,

.. math::

    \dot{\theta}_i=\frac{\partial H}{\partial J_i}, \qquad \dot{J}_i=-\frac{\partial H}{\partial\theta_i},

give

.. math::

    \begin{aligned}
        \dot{\theta}_1 &= 1-2J_1-3J_2+\alpha J_2\cos\phi_{22}+\beta J_2^{3/2}\cos\phi_{23}, \\
        \dot{\theta}_2 &= 1-3J_1+2J_2+\alpha J_1\cos\phi_{22}+\frac{3}{2}\beta J_1\sqrt{J_2}\cos\phi_{23}, \\
        \dot{J}_1 &= 2\alpha J_1J_2\sin\phi_{22}+2\beta J_1J_2^{3/2}\sin\phi_{23}, \\
        \dot{J}_2 &= -2\alpha J_1J_2\sin\phi_{22}-3\beta J_1J_2^{3/2}\sin\phi_{23}.
    \end{aligned}

For :math:`\mathbf{z}=(\theta_1,\theta_2,J_1,J_2)`, the nonzero entries of the symmetric Hessian are

.. math::

    \begin{aligned}
        H_{\theta_1\theta_1} &= -4\alpha J_1J_2\cos\phi_{22}-4\beta J_1J_2^{3/2}\cos\phi_{23}, \\
        H_{\theta_2\theta_2} &= -4\alpha J_1J_2\cos\phi_{22}-9\beta J_1J_2^{3/2}\cos\phi_{23}, \\
        H_{\theta_1\theta_2} &= 4\alpha J_1J_2\cos\phi_{22}+6\beta J_1J_2^{3/2}\cos\phi_{23}, \\
        H_{\theta_1J_1} &= -2\alpha J_2\sin\phi_{22}-2\beta J_2^{3/2}\sin\phi_{23}, \\
        H_{\theta_1J_2} &= -2\alpha J_1\sin\phi_{22}-3\beta J_1\sqrt{J_2}\sin\phi_{23}, \\
        H_{\theta_2J_1} &= 2\alpha J_2\sin\phi_{22}+3\beta J_2^{3/2}\sin\phi_{23}, \\
        H_{\theta_2J_2} &= 2\alpha J_1\sin\phi_{22}+\frac{9}{2}\beta J_1\sqrt{J_2}\sin\phi_{23}, \\
        H_{J_1J_1} &= -2, \\
        H_{J_1J_2} &= -3+\alpha\cos\phi_{22}+\frac{3}{2}\beta\sqrt{J_2}\cos\phi_{23}, \\
        H_{J_2J_2} &= 2+\frac{3}{4}\beta\frac{J_1}{\sqrt{J_2}}\cos\phi_{23}.
    \end{aligned}

Define ``eom`` to return ``(qdot, pdot)`` in that order, then assemble the symmetric Hessian:

.. code-block:: python

    import numpy as np
    from numba import njit
    from pynamicalsys import HamiltonianSystem as hs

    @njit
    def walker_ford_eom(q, p, parameters):
        theta1, theta2 = q
        J1, J2 = p
        alpha, beta = parameters

        sqrt_J2 = np.sqrt(J2)
        J2_32 = J2 * sqrt_J2
        phase_22 = 2.0 * theta1 - 2.0 * theta2
        phase_23 = 2.0 * theta1 - 3.0 * theta2
        cos_22 = np.cos(phase_22)
        cos_23 = np.cos(phase_23)
        sin_22 = np.sin(phase_22)
        sin_23 = np.sin(phase_23)

        qdot = np.empty(2)
        pdot = np.empty(2)
        qdot[0] = 1.0 - 2.0 * J1 - 3.0 * J2 + alpha * J2 * cos_22 + beta * J2_32 * cos_23
        qdot[1] = 1.0 - 3.0 * J1 + 2.0 * J2 + alpha * J1 * cos_22 + 1.5 * beta * J1 * sqrt_J2 * cos_23
        pdot[0] = 2.0 * alpha * J1 * J2 * sin_22 + 2.0 * beta * J1 * J2_32 * sin_23
        pdot[1] = -2.0 * alpha * J1 * J2 * sin_22 - 3.0 * beta * J1 * J2_32 * sin_23
        return qdot, pdot

    @njit
    def walker_ford_hess_H(q, p, parameters):
        theta1, theta2 = q
        J1, J2 = p
        alpha, beta = parameters

        sqrt_J2 = np.sqrt(J2)
        J2_32 = J2 * sqrt_J2
        phase_22 = 2.0 * theta1 - 2.0 * theta2
        phase_23 = 2.0 * theta1 - 3.0 * theta2
        cos_22 = np.cos(phase_22)
        cos_23 = np.cos(phase_23)
        sin_22 = np.sin(phase_22)
        sin_23 = np.sin(phase_23)

        common_22 = alpha * J1 * J2
        common_23 = beta * J1 * J2_32
        H = np.zeros((4, 4))

        H[0, 0] = -4.0 * common_22 * cos_22 - 4.0 * common_23 * cos_23
        H[1, 1] = -4.0 * common_22 * cos_22 - 9.0 * common_23 * cos_23
        H[0, 1] = 4.0 * common_22 * cos_22 + 6.0 * common_23 * cos_23
        H[1, 0] = H[0, 1]

        H[0, 2] = -2.0 * alpha * J2 * sin_22 - 2.0 * beta * J2_32 * sin_23
        H[2, 0] = H[0, 2]
        H[0, 3] = -2.0 * alpha * J1 * sin_22 - 3.0 * beta * J1 * sqrt_J2 * sin_23
        H[3, 0] = H[0, 3]
        H[1, 2] = 2.0 * alpha * J2 * sin_22 + 3.0 * beta * J2_32 * sin_23
        H[2, 1] = H[1, 2]
        H[1, 3] = 2.0 * alpha * J1 * sin_22 + 4.5 * beta * J1 * sqrt_J2 * sin_23
        H[3, 1] = H[1, 3]

        H[2, 2] = -2.0
        H[2, 3] = -3.0 + alpha * cos_22 + 1.5 * beta * sqrt_J2 * cos_23
        H[3, 2] = H[2, 3]
        H[3, 3] = 2.0 + 0.75 * beta * J1 * cos_23 / sqrt_J2
        return H

    system = hs(
        eom=walker_ford_eom,
        hess_H=walker_ford_hess_H,
        degrees_of_freedom=2,
        parameters=[0.02, 0.02],
    )

This construction selects ``imp`` automatically. The functions require :math:`J_2>0` because the Hamiltonian and Hessian contain :math:`\sqrt{J_2}` and :math:`1/\sqrt{J_2}`.

The later trajectory tutorial uses this distinction when comparing the available integrators. The explicit ``svy4`` and ``vv2`` methods apply to the separable Hénon-Heiles construction, while ``imp`` applies to the general Walker-Ford construction.

The ``info`` property describes built-in models. The package cannot infer descriptive metadata or equations from custom functions.

References
~~~~~~~~~~

.. container:: references-list

    - G\. H\. Walker and J\. Ford, `Amplitude instability and ergodic behavior for conservative nonlinear oscillator systems <https://doi.org/10.1103/PhysRev.188.416>`_, Physical Review 188, 416-432 (1969).
