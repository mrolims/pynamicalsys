Creating a Hamiltonian system
-----------------------------

The :py:class:`HamiltonianSystem <pynamicalsys.core.hamiltonian_systems.HamiltonianSystem>` class allows you to create a discrete dynamical system object. You can use built-in systems or define your own continuous dynamical system.

Using built-in systems
~~~~~~~~~~~~~~~~~~~~~~

To check available built-in systems, you can use the :py:meth:`available_models <pynamicalsys.core.hamiltonian_systems.HamiltonianSystem.available_models>` method:

.. code-block:: python

    available_models = HamiltonianSystem.available_models()
    print(available_models)

.. code-block:: text

    ['henon heiles']

For example, you can create the two degrees of freedom Hénon-Heiles object, given by the Hamiltonian function

.. math::

    \begin{align*}
        H(x, y, p_x, p_y) = \frac{1}{2}(p_x^2 + p_y^2) + \frac{1}{2}(x^2 + y^2) + x^2y - \frac{y^3}{3},
    \end{align*}
    
using:

.. code-block:: python

    ds = HamiltonianSystem(model="henon heiles")

and then all the methods available for the :py:class:`HamiltonianSystem <pynamicalsys.core.hamiltonian_systems.HamiltonianSystem>` class can be used to run simulations and analyze the system.

Creating custom separable Hamiltonian systems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A separable Hamiltonian system is a system where the kinetic energy depends only on the momenta and the potential energy depends only on the coordinates, i,e., :math:`H(\mathbf{q}, \mathbf{q}) = T(\mathbf{p}) + V(\mathbf{q})`. For this type of Hamiltonian, the available integrators are the 2nd order velocity Verlet and the 4th order Yoshida. By default, the class uses the 4th order Yosida. You can select a differnt integrator by calling the :py:meth:`integrator <pynamicalsys.core.hamiltonian_system.HamiltonianSystem.integrator>` method:

.. code-block::python
    ds.integrator("vv2", time_step=0.001)
    ds.integrator("svy4", time_step=0.001)

To list all the available integrators, use the :py:meth:`available_integrators <pynamicalsys.core.hamiltonian_systems.HamiltonianSystem.available_integrators>`

.. code-block:: python
    HamiltonianSystem.available_integrators()

.. code-block:: text

    ['vv2', 'svy4', 'imp']

You can also create your own separable Hamiltonian system by defining a function that calculates the gradient of the kinetic and potential energy. For example, for the Hénon-Heiles system, whose Hamiltonian is separable, we have

.. math::
    \begin{equation*}
        \frac{\partial T}{\partial \mathbf{p}} =
        \begin{pmatrix}
            p_x \\[0.3em]
            p_y
        \end{pmatrix},
        \qquad
        \frac{\partial V}{\partial\mathbf{q}} =
        \begin{pmatrix}
            x + 2xy \\[0.3em]
            y + x^2 - y^2
        \end{pmatrix}.
    \end{equation*}

We then define functions that take the generalized coordinates and the parameters and generalized momenta and parameters:

.. code-block:: python
    
    from numba import njit
    
    @njit
    def henon_heiles_grad_T(p, parameters=None):
        return np.array([p[0], p[1]])
    
    @njit
    def henon_heiles_grad_V(q, parameters=None):
        q0, q1 = q[0], q[1]
        dV_dq0 = q0 * (1.0 + 2.0 * q1)
        dV_dq1 = q1 + q0 * q0 - q1 * q1
        return np.array([dV_dq0, dV_dq1])    

Note that we use :code:`@njit` to compile the function for performance. Most methods inside the :py:class:`HamiltonianSystem <pynamicalsys.core.hamiltonian_systems.HamiltonianSystem>` class are decoreted with :code:`@njit`. Therefore, it is absolute necessary that all custom mapping function be decoreted with it as well. You can then create a Hamiltonian system object with this custom function:

.. code-block:: python

    ds = hs(
        grad_T=henon_heiles_grad_T,
        grad_V=henon_heiles_grad_V,
        degrees_of_freedom=2,
        number_of_parameters=0,
    )

An alternative is to inform the list of parameters instead of the number of them. Since the Hénon-Heiles system has no parameter, we can pass an empty list to the `parameters` argument

.. code-block:: python

    ds = hs(
        grad_T=henon_heiles_grad_T,
        grad_V=henon_heiles_grad_V,
        degrees_of_freedom=2,
        parameters=[],
    )
    print(ds.get_parameters())

.. code-block:: text
    []

After creating the object, the parameters passed to the constructor are stored internally and used by default by all methods of the :py:class:`HamiltonianSystem <pynamicalsys.core.hamiltonian_systems.HamiltonianSystem>` instance. In this configuration, every method call that does not explicitly specify parameters will use the internally stored value ([]). You can permanently modify these stored parameters using the
:py:meth:`set_parameters <pynamicalsys.core.hamiltonian_systems.HamiltonianSystem.set_parameters>` method:

.. code-block:: python

    ds.set_parameters([4.0])  # ds.set_parameters(4.0) works as well for single values

This updates the parameters at the object level, so all subsequent method calls will use [4.0] by default.
Note that for the Hénon–Heiles system this would result in an error, since the system does not take any parameters.
Nevertheless, setting parameters in this way is valid for any Hamiltonian system that depends on a nonzero number of parameters. Finally, all methods of :py:class:`HamiltonianSystem <pynamicalsys.core.hamiltonian_systems.HamiltonianSystem>` also accept a parameters argument. When this argument is provided, it temporarily overrides the internally stored parameters for that specific method call only. The parameters stored in the object remain unchanged.

.. note::

   In other words:

   - ``set_parameters(...)`` → persistent change (updates the system's internal parameters)
   - ``parameters=...`` in a method call → temporary, local override (applies only to that call)

Creating custom general Hamiltonian systems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the general case, the system is defined directly from the full Hamiltonian :math:`H(\mathbf{q}, \mathbf{p})`. The user must provide functions that compute the Hamiltonian equations of motion, given by Hamilton’s equations, as well as the Hessian of :math:`H` with respect to the phase-space variables :math:`\mathbf{z} = (\mathbf{q}, \mathbf{p})`.

As an example, let's consider the following Hamiltonian, from the classical work of `Walker and Ford <https://doi.org/10.1103/PhysRev.188.416>`_:

.. math::
    \begin{align*}
        H(J_1, J_2, \theta_1, \theta_2) &= J_1 + J_2 - J_1^2 - 3J_1J_2 + J_2^2 + \\
        &+ \alpha J_1J_2 \cos(2\theta_1 - 2\theta_2) + \\
        &+ \beta J_1J_2^{3/2}\cos(2\theta_1 - 3\theta_2),
    \end{align*}

where :math:`\boldsymbol{\theta}=(\theta_1,\theta_2)` are the angles, :math:`\mathbf{J}=(J_1,J_2)` are the actions, and :math:`\alpha` and :math:`\beta` are the system parameters. The corresponding equations of motion follow from Hamilton's equations,

.. math::

    \dot{\theta}_i=\frac{\partial H}{\partial J_i}, \qquad \dot{J}_i=-\frac{\partial H}{\partial \theta_i},

yielding

.. math::

    \begin{align*}
        \dot{\theta}_1 &= 1 - 2J_1 - 3J_2 + \alpha J_2\cos(2\theta_1 - 2\theta_2) + \beta J_2^{3/2}\cos(2\theta_1 - 3\theta_2), \\
        \dot{\theta}_2 &= 1 - 3J_1 + 2J_2 + \alpha J_1\cos(2\theta_1 - 2\theta_2) +\frac{3}{2}\beta J_1\sqrt{J_2}\cos(2\theta_1 - 3\theta_2), \\
        \dot{J}_1 &= 2\alpha J_1J_2\sin(2\theta_1 - 2\theta_2) + 2\beta J_1J_2^{3/2}\sin(2\theta_1 - 3\theta_2), \\
        \dot{J}_2 &= -2\alpha J_1J_2\sin(2\theta_1 - 2\theta_2) - 3\beta J_1J_2^{3/2}\sin(2\theta_1 - 3\theta_2).
    \end{align*}

The Hessian of the Hamiltonian with respect to the phase-space variables :math:`(\theta_1, \theta_2, J_1, J_2)` is

.. math::
    \begin{equation*}
        \nabla^2 H =\left(\frac{\partial^2 H}{\partial z_i \partial z_j}\right), \qquad \mathbf{z} = (\theta_1, \theta_2, J_1, J_2).
    \end{equation*}

Its nonzero entries are

.. math::
    \begin{align*}
        \frac{\partial^2 H}{\partial \theta_1^2} &= -4\alpha J_1J_2\cos(2\theta_1 - 2\theta_2) - 4\beta J_1J_2^{3/2}\cos(2\theta_1 - 3\theta_2), \\
        \frac{\partial^2 H}{\partial \theta_2^2} &= -4\alpha J_1J_2\cos(2\theta_1 - 2\theta_2) - 9\beta J_1J_2^{3/2}\cos(2\theta_1 - 3\theta_2), \\
        \frac{\partial^2 H}{\partial \theta_1 \partial \theta_2} &= 4\alpha J_1J_2\cos(2\theta_1 - 2\theta_2) + 6\beta J_1J_2^{3/2}\cos(2\theta_1 - 3\theta_2), \\
        \frac{\partial^2 H}{\partial \theta_1 \partial J_1} &= -2\alpha J_2\sin(2\theta_1 - 2\theta_2) - 2\beta J_2^{3/2}\sin(2\theta_1 - 3\theta_2), \\
        \frac{\partial^2 H}{\partial \theta_1 \partial J_2} &= -2\alpha J_1\sin(2\theta_1 - 2\theta_2) - 3\beta J_1\sqrt{J_2}\sin(2\theta_1 - 3\theta_2), \\
        \frac{\partial^2 H}{\partial \theta_2 \partial J_1} &= 2\alpha J_2\sin(2\theta_1 - 2\theta_2) + 3\beta J_2^{3/2}\sin(2\theta_1 - 3\theta_2), \\
        \frac{\partial^2 H}{\partial \theta_2 \partial J_2} &= 2\alpha J_1\sin(2\theta_1 - 2\theta_2) + \frac{9}{2}\beta J_1\sqrt{J_2}\sin(2\theta_1 - 3\theta_2), \\
        \frac{\partial^2 H}{\partial J_1^2} &= -2, \\
        \frac{\partial^2 H}{\partial J_1 \partial J_2} &= -3 + \alpha\cos(2\theta_1 - 2\theta_2) + \frac{3}{2}\beta\sqrt{J_2}\cos(2\theta_1 - 3\theta_2), \\
        \frac{\partial^2 H}{\partial J_2^2} &= 2 + \frac{3}{4}\beta\frac{J_1}{\sqrt{J_2}}\cos(2\theta_1 - 3\theta_2).
    \end{align*}

We use these equations then to define the custom functions:

.. code-block:: python

    from numba import njit
    import numpy as np

    @njit
    def two_res_eom(q, p, parameters):
        """
        Equations of motion for the two-resonance Hamiltonian [1]
            H = J1 + J2 - J1^2 - 3*J1*J2 + J2^2
                + alpha*J1*J2*cos(2*theta1 - 2*theta2)
                + beta*J1*J2^1.5*cos(2*theta1 - 3*theta2).
        Returns (dq/dt, dp/dt) = (dH/dp, -dH/dq).
        parameters[0] = alpha, parameters[1] = beta.

        [1] G. H. Walker and J. Ford, Amplitude Instability and Ergodic Behavior for
        Conservative Nonlinear Oscillator Systems, Phys. Rev. 188, 416 (1969)
        """
        theta1 = q[0]
        theta2 = q[1]
        J1 = p[0]
        J2 = p[1]

        alpha = parameters[0]
        beta = parameters[1]

        J2_12 = np.sqrt(J2)
        J2_32 = J2 * J2_12

        phase_22 = 2.0 * theta1 - 2.0 * theta2
        phase_23 = 2.0 * theta1 - 3.0 * theta2
        c22 = np.cos(phase_22)
        c23 = np.cos(phase_23)
        s22 = np.sin(phase_22)
        s23 = np.sin(phase_23)

        qdot = np.empty(2)
        pdot = np.empty(2)

        qdot[0] = 1.0 - 2.0 * J1 - 3.0 * J2 + alpha * J2 * c22 + beta * J2_32 * c23
        qdot[1] = (
            1.0 - 3.0 * J1 + 2.0 * J2 + alpha * J1 * c22 + 1.5 * beta * J1 * J2_12 * c23
        )
        pdot[0] = 2.0 * alpha * J1 * J2 * s22 + 2.0 * beta * J1 * J2_32 * s23
        pdot[1] = -2.0 * alpha * J1 * J2 * s22 - 3.0 * beta * J1 * J2_32 * s23

        return qdot, pdot


    @njit
    def two_res_hess_H(q, p, parameters):
        """
        Full Hessian of the two-resonance Hamiltonian H(q, p) w.r.t. the
        combined state z = (theta1, theta2, J1, J2), shape (4, 4).
        parameters[0] = alpha, parameters[1] = beta.
        """
        theta1 = q[0]
        theta2 = q[1]
        J1 = p[0]
        J2 = p[1]

        alpha = parameters[0]
        beta = parameters[1]

        J2_12 = np.sqrt(J2)
        J2_32 = J2 * J2_12

        phase_22 = 2.0 * theta1 - 2.0 * theta2
        phase_23 = 2.0 * theta1 - 3.0 * theta2
        c22 = np.cos(phase_22)
        s22 = np.sin(phase_22)
        c23 = np.cos(phase_23)
        s23 = np.sin(phase_23)

        common22 = alpha * J1 * J2
        common23 = beta * J1 * J2_32

        d2H_dtheta1_2 = -4.0 * common22 * c22 - 4.0 * common23 * c23
        d2H_dtheta2_2 = -4.0 * common22 * c22 - 9.0 * common23 * c23
        d2H_dtheta1_dtheta2 = 4.0 * common22 * c22 + 6.0 * common23 * c23

        d2H_dtheta1_dJ1 = -2.0 * alpha * J2 * s22 - 2.0 * beta * J2_32 * s23
        d2H_dtheta2_dJ1 = 2.0 * alpha * J2 * s22 + 3.0 * beta * J2_32 * s23

        d2H_dtheta1_dJ2 = -2.0 * alpha * J1 * s22 - 3.0 * beta * J1 * J2_12 * s23
        d2H_dtheta2_dJ2 = 2.0 * alpha * J1 * s22 + 4.5 * beta * J1 * J2_12 * s23

        d2H_dJ1_2 = -2.0
        d2H_dJ2_2 = 2.0 + beta * J1 * (0.75 / J2_12) * c23
        d2H_dJ1_dJ2 = -3.0 + alpha * c22 + 1.5 * beta * J2_12 * c23

        H = np.zeros((4, 4))

        H[0, 0] = d2H_dtheta1_2
        H[0, 1] = d2H_dtheta1_dtheta2
        H[0, 2] = d2H_dtheta1_dJ1
        H[0, 3] = d2H_dtheta1_dJ2

        H[1, 0] = d2H_dtheta1_dtheta2
        H[1, 1] = d2H_dtheta2_2
        H[1, 2] = d2H_dtheta2_dJ1
        H[1, 3] = d2H_dtheta2_dJ2

        H[2, 0] = d2H_dtheta1_dJ1
        H[2, 1] = d2H_dtheta2_dJ1
        H[2, 2] = d2H_dJ1_2
        H[2, 3] = d2H_dJ1_dJ2

        H[3, 0] = d2H_dtheta1_dJ2
        H[3, 1] = d2H_dtheta2_dJ2
        H[3, 2] = d2H_dJ1_dJ2
        H[3, 3] = d2H_dJ2_2

        return H

The Hamiltonian system object is then created by instancianting the :py:class:`HamiltonianSystem <pynamicalsys.core.hamiltonian_systems.HamiltonianSystem>` class using the `eom` and `hess_H` parameters:

.. code-block:: python

    ds = HamiltonianSystem(
        eom=two_res_eom,
        hess_H=two_res_hess_H,
        degrees_of_freedom=2,
        parameters=[0.02, 0.02],
    )
