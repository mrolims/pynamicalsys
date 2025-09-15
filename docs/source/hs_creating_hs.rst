Creating a discrete dynamical system
------------------------------------

The :py:class:`HamiltonianSystem <pynamicalsys.core.hamiltonian_systems.HamiltonianSystem>` class allows you to create a discrete dynamical system object. You can use built-in systems or define your own continuous dynamical system.

Using built-in systems
~~~~~~~~~~~~~~~~~~~~~~

To check available built-in systems, you can use the :py:meth:`available_models <pynamicalsys.core.hamiltonian_systems.HamiltonianSystem.available_models>` method:

.. code-block:: python

    available_models = HS.available_models()
    print(available_models)

.. code-block:: text

    ['henon heiles']

For example, you can create the two degrees of freedom Hénon-Heiles, given by the Hamiltonian function

.. math::

    \begin{align*}
        H(x, y, p_x, p_y) = \frac{1}{2}(p_x^2 + p_y^2) + \frac{1}{2}(x^2 + y^2) + x^2y - \frac{y^3}{3},
    \end{align*}
    
using:

.. code-block:: python

    hs = HS(model="henon heiles")

and then all the methods available for the :py:class:`HamiltonianSystem <pynamicalsys.core.hamiltonian_systems.HamiltonianSystem>` class can be used to run simulations and analyze the system.

Creating custom Hamiltonian systems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can also create your own Hamiltonian system by defining a function that calculates the gradient of the kinetic and potential energy. For example, for the Hénon-Heiles system, we have

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

    hs = HS(
        grad_T=henon_heiles_grad_T,
        grad_V=henon_heiles_grad_V,
        system_dimension=4,
        number_of_parameters=0,
    )
