Creating a discrete dynamical system
------------------------------------

The :py:class:`HamiltonianSystem <pynamicalsys.core.hamiltonian_systems.HamiltonianSystem>` class allows you to create a discrete dynamical system object. You can use built-in systems or define your own continuous dynamical system.

Using built-in systems
~~~~~~~~~~~~~~~~~~~~~~

To check available built-in systems, you can use the :py:meth:`available_models <pynamicalsys.core.hamiltonian_systems.HamiltonianSystem.available_models>` method:

.. code-block:: python

    available_models = hs.available_models()
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

    hs = hs(model="henon heiles")

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
