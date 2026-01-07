Creating a discrete dynamical system
------------------------------------

The :py:class:`DiscreteDynamicalSystem <pynamicalsys.core.discrete_dynamical_systems.DiscreteDynamicalSystem>` class allows you to create a discrete dynamical system object. You can use built-in systems or define your own discrete maps.

Using built-in systems
~~~~~~~~~~~~~~~~~~~~~~

To check available built-in systems, you can use the :py:meth:`available_models <pynamicalsys.core.discrete_dynamical_systems.DiscreteDynamicalSystem.available_models>` method:

.. code-block:: python

    available_models = dds.available_models()
    print(available_models)

.. code-block:: text

    ['standard map',
    'unbounded standard map',
    'henon map',
    'lozi map',
    'rulkov map',
    'logistic map',
    'standard nontwist map',
    'extended standard nontwist map',
    'leonel map',
    '4d symplectic map']

For example, you can create a Chirikov-Taylor standard map, given by:

.. math::

    \begin{align*}
        y_{n+1} &= y_n + \frac{k}{2\pi} \sin(2\pi x_n) \bmod1,\\
        x_{n+1} &= x_n + y_{n+1} \bmod1,
    \end{align*}
    
where :math:`k` is a constant. You can create this system using:

.. code-block:: python

    ds = dds(model="standard map")

and then all the methods available for the :py:class:`DiscreteDynamicalSystem <pynamicalsys.core.discrete_dynamical_systems.DiscreteDynamicalSystem>` class can be used to run simulations and analyze the system.

Creating custom discrete maps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can also create your own discrete maps by defining a function that takes the current state and a list of parameters, and returns the next state. For example, let us create the standard map as a custom function:

.. code-block:: python

    from numba import njit

    @njit
    def standard_map(state, params):
        k = params[0]
        x, y = state
        y_next = (y + k / (2 * np.pi) * np.sin(2 * np.pi * x)) % 1
        x_next = (x + y_next) % 1
        return np.array([x_next, y_next])

Note that we use `numba` to compile the function for performance. Most methods inside the :py:class:`DiscreteDynamicalSystem <pynamicalsys.core.discrete_dynamical_systems.DiscreteDynamicalSystem>` class are decoreted with numba. Therefore, it is absolute necessary that all custom mapping function be decoreted with it as well. You can then create a discrete dynamical system object with this custom function by informing the mapping function, the system dimension and the number of parameters the system has:

.. code-block:: python

    ds = dds(mapping=standard_map, system_dimension=2, number_of_parameters=1)

An alternative is to inform the list of parameters instead of the number of them:

.. code-block:: python

    parameters = [1.5]  # parameters = 1.5 works as well for single values
    ds = dds(mapping=standard_map, system_dimension=2, parameters=parameters)
    print(ds.get_parameters())

.. code-block:: text
    [1.5]

After creating the object, the parameters passed to the constructor are stored internally and used by default by all methods of the :py:class:`DiscreteDynamicalSystem <pynamicalsys.core.discrete_dynamical_systems.DiscreteDynamicalSystem>` instance. In this configuration, every method call that does not explicitly specify parameters will use the internally stored value ([1.5]). You can permanently modify these stored parameters using the
:py:meth:`set_parameters <pynamicalsys.core.discrete_dynamical_systems.DiscreteDynamicalSystem.set_parameters>` method:

.. code-block:: python

    ds.set_parameters([4.0])  # ds.set_parameters(4.0) works as well for single values
    print(ds.get_parameters())

.. code-block:: text
    [4.0]

This updates the parameters at the object level, so all subsequent method calls will now use [4.0] by default. Finally, all methods of :py:class:`DiscreteDynamicalSystem <pynamicalsys.core.discrete_dynamical_systems.DiscreteDynamicalSystem>` also accept a parameters argument. When this argument is provided, it temporarily overrides the internally stored parameters for that specific method call only. The parameters stored in the object remain unchanged.

.. note::

   In other words:

   - ``set_parameters(...)`` → persistent change (updates the system's internal parameters)
   - ``parameters=...`` in a method call → temporary, local override (applies only to that call)
