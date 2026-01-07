Creating a continuous dynamical system
--------------------------------------

The :py:class:`ContinuousDynamicalSystem <pynamicalsys.core.continuous_dynamical_systems.ContinuousDynamicalSystem>` class allows you to create a continuous dynamical system object. You can use built-in systems or define your own continuous dynamical system.

Using built-in systems
~~~~~~~~~~~~~~~~~~~~~~

To check available built-in systems, you can use the :py:meth:`available_models <pynamicalsys.core.continuous_dynamical_systems.ContinuousDynamicalSystem.available_models>` method:

.. code-block:: python

    available_models = cds.available_models()
    print(available_models)

.. code-block:: text

    ['lorenz system']

For example, you can create the 63 Lorenz system, given by:

.. math::

    \begin{align*}
        \dot{x} &= \sigma(y - x),\\
        \dot{y} &= x(\rho - z) - y,\\
        \dot{z} &= xy - \beta z.
    \end{align*}
    
You can create this system using:

.. code-block:: python

    ds = cds(model="lorenz system")

and then all the methods available for the :py:class:`ContinuousDynamicalSystem <pynamicalsys.core.continuous_dynamical_systems.ContinuousDynamicalSystem>` class can be used to run simulations and analyze the system.

Creating custom continuous dynamical systems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can also create your own continuous system by defining a function that takes the current state and a list of parameters, and returns the time derivative of the state. For example, let us create the Lorenz system as a custom function:

.. code-block:: python

    from numba import njit

    @njit
    def lorenz_system(time, state, params):
        sigma, rho, beta = params
        x, y, z = state
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z

        return np.array([dx, dy, dz])

Note that we use :code:`@njit` to compile the function for performance. Most methods inside the :py:class:`ContinuousDynamicalSystem <pynamicalsys.core.continuous_dynamical_systems.ContinuousDynamicalSystem>` class are decoreted with :code:`@njit`. Therefore, it is absolute necessary that all custom mapping function be decoreted with it as well. You can then create a continuous dynamical system object with this custom function:

.. code-block:: python

    ds = cds(equations_of_motion=lorenz_system, system_dimension=3, number_of_parameters=3)

An alternative is to inform the list of parameters instead of the number of them:

.. code-block:: python

    sigma, rho, beta = 10.0, 28.0, 8/3
    parameters = [sigma, rho, beta]
    ds = cds(equations_of_motion=lorenz_system, system_dimension=3, parameters=parameters)
    print(ds.get_parameters())

.. code-block:: text
    [10.0, 28.0, 2.6666666666666665]

After creating the object, the parameters passed to the constructor are stored internally and used by default by all methods of the :py:class:`ContinuousDynamicalSystem <pynamicalsys.core.continuous_dynamical_systems.ContinuousDynamicalSystem>` instance. In this configuration, every method call that does not explicitly specify parameters will use the internally stored value ([10.0, 28.0, 2.6666666666666665]). You can permanently modify these stored parameters using the
:py:meth:`set_parameters <pynamicalsys.core.continuous_dynamical_systems.ContinuousDynamicalSystem.set_parameters>` method:

.. code-block:: python

    sigma, rho, beta = 11.0, 20.0, 3
    parameters = [sigma, rho, beta]
    ds.set_parameters(parameters)
    print(ds.get_parameters())

.. code-block:: text
    [11.0, 20.0, 3.0]

This updates the parameters at the object level, so all subsequent method calls will now use [11.0, 20.0, 3.0] by default. Finally, all methods of :py:class:`ContinuousDynamicalSystem <pynamicalsys.core.continuous_dynamical_systems.ContinuousDynamicalSystem>` also accept a parameters argument. When this argument is provided, it temporarily overrides the internally stored parameters for that specific method call only. The parameters stored in the object remain unchanged.

.. note::

   In other words:

   - ``set_parameters(...)`` → persistent change (updates the system's internal parameters)
   - ``parameters=...`` in a method call → temporary, local override (applies only to that call)
