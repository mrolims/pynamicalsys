Creating a discrete dynamical system
------------------------------------

The :py:class:`ContinuousDynamicalSystem <pynamicalsys.core.discrete_dynamical_systems.ContinuousDynamicalSystem>` class allows you to create a discrete dynamical system object. You can use built-in systems or define your own continuous dynamical system.

Using built-in systems
~~~~~~~~~~~~~~~~~~~~~~

To check available built-in systems, you can use the :py:meth:`available_models <pynamicalsys.core.discrete_dynamical_systems.ContinuousDynamicalSystem.available_models>` method:

.. code-block:: python

    available_models = cds.available_models()
    print(available_models)

.. code-block:: text

    ['lorenz system']

For example, you can create the 86 Lorenz system, given by:

.. math::

    \begin{align*}
        \dot{x} &= \sigma(y - x),\\
        \dot{y} &= x(\rho - z) - y,\\
        \dot{z} &= xy - \beta z.
    \end{align*}
    
You can create this system using:

.. code-block:: python

    ds = cds(model="lorenz system")

and then all the methods available for the :py:class:`ContinuousDynamicalSystem <pynamicalsys.core.discrete_dynamical_systems.ContinuousDynamicalSystem>` class can be used to run simulations and analyze the system.

Creating custom continuous dynamical systems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can also create your own continuous system by defining a function that takes the current state and a list of parameters, and returns the time derivative of the state. For example, let us create the Lorenz system as a custom function:

.. code-block:: python

    from numba import njit

    @njit
    def lorenz_system(state, params):
        sigma, rho, beta = params
        x, y, z = state
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z

        return np.array([dx, dy, dz])

Note that we use `numba` to compile the function for performance. Most methods inside the :py:class:`ContinuousDynamicalSystem <pynamicalsys.core.discrete_dynamical_systems.ContinuousDynamicalSystem>` class are decoreted with numba. Therefore, it is absolute necessary that all custom mapping function be decoreted with it as well. You can then create a discrete dynamical system object with this custom function:

.. code-block:: python

    ds = cds(mapping=lorenz_system, system_dimension=3, number_of_parameters=3)
