Creating a continuous dynamical system
--------------------------------------

The :py:class:`ContinuousDynamicalSystem <pynamicalsys.core.continuous_dynamical_systems.ContinuousDynamicalSystem>` class represents systems defined by ordinary differential equations. You can select a built-in model or provide your own equations of motion.

Using a built-in model
~~~~~~~~~~~~~~~~~~~~~~

Import the class as ``cds`` and call :py:meth:`available_models <pynamicalsys.core.continuous_dynamical_systems.ContinuousDynamicalSystem.available_models>` to see the built-in models:

.. code-block:: python

    from pynamicalsys import ContinuousDynamicalSystem as cds

    for model in cds.available_models():
        print(model)

.. code-block:: text

    lorenz system
    henon heiles
    rossler system
    4d rossler system
    duffing

Select a model by passing its name to ``model``. For example, the Lorenz system is

.. math::

    \begin{aligned}
        \dot{x} &= \sigma(y-x), \\
        \dot{y} &= x(\rho-z)-y, \\
        \dot{z} &= xy-\beta z.
    \end{aligned}

Create the system and inspect its parameter order with the ``info`` property:

.. code-block:: python

    system = cds(model="lorenz system")
    print(system.info["parameters"])

.. code-block:: text

    ['sigma', 'rho', 'beta']

The state is ordered as ``[x, y, z]``, and parameters must follow the order shown by ``info``. The methods of ``system`` can now integrate trajectories and perform the analyses supported by the class.

Creating a custom system
~~~~~~~~~~~~~~~~~~~~~~~~

A custom equations-of-motion function receives the current time, state, and one-dimensional parameter array, then returns the time derivative of the state. The time argument is required even for an autonomous system such as the Lorenz system:

.. code-block:: python

    import numpy as np
    from numba import njit
    from pynamicalsys import ContinuousDynamicalSystem as cds

    @njit
    def lorenz_system(time, state, parameters):
        x, y, z = state
        sigma, rho, beta = parameters
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        return np.array([dx, dy, dz])

    @njit
    def lorenz_jacobian(time, state, parameters):
        x, y, z = state
        sigma, rho, beta = parameters
        return np.array(
            [
                [-sigma, sigma, 0.0],
                [rho - z, -1.0, -x],
                [y, x, -beta],
            ]
        )

The ``@njit`` decorator compiles both functions for use by the package's Numba-accelerated numerical routines. Keep the functions limited to operations supported by Numba.

When the parameter values are already known, pass them while creating the system:

.. code-block:: python

    parameters = [10.0, 28.0, 8.0 / 3.0]
    system = cds(
        equations_of_motion=lorenz_system,
        jacobian=lorenz_jacobian,
        system_dimension=3,
        parameters=parameters,
    )

Here, ``system_dimension=3`` corresponds to the three state variables ``[x, y, z]``. Supplying ``parameters`` also tells the class that the system expects three parameters and stores their values for later method calls.

If you want to provide the parameters only when performing a calculation, declare their number instead:

.. code-block:: python

    system = cds(
        equations_of_motion=lorenz_system,
        jacobian=lorenz_jacobian,
        system_dimension=3,
        number_of_parameters=3,
    )

    final_state = system.evolve_system(
        [1.0, 1.0, 1.0],
        total_time=0.1,
        parameters=[10.0, 28.0, 8.0 / 3.0],
    )

In this case, a method call must supply ``parameters`` until values are stored with :py:meth:`set_parameters <pynamicalsys.core.continuous_dynamical_systems.ContinuousDynamicalSystem.set_parameters>`.

Managing parameters
~~~~~~~~~~~~~~~~~~~

Use ``set_parameters`` to store parameter values for subsequent calculations:

.. code-block:: python

    system.set_parameters([10.0, 28.0, 8.0 / 3.0])
    print(system.get_parameters())

.. code-block:: text

    [10.         28.          2.66666667]

Methods use the stored parameters when their ``parameters`` argument is omitted. Passing the argument to an individual call temporarily overrides the stored values:

.. code-block:: python

    stored_result = system.evolve_system([1.0, 1.0, 1.0], total_time=0.1)
    override_result = system.evolve_system(
        [1.0, 1.0, 1.0],
        total_time=0.1,
        parameters=[11.0, 20.0, 3.0],
    )
    print(system.get_parameters())

.. code-block:: text

    [10.         28.          2.66666667]

The second call uses the temporary parameter values, but the values stored in ``system`` remain unchanged. Calling ``system.set_parameters([11.0, 20.0, 3.0])`` would replace the stored values for all later calls.

Providing a Jacobian
~~~~~~~~~~~~~~~~~~~~

The equations of motion alone are enough to integrate trajectories. A custom Jacobian must accept the same ``(time, state, parameters)`` arguments and return the matrix of partial derivatives with respect to the state variables.

The Jacobian is required for tangent-space calculations such as Lyapunov exponents, covariant Lyapunov vectors, SALI, LDI, and GALI. Decorate it with ``@njit`` and pass it as ``jacobian=...`` when constructing the system, as shown in the Lorenz example above.

The ``info`` property describes built-in models. The package cannot infer descriptive metadata or equations from custom functions.
