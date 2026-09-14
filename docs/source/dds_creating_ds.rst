Creating a discrete dynamical system
------------------------------------

The :py:class:`DiscreteDynamicalSystem <pynamicalsys.core.discrete_dynamical_systems.DiscreteDynamicalSystem>` class represents maps that advance a state from one iteration to the next. You can select a built-in model or provide your own mapping function.

Using a built-in model
~~~~~~~~~~~~~~~~~~~~~~

Import the class as ``dds`` and call :py:meth:`available_models <pynamicalsys.core.discrete_dynamical_systems.DiscreteDynamicalSystem.available_models>` to see the built-in models:

.. code-block:: python

    from pynamicalsys import DiscreteDynamicalSystem as dds

    for model in dds.available_models():
        print(model)

.. code-block:: text

    standard map
    unbounded standard map
    henon map
    lozi map
    rulkov map
    logistic map
    standard nontwist map
    extended standard nontwist map
    leonel map
    4d symplectic map

Select a model by passing its name to ``model``. For example, the Chirikov-Taylor standard map is

.. math::

    \begin{aligned}
        y_{n+1} &= y_n + \frac{k}{2\pi}\sin(2\pi x_n) \pmod{1}, \\
        x_{n+1} &= x_n + y_{n+1} \pmod{1}.
    \end{aligned}

Create the system and inspect its parameter order with the ``info`` property:

.. code-block:: python

    system = dds(model="standard map")
    print(system.info["parameters"])

.. code-block:: text

    ['k']

The state is ordered as ``[x, y]``, and the parameter list contains the stochasticity parameter ``k``. The methods of ``system`` can now generate trajectories and perform the analyses supported by the class.

Creating a custom map
~~~~~~~~~~~~~~~~~~~~~

A custom mapping function receives the current state and a one-dimensional parameter array, then returns the state at the next iteration. The following function reproduces the standard map:

.. code-block:: python

    import numpy as np
    from numba import njit
    from pynamicalsys import DiscreteDynamicalSystem as dds

    @njit
    def standard_map(state, parameters):
        x, y = state
        k = parameters[0]
        y_next = (y + k * np.sin(2.0 * np.pi * x) / (2.0 * np.pi)) % 1.0
        x_next = (x + y_next) % 1.0
        return np.array([x_next, y_next])

    @njit
    def standard_map_jacobian(state, parameters, mapping):
        x, _ = state
        k = parameters[0]
        derivative = k * np.cos(2.0 * np.pi * x)
        return np.array(
            [
                [1.0 + derivative, 1.0],
                [derivative, 1.0],
            ]
        )

The ``@njit`` decorator compiles the mapping for use by the package's Numba-accelerated numerical routines. Keep the function limited to operations supported by Numba.

When the parameter values are already known, pass them while creating the system:

.. code-block:: python

    system = dds(
        mapping=standard_map,
        jacobian=standard_map_jacobian,
        system_dimension=2,
        parameters=[1.5],
    )

Here, ``system_dimension=2`` corresponds to the two state variables ``[x, y]``. Supplying ``parameters=[1.5]`` also tells the class that the mapping expects one parameter and stores that value for later method calls.

If you want to provide the parameter value only when performing a calculation, declare the number of parameters instead:

.. code-block:: python

    system = dds(
        mapping=standard_map,
        jacobian=standard_map_jacobian,
        system_dimension=2,
        number_of_parameters=1,
    )

    next_state = system.step([0.1, 0.2], parameters=[1.5])

In this case, a method call must supply ``parameters`` until a value is stored with :py:meth:`set_parameters <pynamicalsys.core.discrete_dynamical_systems.DiscreteDynamicalSystem.set_parameters>`.

Managing parameters
~~~~~~~~~~~~~~~~~~~

Use ``set_parameters`` to store a parameter value for subsequent calculations:

.. code-block:: python

    system.set_parameters([1.5])
    print(system.get_parameters())

.. code-block:: text

    [1.5]

Methods use the stored parameters when their ``parameters`` argument is omitted. Passing the argument to an individual call temporarily overrides the stored value:

.. code-block:: python

    stored_result = system.step([0.1, 0.2])
    override_result = system.step([0.1, 0.2], parameters=[0.5])
    print(system.get_parameters())

.. code-block:: text

    [1.5]

The second call uses :math:`k=0.5`, but the value stored in ``system`` remains :math:`k=1.5`. Calling ``system.set_parameters([0.5])`` would replace the stored value for all later calls.

Jacobians and backward mappings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The mapping alone is enough to generate trajectories. If you do not provide a Jacobian, the class estimates one with finite differences when a tangent-space calculation needs it. Supplying an analytical Jacobian, as in the example above, can improve the accuracy and speed of Lyapunov exponents, stability calculations, and other tangent-space analyses.

The Jacobian must accept ``(state, parameters, mapping)`` because the numerical routines use the same calling convention for analytical Jacobians and the finite-difference fallback. An analytical Jacobian does not need to use ``mapping``, but the third argument must still be present. Decorate the function with ``@njit`` and pass it as ``jacobian=...`` when constructing the system.

A backward mapping is optional and is only required by calculations that iterate the system backward, such as stable invariant manifolds. It uses the same ``(state, parameters)`` interface as the forward mapping and should also be decorated with ``@njit``. Pass it as ``backwards_mapping=...`` when constructing a custom system.

The ``info`` property is available only for built-in models because the package cannot infer descriptive metadata or equations from a custom function.
