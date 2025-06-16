Weighted Birkhoff average
~~~~~~~~~~~~~~~~~~~~~~~~~

The weighted Birkhoff average is a powerful tool to classify the dynamics as regular or chaotic in a discrete dynamical system :math:`\mathbf{x}_{n + 1} = \mathbf{M}(\mathbf{x}_n) = \mathbf{M}^n(\mathbf{x}_0)`. It is defined as a weighted average of a function :math:`f` over the time evolution of the system. The weights are given by an exponential bump function :math:`g`. The weighted Birkhoff average is defined as:

.. math::

   \begin{equation}
        W\!B_N(f)({\bf x}_0) = \sum_{n=0}^{N-1} w_{n,N} f \circ {\bf M}^n({\bf x}_0),
    \end{equation}

where 

.. math::

   \begin{equation}   
      w_{n,N} = \frac{g(n/N)}{\sum_{n=0}^{N-1} g(n/N)},
   \end{equation}

with the exponential bump function defined as:

.. math::

   \begin{equation}
        g(z) = \begin{cases}
              \exp\{-{\lbrack z(1-z)\rbrack}^{-1} \} & \text{if $0 < z < 1$} \\
              0 & \text{otherwise}
           \end{cases}
    \end{equation}

The convergence of the weighted Birkhoff average can be measured by calculating the number of zeros after the decimal point in the difference between :math:`W\!B_N(f)({\bf x}_0)` and :math:`W\!B_{N}(f)({\bf x}_{N})`. The larget the number of zeros, the faster the convergence. The number of zeros is defined as:

.. math::

   \begin{equation}
        {\mathrm{dig}} = - \log_{10} \left\vert W\!B_N(f)({\bf x}_0) - W\!B_N(f)({\bf x}_{N}) \right\vert.
    \end{equation}

The weighted Birkhoff average, in contrast to the standard Birkhoff average, only improves the convergence of the average for regular orbits. Therefore, a high value of dig, which indicates a fast convergence, is a strong indicator for regular dynamics. A low value of dig, which indicates a slow convergence, is a strong indicator for chaotic dynamics.

The following code snippet shows how to calculate the number of digits of the weighted Birkhoff average convergence using the :py:meth:`dig <pynamicalsys.core.discrete_dynamical_systems.DiscreteDynamicalSystem.dig>` method from the :py:class:`DiscreteDynamicalSystem <pynamicalsys.core.discrete_dynamical_systems.DiscreteDynamicalSystem>` class for the standard map with three different functions.

.. code-block:: python

   from pynamicalsys import DiscreteDynamicalSystem as dds
   import numpy as np

   # Create the discrete dynamical system object 
   ds = dds(model="standard map")

   # Create the random initial conditions
   num_ic = 250
   x_range = (0, 1)  # x range for initial conditions
   y_range = (0, 1)  # y range for initial conditions
   np.random.seed(0)  # Set the seed for reproducibility
   x_ic = np.random.uniform(x_range[0], x_range[1], num_ic)
   y_ic = np.random.uniform(y_range[0], y_range[1], num_ic)
   u = np.column_stack((x_ic, y_ic))  # Initial conditions array with shape (num_ic, d)

   # Parameter for the standard map
   k = 1.5

   # Total iteration time
   total_time = 10000

   # Calculate dig using the default function: f(x) = cos(2 * pi * x)
   dig1 = [ds.dig(u[i], total_time, parameters=k) for i in range(num_ic)]
   dig1 = np.array(dig1)

   # Calculate dig using a custom function: f(x) = sin(2 * pi * x)
   dig2 = [ds.dig(u[i], total_time, parameters=k, func=lambda x: np.sin(2 * np.pi * x[:, 0])) for i in range(num_ic)]
   dig2 = np.array(dig2)

   # Calculate dig using another custom function: f(x, y) = sin(2 * pi * (x + y))
   dig3 = [ds.dig(u[i], total_time, parameters=k, func=lambda x: np.sin(2 * np.pi * (x[:, 0] + x[:, 1]))) for i in range(num_ic)]
   dig3 = np.array(dig3)

   # Also calculate the trajectories for each initial condition
   trajectories = ds.trajectory(u, k, total_time)
   trajectories_reshaped = trajectories.reshape(num_ic, total_time, 2)

We can visualize the results by plotting the number of digits for each initial condition. The following code snippet shows how to create a scatter plot of the number of digits for each initial condition.

.. code-block:: python

   from pynamicalsys import PlotStyler
   import matplotlib.pyplot as plt

   # Set the plot style
   ps = PlotStyler(fontsize=24)
   ps.apply_style()
   
   # Create the figure and axes
   fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharey=True, sharex=True)

   # Set the x padding for the axes 
   [ps.set_tick_padding(ax[i], pad_x = 8) for i in range(3)]

   # Plot the number of digits for each initial condition
   for i in range(num_ic):
      hm1 = ax[0].scatter(trajectories_reshaped[i, :, 0], trajectories_reshaped[i, :, 1], c=dig1[i] * np.ones(total_time), s=0.05, edgecolor='none', cmap="nipy_spectral", vmin=0, vmax=dig1[dig1 != np.inf].max())
      hm2 = ax[1].scatter(trajectories_reshaped[i, :, 0], trajectories_reshaped[i, :, 1], c=dig2[i] * np.ones(total_time), s=0.05, edgecolor='none', cmap="nipy_spectral", vmin=0, vmax=dig2.max())
      hm3 = ax[2].scatter(trajectories_reshaped[i, :, 0], trajectories_reshaped[i, :, 1], c=dig3[i] * np.ones(total_time), s=0.05, edgecolor='none', cmap="nipy_spectral", vmin=0, vmax=dig3.max())

   # Create the colorbars and set the labels and limits
   plt.colorbar(hm1, ax=ax[0], label=r"dig with $f(x) = \cos(2\pi x)$", location="top", aspect=40, pad=0.01)
   plt.colorbar(hm2, ax=ax[1], label=r"dig with $f(x) = \sin(2\pi x)$", location="top", aspect=40, pad=0.01)
   plt.colorbar(hm3, ax=ax[2], label=r"dig with $f(x, y) = \sin(2\pi(x + y))$", location="top", aspect=40, pad=0.01)
   ax[0].set_xlim(0, 1)
   ax[0].set_ylim(0, 1)
   ax[0].set_xlabel("$x$")
   ax[0].set_ylabel("$y$")
   ax[1].set_xlabel("$x$")
   ax[2].set_xlabel("$x$")

   plt.tight_layout(pad=0.05)
   plt.show()

.. figure:: images/standard_map_dig.png
   :align: center
   :width: 100%
   
   dig for the standard map using three different functions.
