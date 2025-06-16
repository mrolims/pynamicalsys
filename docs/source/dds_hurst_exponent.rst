Hurst exponent
~~~~~~~~~~~~~~

The Hurst exponent is a measure of the long-term memory of time series data. It can be used to determine whether a time series is trending, mean-reverting, or random. The Hurst exponent can take values in the range [0, 1]:

- If :math:`H < 0.5`, the time series is mean-reverting (anti-persistent).
- If :math:`H = 0.5`, the time series is a random walk (Brownian motion).
- If :math:`H > 0.5`, the time series is trending (persistent).

The Hurst exponent can be calculated using the :py:meth:`hurst_exponent <pynamicalsys.core.discrete_dynamical_systems.DiscreteDynamicalSystem.hurst_exponent>` method from the :py:class:`DiscreteDynamicalSystem <pynamicalsys.core.discrete_dynamical_systems.DiscreteDynamicalSystem>` class. The method takes the initial conditions and the number of iterations as input and returns the Hurst exponent for each time series.
The following code snippet shows how to calculate the Hurst exponent for the standard map with three different parameter values.

.. code-block:: python

   from pynamicalsys import DiscreteDynamicalSystem as dds
   import numpy as np

   # Create the discrete dynamical system object 
   ds = dds(model="standard map")

   # Create the random initial conditions
   num_ic = 250
   x_range = (0, 1)  # x range for initial conditions
   y_range = (0, 1)  # y range for initial conditions
   np.random.seed(0)  # Set the seed for reproducibility
   x_ic = np.random.uniform(x_range[0], x_range[1], num_ic)
   y_ic = np.random.uniform(y_range[0], y_range[1], num_ic)
   u = np.column_stack((x_ic, y_ic))  # Initial conditions array with shape (num_ic, d)

   # Parameter values for the standard map
   k_values = [0.9, 1.5, 3.9]

   # Calculate the Hurst exponent for each parameter value
   H = [ds.hurst_exponent(u[i], total_time, parameters=k[j]) for i in range(num_ic) for j in range(len(k))]
   H = np.array(H).reshape(num_ic, len(k), 2)

   # We also calculate the trajectories for each initial condition
   trajectories = [ds.trajectory(u, total_time, parameters=k[i]) for i in range(len(k))]
   trajectories_reshaped = []
   for trajectory in trajectories:
      trajectory_reshaped = trajectory.reshape(num_ic, total_time, 2)
      trajectories_reshaped.append(trajectory_reshaped)

We can visualize the Hurst exponent for the :math:`x` time series of the standard map with three different parameter values. The following code snippet shows how to create a plot of the Hurst exponent for each initial condition.

.. code-block:: python

   from pynamicalsys import PlotStyler
   import matplotlib.pyplot as plt

   # Set the plot style
   ps = PlotStyler(fontsize=24)
   ps.apply_style()

   # Create the figure and axes
   fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharey=True, sharex=True)
   [ps.set_tick_padding(ax[i], pad_x = 8) for i in range(3)]

   # Plot the Hurst exponent for each initial condition
   hms = [0, 0, 0]
   for j in range(len(k)):
      for i in range(num_ic):
         hm = ax[j].scatter(trajectories_reshaped[j][i, :, 0],
                              trajectories_reshaped[j][i, :, 1],
                              c=H[i, j, 0] * np.ones(total_time),
                              s=0.05,
                              edgecolor='none',
                              cmap="nipy_spectral",
                              vmin=0,
                              vmax=H[:, j, 0].max())
         hms[j] = hm

   # Add colorbars and labels
   [plt.colorbar(hms[i], ax=ax[i], label=rf"Hurst exponent with $k = {k[i]:.1f}$", location="top", aspect=40, pad=0.01) for i in range(len(k))]
   ax[0].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
   ax[0].set_xlim(0, 1)
   ax[0].set_ylim(0, 1)
   ax[0].set_ylabel("$y$")
   [ax[i].set_xlabel("$x$") for i in range(len(k))]

   plt.tight_layout(pad=0.05)
   plt.show()

.. figure:: images/standard_map_HE.png
   :align: center
   :width: 100%
   
   Hurst exponent of the :math:`x` time series for the standard map with three different parameter values.
