import numpy as np
from numba import njit
from joblib import Parallel, delayed
from pynamicalsys import DiscreteDynamicalSystem as dds


def main():
    ds = dds(
        mapping=standard_map,
        jacobian=standard_map_jacobian,
        system_dimension=2,
        number_of_parameters=1,
    )
    _ = ds.SALI([1.2, 2], 1000, parameters=4)

    grid_size = 5000
    x_range = (-1.5, 1.5, grid_size)
    y_range = (-2.5, 2.5, grid_size)
    x = np.linspace(*x_range)
    y = np.linspace(*y_range)
    X, Y = np.meshgrid(x, y, indexing="ij")
    total_time = 1000

    sali = Parallel(n_jobs=-1)(
        delayed(ds.SALI)(np.array([X[i, j], Y[i, j]]), total_time, parameters=4)
        for i in range(grid_size)
        for j in range(grid_size)
    )
    sali = np.array(sali).reshape((grid_size, grid_size))

    data = np.zeros((grid_size**2, 3))
    data[:, 0] = X.flatten()
    data[:, 1] = Y.flatten()
    data[:, 2] = sali.flatten()
    datafile = "sali_standard_map.dat"
    np.savetxt(datafile, data, fmt="%.16f", delimiter=" ")


@njit
def standard_map(u, parameters):
    k = parameters[0]
    x, y = u

    y_new = y - k * np.sin(x)
    x_new = x + y_new

    x_new = (x_new + np.pi) % (2 * np.pi) - np.pi
    y_new = (y_new + np.pi) % (2 * np.pi) - np.pi

    return np.array([x_new, y_new])


@njit
def standard_map_jacobian(u, parameters, *args):
    k = parameters[0]
    x, y = u

    return np.array([[1 - k * np.cos(x), 1], [-k * np.cos(x), 1]])


if __name__ == "__main__":
    main()
