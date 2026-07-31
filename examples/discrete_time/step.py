import numpy as np

from pynamicalsys import DiscreteDynamicalSystem as dds

ds = dds(model="standard map")

x = 0.2
y = 0.5
u = [x, y]
trajectory = ds.trajectory(u, 1, parameters=1.5)
print(trajectory)
print()

x = 0.2
y = 0.3
u = [x, y]
trajectory = ds.trajectory(u, 1, parameters=1.5)
print(trajectory)
print()

x = 0.2
y = 0.6
u = [x, y]
trajectory = ds.trajectory(u, 1, parameters=1.5)
print(trajectory)
print()

u = np.array([[0.2, 0.5], [0.2, 0.3], [0.2, 0.6]])
print(u.shape)
u_next = ds.step(u, parameters=1.5)
print(u_next)
