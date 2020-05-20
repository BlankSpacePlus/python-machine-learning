import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.gca(projection='3d')
x, y = np.mgrid[0:6 * np.pi:0.25, 0:4 * np.pi:0.25]
z = np.sqrt(np.abs(np.cos(x) + np.cos(y)))

surf = ax.plot_surface(x, y, z, cmap='autumn', cstride=2, rstride=2)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_zlim(0, 2)

plt.show()
