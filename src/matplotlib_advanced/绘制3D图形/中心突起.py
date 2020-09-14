from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

r = np.linspace(0, 1.25, 50)
p = np.linspace(0, 2*np.pi, 50)
r, p = np.meshgrid(r, p)
z = ((r ** 2 - 1) ** 2)

# Express the mesh in the cartesian system.
x, y = r * np.cos(p), r * np.sin(p)

# Plot the surface.
ax.plot_surface(x, y, z, cmap=plt.get_cmap('rainbow'))

# Tweak the limits and add latex math labels.
ax.set_zlim(0, 1)
ax.set_xlabel(r'$\phi_\mathrm{real}$')
ax.set_ylabel(r'$\phi_\mathrm{im}$')
ax.set_zlabel(r'$V(\phi)$')

plt.show()
