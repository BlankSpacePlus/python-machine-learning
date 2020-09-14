import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection="3d")

x = np.arange(-3, 3, 0.25)
y = np.arange(-3, 3, 0.25)
x, y = np.meshgrid(x, y)
r = np.sqrt(np.power(x, 2) + np.power(y, 2))
z = np.sin(r)

# plot 3d surface
surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)

# customize the z axis
ax.set(zlim=(-1, 1))
ax.zaxis.set_major_locator(LinearLocator(7))
ax.zaxis.set_major_formatter(FormatStrFormatter("%3.2f"))

# add a color bar mapping values to colors
fig.colorbar(surf, shrink=0.6, aspect=10)

plt.show()
