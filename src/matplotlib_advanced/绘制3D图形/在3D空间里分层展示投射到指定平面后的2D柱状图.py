import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection="3d")

colorList = ["#9900FF", "#FF6600", "#FF3366"]
yLayerList = [2, 1, 0]

for color, layer in zip(colorList, yLayerList):
    x = np.arange(10)
    y = np.random.rand(10)
    ax.bar(x, y, zs=layer, zdir="y", color=color, alpha=0.7)

ax.set(xlabel="X", ylabel="Y", zlabel="Z", yticks=yLayerList)

plt.show()
