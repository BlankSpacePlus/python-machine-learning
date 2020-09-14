import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.gca()

x = np.random.rand(50)*10
y = np.random.rand(50)*10+20
s = np.random.rand(50)*100
c = np.random.rand(50)

data = {"a": x, "b": y, "color": c, "size": s}

# with the "data" keyboard argument
ax.scatter("a", "b", c="color", s="size", data=data)

ax.set(xlabel="X", ylabel="Y")

plt.show()
