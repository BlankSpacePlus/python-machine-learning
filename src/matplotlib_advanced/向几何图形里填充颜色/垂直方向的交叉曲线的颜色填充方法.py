import matplotlib.pyplot as plt
import numpy as np

y = np.linspace(0, 2, 500)
x1 = np.sin(2*np.pi*y)
x2 = 1.1*np.sin(3*np.pi*y)

fig = plt.figure()

ax = fig.add_subplot(111)

# plot x1 and plot x2
ax.plot(x1, y, color="k", lw=1, ls="-")
ax.plot(x1, y, color="k", lw=1, ls="-")

# "where x1 <= x2"
ax.fill_betweenx(y, x1, x2, where=x2 >= x1, facecolor="darkred", alpha=0.7)

# "where x1 >= x2"
ax.fill_betweenx(y, x1, x2, where=x2 <= x1, facecolor="orange", alpha=0.7)

ax.set_xlim(-1.2, 1.2)
ax.set_ylim(0, 2)

ax.grid(ls=":", lw=1, color="gray", alpha=0.5)

plt.show()

