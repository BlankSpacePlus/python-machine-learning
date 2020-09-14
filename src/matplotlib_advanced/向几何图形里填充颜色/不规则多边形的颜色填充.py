import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 2*np.pi, 500)
y = np.sin(x)

plt.fill(x, y, color="cornflowerblue", alpha=0.4)
plt.fill(x, y, color="cornflowerblue", alpha=0.8)
plt.fill([x[0], x[-1]], [y[0], y[-1]], color="cornflowerblue", alpha=0.8)

plt.xlim(0, 2*np.pi)
plt.ylim(-1.1, 1.1)

plt.show()
