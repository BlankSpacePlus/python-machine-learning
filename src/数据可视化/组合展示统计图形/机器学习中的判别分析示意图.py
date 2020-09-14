import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()

num = 50

# new sample
sample = 10*np.random.rand(num, 2)
var1 = sample[:, 0]
var2 = sample[:, 1]

# threshold value
td = 12

# discriminant function
df = 2*var1+var2

cates11 = np.ma.masked_where(df >= td, var1)
cates12 = np.ma.masked_where(df >= td, var2)

cates21 = np.ma.masked_where(df <= td, var1)
cates22 = np.ma.masked_where(df <= td, var2)

ax.scatter(var1, var2, s=cates11*50, marker="s", c=cates11)
ax.scatter(var1, var2, s=cates21*50, marker="o", c=cates21)

ax.plot(var1, -2*var1+12, lw=1, color="r", alpha=0.65)

ax.axis([-1, 11, -1, 11])

plt.show()
