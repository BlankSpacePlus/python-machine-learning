import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("1.png", 0)

# 特征点在图片中的坐标位置
m = 448
n = 392

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

# setup the figure and axes
fig = plt.figure(figsize=(10, 5))  # 画布宽长比例
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')
ax1.set_title('Shaded')
ax2.set_title("colored")

# fake data
_x = np.arange(444, 453)
_y = np.arange(388, 397)
_xx, _yy = np.meshgrid(_x, _y)
x, y = _xx.ravel(), _yy.ravel()  # ravel扁平化
# 函数
top = []
for i in range(-4, 5):
    for j in range(-4, 5):
        top.append(img[i + n][j + m])

bottom = np.zeros_like(top)  # 每个柱的起始位置
width = depth = 1  # x,y方向的宽厚

ax1.bar3d(x, y, bottom, width, depth, top, shade=True)  # x，y为数组

ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('pixel value')

for i in range(-4, 5):
    for j in range(-4, 5):
        z = img[i + n][j + m]  # 该柱的高
        color = np.array([255, 255, z]) / 255.0  # 颜色 其中每个元素在0~1之间
        ax2.bar3d(j + m, i + n, 0, width, depth, z, color=color)  # 每次画一个柱

ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('pixel value')
plt.show()
