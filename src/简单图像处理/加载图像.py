import numpy as np
from matplotlib import pyplot as plt
import cv2

# 把图像导入为灰度图
image = cv2.imread("plane.jpg", cv2.IMREAD_GRAYSCALE)

# 使用matplotlib显示图像
plt.imshow(image, cmap="gray"), plt.axis("off")
plt.show()

# 查看图像数据类型
print(type(image))

# 查看图像数据
print(image)

# 显示图像矩阵维度
print(image.shape)

# 显示第一个像素点的像素值
print(image[0, 0])

# 以彩色模式加载图像
image_bgr = cv2.imread("plane.jpg", cv2.IMREAD_COLOR)
# 显示像素值
print(image_bgr[0, 0])

# 转换为RGB格式
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
# 显示图像
plt.imshow(image_rgb), plt.axis("off")
plt.show()
