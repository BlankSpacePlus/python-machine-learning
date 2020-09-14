import numpy as np
from matplotlib import pyplot as plt
import cv2

image = cv2.imread("plane_256x256.jpg", cv2.IMREAD_GRAYSCALE)

# 将图像尺寸转换成10x10
image10x10 = cv2.resize(image, (10, 10))

# 将图像数据转换成一维向量
print(image10x10.flatten())

plt.imshow(image10x10, cmap="gray"), plt.axis("off")
plt.show()

print(image10x10.shape)
print(image10x10.flatten().shape)

# 以彩色模式加载图像
image_color = cv2.imread("plane_256x256.jpg", cv2.IMREAD_COLOR)
image_color_10x10 = cv2.resize(image_color, (10, 10))

# 将该图像数据转换成一维数组并显示数组维度
print(image_color_10x10.flatten().shape)

image_256x256_gray = cv2.imread("plane_256x256.jpg", cv2.IMREAD_GRAYSCALE)
print(image_256x256_gray.flatten().shape)

image_256x256_color = cv2.imread("plane_256x256.jpg", cv2.IMREAD_COLOR)
print(image_256x256_color.flatten().shape)
