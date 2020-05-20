import numpy as np
from matplotlib import pyplot as plt
import cv2

image = cv2.imread("plane_256x256.jpg", cv2.IMREAD_GRAYSCALE)

# 创建核
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

# 锐化图像
image_shape = cv2.filter2D(image, -1, kernel)

plt.imshow(image_shape, cmap="gray"), plt.axis("off")
plt.show()
