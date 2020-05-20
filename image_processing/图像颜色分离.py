import numpy as np
from matplotlib import pyplot as plt
import cv2

image_bgr = cv2.imread("plane_256x256.jpg")

# 将BGR格式转为HSV格式
image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

# 定义HSV格式中蓝色分量的区间
lower_blue = np.array([50, 100, 50])
upper_blue = np.array([130, 255, 255])

# 创建掩模
mask = cv2.inRange(image_hsv, lower_blue, upper_blue)

# 应用掩模
image_bgr_masked = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)

# 从BGR格式转为RGB格式
image_rgb = cv2.cvtColor(image_bgr_masked, cv2.COLOR_BGR2RGB)

plt.imshow(image_rgb), plt.axis("off")
plt.show()

plt.imshow(mask, cmap='gray'), plt.axis("off")
plt.show()
