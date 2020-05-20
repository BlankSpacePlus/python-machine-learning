import numpy as np
from matplotlib import pyplot as plt
import cv2

image_bgr = cv2.imread("plane_256x256.jpg")
image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
image_gray = np.float32(image_gray)

# 设置角点探测器的参数
block_size = 2
aperture = 29
free_parameter = 0.04

# 检测角点
detector_responses = cv2.cornerHarris(image_gray, block_size, aperture, free_parameter)

# 设置角点检测器参数
detector_responses = cv2.dilate(detector_responses, None)

# 只保留大于阈值的检测结果，并把它们标记成白色
threshold = 0.02
image_bgr[detector_responses > threshold * detector_responses.max()] = [255, 255, 255]

# 转换成灰度图
image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

plt.imshow(image_gray, cmap="gray"), plt.axis("off")
plt.show()

# 画一张显示可能的角点的灰度图
plt.imshow(detector_responses, cmap="gray"), plt.axis("off")
plt.show()
