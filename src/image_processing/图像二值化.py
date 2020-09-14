import numpy as np
from matplotlib import pyplot as plt
import cv2

image_gray = cv2.imread("plane_256x256.jpg", cv2.IMREAD_GRAYSCALE)

# 应用自适应阈值处理(阈值处理的主要优点之一是图像去噪)
max_output_value = 255
neighborhood_size = 99
subtract_from_mean = 10
image_binarized = cv2.adaptiveThreshold(image_gray, max_output_value, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                        neighborhood_size, subtract_from_mean)

plt.imshow(image_binarized, cmap="gray"), plt.axis("off")
plt.show()

# 使用 cv2.ADAPTIVE_THRESH_MEAN_C
image_mean_threhold = cv2.adaptiveThreshold(image_gray, max_output_value, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
                                            neighborhood_size, subtract_from_mean)
plt.imshow(image_mean_threhold, cmap="gray"), plt.axis("off")
plt.show()
