import numpy as np
from matplotlib import pyplot as plt
import cv2

image = cv2.imread("plane_256x256.jpg", cv2.IMREAD_GRAYSCALE)

# 增强图像
image_enhanced = cv2.equalizeHist(image)

plt.imshow(image_enhanced, cmap="gray"), plt.axis("off")
plt.show()

# 加载图像
image_bgr = cv2.imread("plane.jpg")

# 转换成YUV格式(Y代表亮度，U和V代表颜色)
image_yuv = cv2.cvtColor(image_bgr, cv2.COLOR_YUV2BGR)

plt.imshow(image_yuv), plt.axis("off")
plt.show()
