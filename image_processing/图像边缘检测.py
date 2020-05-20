import numpy as np
from matplotlib import pyplot as plt
import cv2

# 以灰度格式加载图像
image_gray = cv2.imread("plane_256x256.jpg", cv2.IMREAD_GRAYSCALE)

# 计算像素强度的中位数
median_intensity = np.median(image_gray)

# 设置阈值
lower_threshold = int(max(0, (1.0 - 0.33) * median_intensity))
upper_threshold = int(min(255, (1.0 + 0.33) * median_intensity))

# 应用Canny边缘检测器
image_canny = cv2.Canny(image_gray, lower_threshold, upper_threshold)

# 显示图像
plt.imshow(image_canny, cmap="gray"), plt.axis("off")
plt.show()
