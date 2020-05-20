import numpy as np
from matplotlib import pyplot as plt
import cv2

image_bgr = cv2.imread("plane_256x256.jpg")
image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

# 设置角点探测器的参数
corners_to_detect = 10
minimum_quality_score = 0.05
minimum_distance = 25

# 检测角点
corners = cv2.goodFeaturesToTrack(image_gray, corners_to_detect, minimum_quality_score, minimum_distance)
corners = np.float32(corners)

# 在每个角点上画白圈
for corner in corners:
    x, y = corner[0]
    cv2.circle(image_bgr, (x, y), 10, (255, 255, 255), -1)

# 转换成灰度图
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

plt.imshow(image_rgb, cmap="gray"), plt.axis("off")
plt.show()

