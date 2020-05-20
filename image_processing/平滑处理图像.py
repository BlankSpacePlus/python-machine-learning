import numpy as np
from matplotlib import pyplot as plt
import cv2

# 以灰度图格式导入图像
image = cv2.imread("plane_256x256.jpg", cv2.IMREAD_GRAYSCALE)

# 平滑处理图像
image_blurry = cv2.blur(image, (5, 5))
plt.imshow(image_blurry, cmap="gray"), plt.axis("off")
plt.show()

image_very_blurry = cv2.blur(image, (100, 100))
plt.imshow(image_very_blurry, cmap="gray"), plt.xticks([]), plt.yticks([])
plt.show()

# 创建核(我们使用的平滑核)
kernel = np.ones((5, 5)) / 25.0
print(kernel)

# 应用核
image_kernel = cv2.filter2D(image, -1, kernel)
plt.imshow(image_kernel, cmap="gray"), plt.xticks([]), plt.yticks([])
plt.show()
