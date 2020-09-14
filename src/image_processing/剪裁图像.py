import numpy as np
from matplotlib import pyplot as plt
import cv2

# 以灰度图格式导入图像
image = cv2.imread("plane_256x256.jpg", cv2.IMREAD_GRAYSCALE)

# 选择所有的行和前128列
image_cropped = image[:, :128]

# 显示图像
plt.imshow(image_cropped, cmap="gray"), plt.axis("off")
plt.show()
