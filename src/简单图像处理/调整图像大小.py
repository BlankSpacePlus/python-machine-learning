import numpy as np
from matplotlib import pyplot as plt
import cv2

# 以灰度图格式导入图像
image = cv2.imread("plane_256x256.jpg", cv2.IMREAD_GRAYSCALE)

# 将图片尺寸调整为50x50像素(机器学习常用的图像规格有：32x32、64x64、96x96、256x256)
image50x50 = cv2.resize(image, (50, 50))

# 查看图像
plt.imshow(image50x50, cmap="gray"), plt.axis("off")
plt.show()
