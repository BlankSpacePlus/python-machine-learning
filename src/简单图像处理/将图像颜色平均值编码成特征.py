import numpy as np
from matplotlib import pyplot as plt
import cv2

# 以BGR格式加载图像
image_bgr = cv2.imread("plane_256x256.jpg", cv2.IMREAD_COLOR)

# 计算每个通道的平均值
channels = cv2.mean(image_bgr)

# 交换红色通道和蓝色通道，将图像从BGR格式转换成RGB格式
observation = np.array([(channels[2], channels[1], channels[0])])

# 显示每个颜色通道的平均值
print(observation)

# 显示颜色图像
plt.imshow(observation), plt.axis("off")
plt.show()
