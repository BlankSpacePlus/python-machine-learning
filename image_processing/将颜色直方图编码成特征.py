import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import cv2

image_bgr = cv2.imread("plane_256x256.jpg", cv2.IMREAD_COLOR)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

features = []
# 为每一个颜色通道计算直方图
colors = ('r', 'g', 'b')

# 为每一个通道计算直方图并把它加入特征值列表中
for i, channel in enumerate(colors):
    histogram = cv2.calcHist([image_rgb], [i], None, [256], [0, 256])
    features.extend(histogram)

# 将样本的特征值展开成一维数组
observation = np.array(features).flatten()

# 显示样本前5个特征值
print(observation[0:5])

# 显示RGB通道的值
print(image_rgb[0, 0])

data = pd.Series([1, 1, 2, 2, 3, 3, 3, 4, 5])

# 显示直方图
data.hist(grid=False)
plt.show()

# 为每个通道绘制直方图
for i, channel in enumerate(colors):
    histogram = cv2.calcHist([image_rgb], [i], None, [256], [0, 256])
    plt.plot(histogram, color=channel)
    plt.xlim([0, 256])

plt.show()
