import numpy as np
from matplotlib import pyplot as plt
import cv2

# 以灰度图的格式导入图像
image = cv2.imread("plane.jpg", cv2.IMREAD_GRAYSCALE)

# 保存图像(有覆盖效果)
cv2.imwrite("plane_new.jpg", image)
