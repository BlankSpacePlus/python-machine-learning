import numpy as np
from matplotlib import pyplot as plt
import cv2

image_bgr = cv2.imread("plane_256x256.jpg", cv2.IMREAD_COLOR)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# 矩形的值，左上角的x坐标，左上角的y坐标，宽，高
rectangle = (0, 56, 256, 150)

# 创建初始掩模
mask = np.zeros(image_rgb.shape[:2], np.uint8)

# 创建grabCut函数所需要的临时数组
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

# 执行grabCut函数
cv2.grabCut(image_rgb, mask, rectangle, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

# 创建一个掩模，将确定或很可能是背景的部分设置为0，其余部分设置为1
mask_2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

# 将图像与掩模相乘除去背景
image_rgb_nobg = image_rgb * mask_2[:, :, np.newaxis]

# 显示图像
plt.imshow(image_rgb_nobg), plt.axis("off")
plt.show()

# 显示掩模
plt.imshow(mask, cmap='gray'), plt.axis("off")
plt.show()

plt.imshow(mask_2, cmap='gray'), plt.axis("off")
plt.show()
