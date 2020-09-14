import numpy as np
from sklearn.preprocessing import Binarizer

'''
将数值型特征离散化，分到多个离散的小区间中
根据数据离散化的方式，有两种方法可以使用：
1.根据阈值将特征二值化
2.根据多个阈值将数值型特征离散化
'''

# 创建特征
age = np.array([[6], [12], [20], [36], [65]])

# 创建二值化器
binarizer = Binarizer(threshold=18)
# 转换特征
print(binarizer.fit_transform(age))

# 根据多个阈值将数值型特征离散化(bins参数是左闭右开的)
print(np.digitize(age, bins=[20, 30, 64]))

# 设置right=True
print(np.digitize(age, bins=[20, 30, 64], right=True))

# 仅指定一个阈值也行
print(np.digitize(age, bins=[18]))
