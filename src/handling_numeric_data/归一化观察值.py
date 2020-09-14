import numpy as np
from sklearn.preprocessing import Normalizer

'''
对观察值的每一个特征进行缩放，使其拥有一致的范数（总长度是1）
'''

# 创建特征矩阵
features = np.array([[0.5, 0.5], [1.1, 3.4], [1.5, 20.2], [1.63, 34.4], [10.9, 3.3]])

# 创建归一化器
normalizer = Normalizer(norm="l2")

# 转换特征矩阵
print(normalizer.transform(features))

# 转换特征矩阵（L2范数）
features_l2_norm = Normalizer(norm="l2").transform(features)

# 查看特征矩阵
print(features_l2_norm)

# 转换特征矩阵（L1范数）
features_l1_norm = Normalizer(norm="l1").transform(features)
# 查看特征矩阵
print(features_l1_norm)
'''
使用L1进行缩放之后，它的元素总和为1
'''
print("Sum of the first observation\'s values", features_l1_norm[0, 0] + features_l1_norm[0, 1])
