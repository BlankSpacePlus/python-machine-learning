import numpy as np
from sklearn import preprocessing

'''
将一个数值型特征的值缩放到两个特定的值之间
'''

# 创建特征
feature = np.array([[-500.5], [-100.1], [0], [100.1], [900.9]])

# 创建缩放器
min_max_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))

# 获取缩放特征的值
scale_feature = min_max_scale.fit_transform(feature)

# 查看特征值
print(scale_feature)
