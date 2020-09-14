import numpy as np
import pandas as pd

# 创建特征矩阵
features = np.array([[1.1, 11.1], [2.2, 22.2], [3.3, 33.3], [4.4, 44.4]])

# 只保留没有（用 ~ 来表示）缺失值的观察值，一行Numpy代码就能实现功能
print(features[~np.isnan(features).any(axis=1)])

# 使用Pandas丢弃含有缺失值的观察值
data_frame = pd.DataFrame(features, columns=["feature1", "feature2"])

# 删除带有缺失值的观察值
print(data_frame.dropna())
