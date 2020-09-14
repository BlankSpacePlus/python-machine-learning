import pandas as pd
import numpy as np

# 创建一个特征矩阵，其中包含两个高度相关的特征
features = np.array([[1, 1, 1], [2, 2, 0], [3, 3, 1], [4, 4, 0], [5, 5, 1], [6, 6, 0], [7, 7, 1], [8, 7, 0], [9, 7, 1]])

# 将特征矩阵转换成DataFrame
data_frame = pd.DataFrame(features)

# 创建相关矩阵
corr_matrix = data_frame.corr().abs()

# 选择相关矩阵的上三角阵
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# 找到相关性大于0.95的特征列的索引
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

# 删除特征
print(data_frame.drop(data_frame.columns[to_drop], axis=1).head(3))

# 相关矩阵
print(data_frame.corr())

# 相关矩阵的上三角矩阵
print(upper)
