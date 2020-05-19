import numpy as np
from fancyimpute import KNN
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.datasets import make_blobs

# 创建模拟特征矩阵
features, _ = make_blobs(n_samples=100, n_features=2, random_state=1)

# 标准化特征
scaler = StandardScaler()
standard_features = scaler.fit_transform(features)

# 将第一个特征向量的第一个值替换为缺失值
true_value = standard_features[0, 0]
standard_features[0, 0] = np.nan

# 预测特征矩阵中的缺失值
features_knn_imputed = KNN(k=5, verbose=0).complete(standard_features)

# 对比真实值和填充值
print("True Value: ", true_value)
print("Imputed Value：", features_knn_imputed[0, 0])



