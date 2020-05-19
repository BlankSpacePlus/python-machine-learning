import numpy as np
from fancyimpute import KNN
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
from sklearn.impute import SimpleImputer

# 创建模拟特征矩阵
features, _ = make_blobs(n_samples=100, n_features=2, random_state=1)

# 标准化特征
scaler = StandardScaler()
standard_features = scaler.fit_transform(features)

# 将第一个特征向量的第一个值替换为缺失值
true_value = standard_features[0, 0]
standard_features[0, 0] = np.nan

# 预测特征矩阵中的缺失值
features_knn_imputed = KNN(k=5, verbose=False).fit_transform(standard_features)

# 对比真实值和填充值
print("True Value: ", true_value)
print("Imputed Value：", features_knn_imputed[0, 0])

# 创建填充器
mean_imputer = SimpleImputer(strategy="mean")

# 填充缺失值
features_mean_imputed = mean_imputer.fit_transform(features)

# 对比真实值和填充值
print("True Value: ", true_value)
print("Imputed Value：", features_mean_imputed[0, 0])
