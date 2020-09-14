from sklearn.preprocessing import StandardScaler
import numpy as np

# 创建特征
features = np.array([[-100.1, 3240.1], [-200.2, -234.1], [5000.5, 150.1], [6000.6, -125.1], [9000.9, -673.1]])

# 创建scaler
scaler = StandardScaler()

# 转换特征
features_standardized = scaler.fit_transform(features)

# 展示特征
print(features_standardized)

# 打印均值和标准差
print("Mean: ", round(features_standardized[:, 0].mean()))
print("Standard deviation: ", round(features_standardized[:, 0].std()))
