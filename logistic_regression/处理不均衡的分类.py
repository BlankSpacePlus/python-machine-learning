import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

# 加载数据
iris = datasets.load_iris()
features = iris.data
target = iris.target

# 移除前40个观察值，使分类严重不均衡
features = features[40:, :]
target = target[40:]

# 创建目标向量，0代表分类为0，1代表除分类以外的其他分类
target = np.where((target == 0), 0, 1)

# 标准化特征
scaler = StandardScaler()
features_standard = scaler.fit_transform(features)

# 创建决策树分类器对象
logistic_regression = LogisticRegression(random_state=0, class_weight="balanced")

# 训练模型
model = logistic_regression.fit(features_standard, target)
