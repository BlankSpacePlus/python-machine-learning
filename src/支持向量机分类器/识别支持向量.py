from sklearn.svm import SVC
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np

# 加载数据，数据中只有两种分类和两个特征
iris = datasets.load_iris()
features = iris.data[:100, :]
target = iris.target[:100]

# 标准化特征
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

# 创建支持向量机分类器
svc = SVC(kernel="linear", random_state=0)

# 训练模型
model = svc.fit(features_standardized, target)

# 查看支持向量
print(model.support_vectors_)
