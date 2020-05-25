from sklearn.svm import SVC
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np

# 加载数据，数据中只有两种分类和两个特征
iris = datasets.load_iris()
features = iris.data[:100, :]
target = iris.target[:100]

# 删除前40个观察值，让各个分类的数据分布不均衡
features = features[40:, :]
target = target[40:]

# 创建目标向量，数值0代表分类0，1代表分类1
target = np.where((target == 0), 0, 1)

# 标准化特征
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

# 创建支持向量机分类器
svc = SVC(kernel="linear", class_weight="balanced", C=1.0, random_state=0)

# 训练模型
model = svc.fit(features_standardized, target)