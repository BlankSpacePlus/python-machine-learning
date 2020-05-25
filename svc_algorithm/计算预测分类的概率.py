from sklearn.svm import SVC
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np

# 加载数据，数据中只有两种分类和两个特征
iris = datasets.load_iris()
features = iris.data
target = iris.target

# 标准化特征
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

# 创建支持向量机分类器
svc = SVC(kernel="linear", probability=True, random_state=0)

# 训练模型
model = svc.fit(features_standardized, target)

# 创建一个新的观察值
new_observation = [[.4, .4, .4, .4]]

# 查看观察值被预测为不同分类的概率
print(model.predict_proba(new_observation))
