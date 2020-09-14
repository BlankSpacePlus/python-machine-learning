import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets

# 加载数据
iris = datasets.load_iris()
features = iris.data
target = iris.target

# 删除前40个样本以获得高度不均衡的数据
features = features[40:, :]
target = target[40:]

# 创建目标向量表明分类是0还是1
target = np.where((target == 0), 0, 1)

# 创建随机森林分类器对象
random_forest = RandomForestClassifier(random_state=0, n_jobs=-1, class_weight="balanced")

# 训练模型
model = random_forest.fit(features, target)
