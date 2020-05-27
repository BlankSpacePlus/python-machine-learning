from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegressionCV

# 加载数据
iris = load_iris()
features = iris.data
target = iris.target

# 创建LogisticRegressionCV对象
logistic = LogisticRegressionCV(Cs=100, max_iter=10000)

# 训练模型
logistic.fit(features, target)
