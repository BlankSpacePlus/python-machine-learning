from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets

# 加载数据
iris = datasets.load_iris()
features = iris.data
target = iris.target

# 创建adaboost树分类器对象
adaboost = AdaBoostClassifier(random_state=0)

# 训练模型
model = adaboost.fit(features, target)
