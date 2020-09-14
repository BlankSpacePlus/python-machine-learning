from sklearn.datasets import load_iris
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 加载数据
iris = load_iris()
features, target = iris.data, iris.target

# 将数据分为测试集和训练集
features_train, features_test, target_train, target_test = train_test_split(features, target, random_state=0)

# 创建DummyClassifier
dummy = DummyClassifier(strategy='uniform', random_state=1)

# 训练模型
dummy.fit(features_train, target_train)

# 计算模型的得分
print(dummy.score(features_test, target_test))

# 创建分类模型
classifier = RandomForestClassifier()

# 训练模型
classifier.fit(features_train, target_train)

# 计算模型得分
print(classifier.score(features_test, target_test))
