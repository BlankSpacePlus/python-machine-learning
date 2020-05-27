from sklearn.datasets import load_boston
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 加载数据
boston = load_boston()
features, target = boston.data, boston.target

# 将数据分为测试集和训练集
features_train, features_test, target_train, target_test = train_test_split(features, target, random_state=0)

# 创建DummyRegressor对象
dummy = DummyRegressor(strategy='mean')

# 训练回归模型
dummy.fit(features_train, target_train)

# 计算R方得分
print(dummy.score(features_test, target_test))

ols = LinearRegression()
ols.fit(features_train, target_train)

# 计算R方得分
print(ols.score(features_test, target_test))

# 创建一个讲所有样本预测为20的DummyRegressor
clf = DummyRegressor(strategy='constant', constant=20)
clf.fit(features_train, target_train)

# 计算模型的得分
print(clf.score(features_test, target_test))
