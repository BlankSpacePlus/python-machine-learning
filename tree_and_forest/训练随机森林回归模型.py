from sklearn.ensemble import RandomForestRegressor
from sklearn import datasets
from sklearn.datasets import load_boston

# 加载仅有两个特征的数据
boston = load_boston()
features = boston.data[:, 0:2]
target = boston.target

# 创建随机森林回归对象
random_forest = RandomForestRegressor(random_state=0, n_jobs=-1)

# 训练模型
model = random_forest.fit(features, target)
