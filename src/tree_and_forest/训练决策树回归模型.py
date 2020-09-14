from sklearn.tree import DecisionTreeRegressor
from sklearn import datasets
from sklearn.datasets import load_boston

# 加载仅有两个特征的数据
boston = load_boston()
features = boston.data[:, 0:2]
target = boston.target

# 创建决策树回归模型对象
decision_tree = DecisionTreeRegressor(random_state=0)

# 训练模型
model = decision_tree.fit(features, target)

# 创建新样本
observation = [[0.02, 16]]

# 预测样本值
print(model.predict(observation))

# 使用MAE创建决策树回归模型
decision_tree_mae = DecisionTreeRegressor(criterion="mae", random_state=0)

# 训练模型
model_mae = decision_tree_mae.fit(features, target)

print(model_mae.predict(observation))
