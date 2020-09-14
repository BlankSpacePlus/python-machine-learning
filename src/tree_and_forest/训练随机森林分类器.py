from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets

# 加载数据
iris = datasets.load_iris()
features = iris.data
target = iris.target

# 创建随机森林分类器对象
random_forest = RandomForestClassifier(random_state=0, n_jobs=-1)

# 训练模型
model = random_forest.fit(features, target)

# 创建新样本
observation = [[5, 4, 3, 2]]

# 预测样本的分类
print(model.predict(observation))

# 使用熵创建随机森林分类器对象
random_forest_entropy = RandomForestClassifier(criterion="entropy", random_state=0)

# 训练模型
model_entropy = random_forest_entropy.fit(features, target)

print(model_entropy.predict(observation))
