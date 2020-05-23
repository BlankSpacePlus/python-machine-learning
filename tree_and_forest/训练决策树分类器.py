from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets

# 加载数据
iris = datasets.load_iris()
features = iris.data
target = iris.target

# 创建决策树分类器对象
decision_tree = DecisionTreeClassifier(random_state=0)

# 训练模型
model = decision_tree.fit(features, target)

# 创建新样本
observation = [[5, 4, 3, 2]]

# 预测样本的分类
print(model.predict(observation))

# 查看样本分别属于三个分类的频率
print(model.predict_proba(observation))

# 使用entry作为不纯度检测方法创建决策树分类器对象
decision_tree_entropy = DecisionTreeClassifier(criterion='entropy', random_state=0)

# 训练模型
model_entropy = decision_tree_entropy.fit(features, target)

print(model_entropy.predict(observation))
print(model_entropy.predict_proba(observation))
