from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets

# 加载数据
iris = datasets.load_iris()
features = iris.data
target = iris.target

# 创建决策树分类器对象
decision_tree = DecisionTreeClassifier(random_state=0, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                                       min_weight_fraction_leaf=0, max_leaf_nodes=None, min_impurity_decrease=0)

# 训练模型
model = decision_tree.fit(features, target)
