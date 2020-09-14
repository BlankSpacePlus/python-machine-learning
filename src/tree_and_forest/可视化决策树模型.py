import pydotplus
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from IPython.display import Image
from sklearn import tree

# 加载数据
iris = datasets.load_iris()
features = iris.data
target = iris.target

# 创建决策树分类器对象
decision_tree = DecisionTreeClassifier(random_state=0)

# 训练模型
model = decision_tree.fit(features, target)

# 创建DOT数据
dot_data = tree.export_graphviz(decision_tree, out_file=None, feature_names=iris.feature_names,
                                class_names=iris.target_names)

# 绘制图形
graph = pydotplus.graph_from_dot_data(dot_data)

# 显示图形
Image(graph.create_png())

# 创建PDF
graph.write_pdf("iris.pdf")

# 创建PNG
graph.write_png("iris.png")
