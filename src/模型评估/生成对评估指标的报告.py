from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 加载数据
iris = datasets.load_iris()
features, target = iris.data, iris.target

# 创建目标分类名的列表
class_names = iris.target_names

# 创建训练集和测试集
features_train, features_test, target_train, target_test = train_test_split(features, target, random_state=1)

# 创建逻辑回归对象
classifier = LogisticRegression(max_iter=3000)

# 训练模型并作出预测
model = classifier.fit(features_train, target_train)
target_predicted = model.predict(features_test)

# 生成分类器的性能报告
print(classification_report(target_test, target_predicted, target_names=class_names))
