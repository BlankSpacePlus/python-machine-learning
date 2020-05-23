from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
import sklearn
# 作用是让pickle文件适用于Numpy数组很大的情况
import joblib

# 加载数据
iris = datasets.load_iris()
features = iris.data
target = iris.target

# 创建决策树分类器对象
classifier = RandomForestClassifier()

# 训练模型
model = classifier.fit(features, target)

# 把模型保存为pickle文件
# joblib.dump(model, "model.pkl")
# 保存的模型可能在各个版本中的scikit-learn中不兼容
scikit_version = joblib.__version__
joblib.dump(model, "model_{version}.pkl".format(version=scikit_version))

# 从文件中加载模型
classifier = joblib.load("model_{version}.pkl".format(version=scikit_version))

# 创建新的样本
new_observation = [[5.2, 3.2, 1.1, 0.1]]

# 预测样本的分类
print(classifier.predict(new_observation))
