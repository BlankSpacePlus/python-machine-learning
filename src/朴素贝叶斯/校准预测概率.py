from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV

# 加载数据
iris = datasets.load_iris()
features = iris.data
target = iris.target

# 创建高斯朴素贝叶斯对象
classifier = GaussianNB()

# 创建使用sigmoid校准调校过的交叉验证模型
classifier_sigmoid = CalibratedClassifierCV(classifier, cv=2, method='sigmoid')

# 校准概率
classifier_sigmoid.fit(features, target)

# 创建新的观察值
new_observation = [[2.6, 2.6, 2.6, 0.4]]

# 查看校准过的概率
print(classifier_sigmoid.predict_proba(new_observation))

# 训练一个高斯朴素贝叶斯分类器来预测观察值的分类概率 与上面的结果很不同
print(classifier.fit(features, target).predict_proba(new_observation))
