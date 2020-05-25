from sklearn import datasets
from sklearn.naive_bayes import GaussianNB

# 加载数据
iris = datasets.load_iris()
features = iris.data
target = iris.target

# 创建高斯朴素贝叶斯对象
classifier = GaussianNB()

# 训练模型
model = classifier.fit(features, target)

# 创建一个观察值
new_observation = [[4, 4, 4, 0.4]]

# 预测分类
print(model.predict(new_observation))

# 给定每个分类的先验概率，创建一个高斯朴素贝叶斯对象
clf = GaussianNB(priors=[0.25, 0.25, 0.5])

# 训练模型
model = classifier.fit(features, target)
