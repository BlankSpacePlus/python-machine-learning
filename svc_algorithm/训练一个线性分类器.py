from sklearn.svm import LinearSVC
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

'''
想要训练一个模型对观察值进行分类，可以使用支持向量机（SVC）来寻找最大化分类间距的超平面
'''
# 加载数据，数据中只有两种分类和两个特征
iris = datasets.load_iris()
features = iris.data[:100, :2]
target = iris.target[:100]

# 标准化特征
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

# 创建支持向量机分类器
svc = LinearSVC(C=1.0)

# 训练模型
model = svc.fit(features_standardized, target)

# 画出样本点，并根据其分类上色
color = ["#6ce73c" if c == 0 else "#e63d32" for c in target]
plt.scatter(features_standardized[:, 0], features_standardized[:, 1], c=color)

# 创建超平面
w = svc.coef_[0]
a = - w[0] / w[1]
xx = np.linspace(-2.5, 2.5)
yy = a * xx - (svc.intercept_[0]) / w[1]

# 画出超平面
plt.plot(xx, yy)
plt.axis("off"), plt.show()

# 创建新的样本点
new_observation = [[-2, 3]]

# 预测新样本点的分类
print(svc.predict(new_observation))
