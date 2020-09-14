import numpy as np
from sklearn.naive_bayes import BernoulliNB

# 创建三个二元特征
features = np.random.randint(2, size=(100, 3))

# 创建一个二元目标向量
target = np.random.randint(2, size=(100, 1)).ravel()

# 给定每个分类的先验概率，创建一个多项式伯努利朴素贝叶斯对象
classifier = BernoulliNB(class_prior=[0.25, 0.5])

# 训练模型
model = classifier.fit(features, target)

# 想指定统一的先验概率，设置fit_prior=False
model_uniform_prior = BernoulliNB(class_prior=None, fit_prior=False)
