import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# 创建文本
text_data = np.array(['I love China. China!', 'China is best', 'Germany beats both'])

# 创建词袋
count = CountVectorizer()
bag_of_words = count.fit_transform(text_data)

# 创建特征矩阵
features = bag_of_words.toarray()

# 创建目标向量
target = np.array([0, 0, 1])

# 给定每个分类的先验概率，创建一个多项式朴素贝叶斯对象
classifier = MultinomialNB(class_prior=[0.25, 0.5])

# 训练模型
model = classifier.fit(features, target)

# 创建一个观察值
new_observation = [[0, 0, 0, 1, 0, 1, 0]]

# 预测新观察值的分类
print(model.predict(new_observation))
