import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# 创建文本
text_data = np.array(['I love China, China!', 'Sweden is best', 'Germany beats both'])

# 创建TF-IDF特征矩阵
tfidf = TfidfVectorizer()
feature_matrix = tfidf.fit_transform(text_data)

# 查看TF-IDF的稀疏特征矩阵
print(feature_matrix)

# 查看TF-IDF特征矩阵的稠密矩阵的形式
print(feature_matrix.toarray())

# 查看特征的名字
print(tfidf.vocabulary_)
