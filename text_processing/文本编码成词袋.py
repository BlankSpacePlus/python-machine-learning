import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# 创建文本
text_data = np.array(['I love Sam. Sam!', 'Sam is best', 'Oh nice'])

# 创建一个词袋特征矩阵
count = CountVectorizer()
# 得到的是稀疏矩阵
bag_of_words = count.fit_transform(text_data)
print(bag_of_words.toarray())

# 查看特征名
print(count.get_feature_names())

# 创建一个只包含Sam的词袋特征矩阵
count_2gram = CountVectorizer(ngram_range=(1, 2), stop_words="english", vocabulary=['sam'])
bag = count_2gram.fit_transform(text_data)

# 查看特征矩阵
print(bag.toarray())

# 查看一元模型和二元模型
print(count_2gram.vocabulary_)
