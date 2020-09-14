import pandas as pd
from sklearn.feature_extraction import DictVectorizer

# 为4个文档创建词频字典
doc_1_word_count = {"Red": 2, "Blue": 4}
doc_2_word_count = {"Red": 4, "Blue": 3}
doc_3_word_count = {"Red": 1, "Yellow": 2}
doc_4_word_count = {"Red": 2, "Yellow": 2}

# 创建列表
doc_word_counts = [doc_1_word_count, doc_2_word_count, doc_3_word_count, doc_4_word_count]

# 将词频字典列表转换成特征矩阵
dictvectorizer = DictVectorizer(sparse=False)
print(dictvectorizer.fit_transform(doc_word_counts))
