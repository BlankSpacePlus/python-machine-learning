import pandas as pd
from sklearn.feature_extraction import DictVectorizer

# 创建一个字典
data_dict = [{"Red": 2, "Blue": 4}, {"Red": 4, "Blue": 3}, {"Red": 1, "Yellow": 2}, {"Red": 2, "Yellow": 2}]

# 创建字典向量化器(sparse=False强制输出一个稠密矩阵而不是稀疏矩阵)
dictvectorizer = DictVectorizer(sparse=False)

# 将字典转换成特征矩阵
features = dictvectorizer.fit_transform(data_dict)

# 查看特征矩阵
print(features)

# 获取特征的名字
feature_names = dictvectorizer.get_feature_names()

# 查看特征的名字
print(feature_names)

# 使用Pandas查看
data_frame = pd.DataFrame(features, columns=feature_names)
