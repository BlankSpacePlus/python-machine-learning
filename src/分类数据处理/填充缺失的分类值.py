import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer

# 用分类特征创建特征矩阵
x = np.array([[0, 2.10, 1.45], [1, 1.18, 1.33], [0, 1.22, 1.27], [1, -0.21, -1.19]])

# 创建带缺失值的特征矩阵
x_with_nan = np.array([[np.nan, 0.87, 1.31], [np.nan, -0.67, -0.22]])

# 训练KNN分类器
clf = KNeighborsClassifier(3, weights='distance')
trained_model = clf.fit(x[:, 1:], x[:, 0])

# 预测缺失值的分类
imputed_values = trained_model.predict(x_with_nan[:, 1:])

# 将所预测的分类和它们的其他特征连接起来
x_with_imputed = np.hstack((imputed_values.reshape(-1, 1), x_with_nan[:, 1:]))

# 连接两个特征矩阵
print(np.vstack((x_with_imputed, x)))

# 还可以用特征中出现最多的值来填充缺失值
x_complete = np.vstack((x_with_nan, x))
imputer = SimpleImputer(strategy='most_frequent')

print(imputer.fit_transform(x_complete))
