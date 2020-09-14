from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import datasets

'''
对于给定的一组特征，在保留信息量的同时减少特征的数量。
主成分分析法（PCA）是一种流行的线性降维方法。
PCA将样本数据映射到特征矩阵的成分空间（主成分空间保留了大部分的数据差异，一般都具有更低的数据维度）。
PCA是一种无监督学习方法，也就是说它只考虑了特征矩阵而不需要目标向量的信息。
'''
# 加载数据
digits = datasets.load_digits()

# 标准化特征矩阵
features = StandardScaler().fit_transform(digits.data)

# 创建可以保留99%系信息量（用方差表示）的PCA
# 还有一个参数：svd_solver=True，表示对每一个主成分都进行转换以保证它们的平均值是1
pca = PCA(n_components=0.99, whiten=True)

# 执行PCA
features_pca = pca.fit_transform(features)

# 显示结果
print("Original number of features: ", features.shape[1])
print("Reduced number of features: ", features_pca.shape[1])
