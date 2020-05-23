from sklearn.decomposition import NMF
from sklearn import datasets

'''
NMF是一种无监督的线性降维方法，它可以分解矩阵（将特征矩阵分解为多个矩阵，其乘积近似于原始矩阵），
  将特征矩阵转换为表示样本与特征之间潜在关系的矩阵。
简单地说，NMF可以减少维数，因为在矩阵乘法中，两个因子（相乘的矩阵）的维数要比得到的乘积矩阵的维数低得多。
正式地，给定一个期望的返回特征的数量，NMF将把特征矩阵分解为：
        V ≈ WH
其中V是d×n维特征矩阵（d个特征，n个样本），W为d×r维矩阵，H是r×n维矩阵，通过调整值，可以设定期望减少的维数。
如果要使用NMA，特征矩阵就不能包含负数值。
此外，在PCA等技术不同，NMA不会告诉我们输出特征中保留了原始数据的信息量。
因此，找出参数n_components的最优质的的最佳方法是不断尝试一系列可能的值，直到找出能生成最佳学习模型的值。
'''

digits = datasets.load_digits()
features = digits.data

nmf = NMF(n_components=10, random_state=1, max_iter=10000)
features_nmf = nmf.fit_transform(features)

print("Original number of features: ", features.shape[1])
print("Reduced number of features: ", features_nmf.shape[1])

