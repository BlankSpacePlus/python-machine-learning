from sklearn.decomposition import PCA, KernelPCA
from sklearn.datasets import make_circles

'''
对线性不可分数据进行特征降维。
PCA能降低特征矩阵的维度（如减少特征数量）。标准PCA使用线性映射减少特征的数量。
如果数据不是线性可分的（也就是说，可以用一条直线或超平面将两类数据分开），那么PCA的处理效果很好。
理想情况下，我们希望维度变换既能降低数据的维度，又能使数据变得线性可分。
KPCA，核主成分分析，PCA的一种扩展，能做到这点。
kernel，也叫核函数，能够将线性不可分的数据映射到更高的维度，数据在这个维度是线性可分的，我们把这种方式叫核机制。
kernel参数指定常用核函数，如高斯径向基函数（rbf）、多项式核（poly）、sigmoid核（sigmoid）等等。
我们甚至可以指定一个线性映射，利用它可以得到与标准PCA相同的结果。
KPCA必须指定参数的数量，此外每个核都有自己的超参数要设置，如rbf的gamma。
参数的设置就需要反复试错调整参数了，即使用不同的核函数和参数反复训练机器学习模型，找出最优模型的参数组合。
'''
# 标准化特征矩阵
features, _ = make_circles(n_samples=1000, random_state=1, noise=0.1, factor=0.1)

# 基于径向基函数核（RBF）的 Kernel PCA 方法
kpca = KernelPCA(kernel="rbf", gamma=15, n_components=1)
features_kpca = kpca.fit_transform(features)

# 显示结果
print("Original number of features: ", features.shape[1])
print("Reduced number of features: ", features_kpca.shape[1])
