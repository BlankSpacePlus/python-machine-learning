from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
from sklearn import datasets
import numpy as np

'''
TSVD与PCA相似，事实上PCA常常在某一个步骤使用非截断奇异值分解(SVD)法。
常规SVD中，对于给定的d个特征，SVD将创建d×d维的因子矩阵，而TSVD将返回n×n维的因子矩阵，其中n是预先指定的参数。
与PCA相比，TSVD的优势在于它适用于稀疏特征矩阵。
但是TSVD有一个问题，其输出值的符号会在多次拟合中不断变化（这是由其使用的随机数生成器的方式决定的）。
  一个简单的方法是对每一个预处理管道只是用一次fit()，然后多次使用transform()。
与LDA一样，TSVD需要通过参数n_components指定想要输出的特征（成分）数。
想寻找最佳特征数的一种方法是在模型选择时将n_components作为超参数进行优化。
由于TSVD提供了每个成分保留的原始特征矩阵信息的比例，因而我们也可以按照要保留的信息量（常用值为95%和99%）选择成分。
'''

# 加载数据
digits = datasets.load_digits()

# 标准化特征矩阵
features = StandardScaler().fit_transform(digits.data)

# 生成稀疏矩阵
features_sparse = csr_matrix(features)

# 创建tsvd
tsvd = TruncatedSVD(n_components=10)

# 在稀疏矩阵上执行TSVD
# tsvd.fit(features_sparse).transform(features_sparse) 有问题
features_sparse_tsvd = tsvd.fit_transform(features_sparse)

print("Original number of features: ", features.shape[1])
print("Reduced number of features: ", features_sparse_tsvd.shape[1])

# 对前三个成分的信息量占比求和（能保留大概30%左右的原始数据信息）
print(tsvd.explained_variance_ratio_[0:3].sum())

# 用比原特征数量小1的值作为n_components的值，创建并运行TSVD
tsvd = TruncatedSVD(n_components=features_sparse.shape[1]-1)
features_tsvd = tsvd.fit(features)

# 获取方差百分比数组
tsvd_var_ratios = tsvd.explained_variance_ratio_


# 定义函数
def select_n_components(var_radio, goal_var: float) -> int:
    # 设置总方差的初始值
    total_variance = 0.0
    # 设置特征数量的初始值
    n_components = 0
    # 遍历方差百分比数组的元素
    for explained_variance in var_radio:
        # 将该百分比加入总方差
        total_variance += explained_variance
        # n_components值+1
        n_components += 1
        # 如果达到目标阈值
        if total_variance >= goal_var:
            # 结束遍历
            break
    return n_components


print(select_n_components(tsvd_var_ratios, 0.95))
