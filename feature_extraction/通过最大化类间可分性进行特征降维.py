from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

'''
LDA，线性判别分析法，将数据集映射到一个可以使类间可分性最大的成分坐标轴上。
LDA是一种分类方法，也是常用的降维方法，它与PCA的原理类似，它将特征空间映射到较低维的空间。
然而在PCA中，只需关注使数据差异最大化的成分轴。
^
|
|     O               O
|___________________________>
上图所示例子中，数据集具有两个分类、两个特征的数据构成。
将数据集投到y轴上，那么两类数据将不易分离；在x轴上，只剩下一个特征向量（降维一度），并能保持类的可分性。
实际情境中可能会更复杂，但概念和思路是一样的。
'''

# 加载 Iris flower 数据集
iris = datasets.load_iris()
features = iris.data
target = iris.target

# 创建并运行LDA，然后用它对特征做变换
lda = LinearDiscriminantAnalysis(n_components=1)
features_lda = lda.fit(features, target).transform(features)

# 打印特征的数量
print("Original number of features: ", features.shape[1])
print("Reduced number of features: ", features_lda.shape[1])

# 查看每个成分保留的信息量（即数据的差异）情况
print(lda.explained_variance_ratio_)

# 创建并运行LDA，然后用它对特征做变换
lda = LinearDiscriminantAnalysis(n_components=None)
feature_lda = lda.fit(features, target)

# 获取方差的百分比数组
lda_var_ratios = lda.explained_variance_ratio_


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


print(select_n_components(lda_var_ratios, 0.95))
