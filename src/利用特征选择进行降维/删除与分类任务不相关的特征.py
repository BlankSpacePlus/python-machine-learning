from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.feature_selection import chi2, f_classif

# 加载数据
iris = load_iris()
features = iris.data
target = iris.target

# 将分类数据转换成整数型数据
features = features.astype(int)

# 选择卡方统计量最大的两个特征
chi2_selector = SelectKBest(chi2, k=2)
features_kbest = chi2_selector.fit_transform(features, target)

# 显示结果
print("Origin number of features: ", features.shape[1])
print("Reduced number of features: ", features_kbest.shape[1])

# 选择F值位于前75%的特征
fvalue_selector = SelectPercentile(f_classif, percentile=75)
features_kbest = fvalue_selector.fit_transform(features, target)

# 显示结果
print("Origin number of features: ", features.shape[1])
print("Reduced number of features: ", features_kbest.shape[1])
