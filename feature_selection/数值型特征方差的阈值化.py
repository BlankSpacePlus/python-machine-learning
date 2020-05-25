from sklearn import datasets
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

# 加载数据
iris = datasets.load_iris()

# 创建features和target
features = iris.data
target = iris.target

# 创建VarianceThreshold对象
thresholder = VarianceThreshold(threshold=.5)

# 创建大方差特征矩阵
features_high_variance = thresholder.fit_transform(features)

# 显示大方差特征矩阵
print(features_high_variance[0:3])

# 显示方差
print(thresholder.fit(features).variances_)

# 标准化特征矩阵
scaler = StandardScaler()
features_std = scaler.fit_transform(features)

# 计算每个特征值的方差
selector = VarianceThreshold()
print(selector.fit(features_std).variances_)
