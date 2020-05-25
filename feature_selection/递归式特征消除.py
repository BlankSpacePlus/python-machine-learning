import warnings
from sklearn.datasets import make_regression
from sklearn.feature_selection import RFECV
from sklearn import datasets, linear_model

# 忽略一些烦人但无害的警告信息
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

# 生成特征矩阵、目标向量以及范数
features, target = make_regression(n_samples=10000, n_features=100, n_informative=2, random_state=1)

# 创建线性回归对象
ols = linear_model.LinearRegression()

# 递归消除特征
rfecv = RFECV(estimator=ols, step=1, scoring="neg_mean_squared_error")
rfecv.fit(features, target)
print(rfecv.transform(features))

# 最优特征的数量
print(rfecv.support_)

# 将特征从最好(1)到最差排序
print(rfecv.ranking_)
