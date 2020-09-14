from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

# 生成特征矩阵和目标向量
features, target = make_regression(n_samples=100, n_features=3, n_informative=3, n_targets=1, noise=50, coef=False,
                                   random_state=1)

# 创建LinerRegression对象
ols = LinearRegression()

# 使用MSE对线性回归做交叉验证
print(cross_val_score(ols, features, target, scoring='neg_mean_squared_error'))

# 使用决定系数对线性回归进行交叉验证
print(cross_val_score(ols, features, target, scoring='r2'))
