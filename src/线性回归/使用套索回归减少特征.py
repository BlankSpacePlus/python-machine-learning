from sklearn.linear_model import Lasso
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler

# 加载数据
boston = load_boston()
features = boston.data
target = boston.target

# 特征标准化
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

# 创建套索回归，并指定alpha值
regression = Lasso(alpha=0.5)

# 拟合线性回归
model = regression.fit(features_standardized, target)

# 查看系数
print(model.coef_)

# 创建一个alpha值为10的套索回归
regression_a10 = Lasso(alpha=10)
model_a10 = regression_a10.fit(features_standardized, target)
print(model_a10.coef_)
