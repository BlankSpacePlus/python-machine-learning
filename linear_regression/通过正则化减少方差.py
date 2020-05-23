from sklearn.linear_model import Ridge, RidgeCV
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler

# 加载数据
boston = load_boston()
features = boston.data
target = boston.target

# 特征标准化
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

# 创建一个包含指定alpha值的岭回归
regression = Ridge(alpha=0.5)

# 拟合线性回归模型
model = regression.fit(features_standardized, target)

# 创建包含3个alpha值的RidgeCV对象
regression_cv = RidgeCV(alphas=[0.1, 1.0, 10.0])

# 拟合线性回归
model_cv = regression_cv.fit(features_standardized, target)

# 查看模型的参数
print(model_cv.coef_)

# 查看alpha的值
print(model_cv.alpha_)
