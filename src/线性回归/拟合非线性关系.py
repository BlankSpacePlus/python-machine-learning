from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.preprocessing import PolynomialFeatures

# 加载只有一个特征的数据集
boston = load_boston()
features = boston.data[:, 0:1]
target = boston.target

# 创建多项式特征x^2和x^3
polynomial = PolynomialFeatures(degree=3, interaction_only=False)
features_polynomial = polynomial.fit_transform(features)

# 创建线性回归对象
regression = LinearRegression()

# 拟合线性回归模型
model = regression.fit(features_polynomial, target)

# 观察第一个样本
print(features[0])

# 观察第一个样本的平方值
print(features[0]**2)

# 观察第一个样本的三次方值
print(features[0]**3)

# 观察第一个样本的所有三个特征x、x的平方和x的三次方
print(features_polynomial[0])
