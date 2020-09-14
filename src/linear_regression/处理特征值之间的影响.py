from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

# 加载只有两个特征的数据集
boston = load_boston()
features = boston.data[:, 0:2]
target = boston.target

# 创建交互特征
interaction = PolynomialFeatures(degree=3, include_bias=False, interaction_only=True)
features_interaction = interaction.fit_transform(features)

# 创建线性回归对象
regression = LinearRegression()

# 拟合线性回归
model = regression.fit(features_interaction, target)

# 查看第一个样本的特征
print(features[0])

# 将每个样本的第一个和第二个特征相乘
interaction_term = np.multiply(features[:, 0], features[:, 1])

# 查看第一个样本的交互特征
print(interaction_term[0])

# 观察第一个样本的值
print(features_interaction[0])
