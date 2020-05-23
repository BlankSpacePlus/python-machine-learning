from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston

# 加载只有两个特征值的数据集
boston = load_boston()
features = boston.data[:, 0:2]
target = boston.target

# 创建线性回归对象
regression = LinearRegression()

# 拟合线性回归模型
model = regression.fit(features, target)

# 查看截距
print(model.intercept_)

# 显示特征值的权重
print(model.coef_)

# 目标向量的第一个值乘以1000
print(target[0]*1000)

# 预测第一个样本的目标值，并乘以1000
print(model.predict(features)[0]*1000)

# 第一个系数乘以1000
print(model.coef_[0]*1000)
