from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

# 生成特征矩阵、目标向量以及模型的系数
features, target, coefficients = make_regression(n_samples=100, n_features=3, n_informative=3, n_targets=1, noise=0.0,
                                                 coef=True, random_state=1)

# 查看特征矩阵和目标向量
print('Feature Matrix\n', features[:3])
print('Target Vector\n', target[:3])

# 查看数据散点图
plt.scatter(features[:, 0], features[:, 1], c=target)
plt.show()
