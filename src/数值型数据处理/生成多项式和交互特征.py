import numpy as np
from sklearn.preprocessing import PolynomialFeatures

'''
当特征和目标值（预测值）之间存在非线性关系的时候，就需要创建多项式特征。
另外，有时候我们会遇到一个特征需要依赖于另一个特征才能对目标值造成影响的情况，
  生成一个交互特征（将两特征相乘），我们就可以为这种关系编码
'''

# 创建特征矩阵
features = np.array([[2, 3], [2, 3], [2, 3]])

# 创建PolynomialFeatures对象(degree=2表示阶数最高为2)
polynomial_interaction = PolynomialFeatures(degree=2, include_bias=False)

# 创建多项式特征
print(polynomial_interaction.fit_transform(features))

interaction = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
print(interaction.fit_transform(features))
