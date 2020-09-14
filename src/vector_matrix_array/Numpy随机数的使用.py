# 加载库
import numpy as np

# 设置随机数种子
np.random.seed(0)

# 生成3个0.0~1.0之间的浮点随机数
print(np.random.random(3))

# 生成3个1~10之间的随机整数
print(np.random.randint(0, 11, 3))

# 从平均值是0.0，标准差是1.0的正态分布中抽取3个数
print(np.random.normal(0.0, 1.0, 3))

# 从平均值是0.0，散布程度是1.0的logistic分布中抽取3个数
print(np.random.logistic(0.0, 1.0, 3))

# 从大于等于1.0，小于2.0的范围内抽取3个数
print(np.random.uniform(1.0, 2.0, 3))
