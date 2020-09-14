import numpy as np
from sklearn import preprocessing

'''
对一个特征进行转换，使其平均值为0、标准差为1
'''

# 创建特征
x = np.array([[-1000.1], [-200.2], [500.5], [600.6], [9000.9]])

# 标准化方法创建缩放器
scaler = preprocessing.StandardScaler()

# 转换特征
standardized = scaler.fit_transform(x)

# 查看特征
print(standardized)

# 打印平均值和标准差，以查看标准化效果
print("平均值：", round(standardized.mean()))
print("标准差：", standardized.std())

# 使用中位数和四分位数间距进行缩放
robust_scaler = preprocessing.RobustScaler()

# 转换特征
print(robust_scaler.fit_transform(x))
