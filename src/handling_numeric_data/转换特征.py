import numpy as np
from sklearn.preprocessing import FunctionTransformer
import pandas as pd

'''
对一个或多个特征进行自定义的转换
'''

features = np.array([[2, 3], [2, 3], [2, 3]])


def add_ten(x):
    return x+10


# 创建转换器
ten_transformer = FunctionTransformer(add_ten)

# 转换特征矩阵
print(ten_transformer.transform(features))

# 创建数据帧，使用Pandas实现同样的效果
data_frame = pd.DataFrame(features, columns=["feature_1", "feature_2"])
print(data_frame.apply(add_ten))
