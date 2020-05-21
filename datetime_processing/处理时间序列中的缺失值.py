import pandas as pd
import numpy as np

'''
插值法是一种填充由缺失值造成的数据缺口的技术。
插值法实际上就是根据缺口附近的已知数据来画一条直线或者曲线，然后利用这条直线或曲线预测合理的值。
当时间间隔确定，数据不会产生剧烈波动且缺失值比较小的时候，插值法尤为有用。
'''

# 创建日期
time_index = pd.date_range("01/01/2010", periods=5, freq="M")

# 创建数据帧，设置索引
data_frame = pd.DataFrame(index=time_index)

# 创建带缺失值数据的特征
data_frame["Sales"] = [1.0, 2.0, np.nan, np.nan, 5.0]

# 对缺失值进行插值
print(data_frame.interpolate())

# 向前填充（用前面的值来替换缺失值）
print(data_frame.ffill())

# 向后填充（用后面的值来替换缺失值）
print(data_frame.bfill())

# 对缺失值进行插值（已知两个已知点之间的线是非线性的）
print(data_frame.interpolate(method="quadratic"))

# 对缺失值进行插值（缺口可能很大，我们不想对整个缺口进行插值，使用limit可以限制插值的数量，limit_direction="forward"表示向前插值）
print(data_frame.interpolate(limit=1, limit_direction="forward"))
