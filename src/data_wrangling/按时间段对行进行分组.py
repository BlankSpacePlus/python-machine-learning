import pandas as pd
import numpy as np

# 创建日期范围
time_index = pd.date_range('05/13/2020', periods=100000, freq='30S')

# 创建数据帧
data_frame = pd.DataFrame(index=time_index)

# 创建一列随机变量
data_frame['SaleAmount'] = np.random.randint(1, 10, 100000)

# 按周对行进行分组，计算每周的总和
print(data_frame.resample('W').sum())

# 查看前三行
print(data_frame.head(3))

# 按两周分组，计算平均值
print(data_frame.resample('2W').mean())

# 按月分组，计算行数（返回时间组右边界的值）
print(data_frame.resample('M').count())

# 按月分组，计算行数（返回时间组左边界的值）
print(data_frame.resample('M', label='left').count())
