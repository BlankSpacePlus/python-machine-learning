import pandas as pd

data_frame = pd.DataFrame()

# 创建两个datetime特征
data_frame['Arrived'] = [pd.Timestamp('01-01-2017'), pd.Timestamp('01-04-2017')]
data_frame['Left'] = [pd.Timestamp('01-01-2017'), pd.Timestamp('01-06-2017')]

# 计算两个特征之间的时间间隔
print(data_frame['Left'] - data_frame['Arrived'])

# 计算两个特征之间的时间间隔（移除days，只保留数值）
print(pd.Series(delta.days for delta in (data_frame['Left'] - data_frame['Arrived'])))
