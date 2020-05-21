import pandas as pd

# 创建数据帧
data_frame = pd.DataFrame()

# 创建datetime
data_frame['date'] = pd.date_range('1/1/2001', periods=100000, freq='H')

# 筛选出两个日期之间的观察值
print(data_frame[(data_frame['date'] > '2002-1-1 01:00:00') & (data_frame['date'] <= '2002-1-1 04:00:00')])

# 将date设为数据帧索引项
date_frame = data_frame.set_index(data_frame['date'])

# 选择两个日期之间的观察值
print(date_frame.loc['2002-1-1 01:00:00': '2002-1-1 04:00:00'])
