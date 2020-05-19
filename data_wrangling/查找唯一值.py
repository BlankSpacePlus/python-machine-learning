import pandas as pd

url = 'titanic.csv'
data_frame = pd.read_csv(url)

# 筛选出唯一值
print(data_frame['Sex'].unique())

# 查看计数
print(data_frame['Sex'].value_counts())

# 数据中存在异常数据*
print(data_frame['PClass'].value_counts())

# 统计唯一值个数
print(data_frame['PClass'].nunique())
