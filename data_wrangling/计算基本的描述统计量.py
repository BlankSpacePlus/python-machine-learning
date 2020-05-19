import pandas as pd

url = 'titanic.csv'
data_frame = pd.read_csv(url)

# 计算描述统计量
print('最大值', data_frame['Age'].max())
print('最小值', data_frame['Age'].min())
print('平均值', data_frame['Age'].mean())
print('总和', data_frame['Age'].sum())
print('计数', data_frame['Age'].count())
print('方差', data_frame['Age'].var())
print('标准差', data_frame['Age'].std())
print('峰态', data_frame['Age'].kurt())
print('偏态', data_frame['Age'].skew())
print('平均值标准误差', data_frame['Age'].sem())
print('众数', data_frame['Age'].mode())
print('中位数', data_frame['Age'].median())

# 对整个数据帧计数
print(data_frame.count())
