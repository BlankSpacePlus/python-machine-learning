import pandas as pd
import numpy as np

url = 'titanic.csv'
data_frame = pd.read_csv(url)

# 筛选出缺失值，查看前两行
print(data_frame[data_frame['Age'].isnull()].head(2))

'''
处理缺失值既是难以避免的，但可能并不简单
处理缺失值可以使用NaN(Not A Number)
可惜Pandas并不支持NaN，所以需要使用Numpy包
'''
# data_frame['Sex'] = data_frame['Sex'].replace('male', np.nan)
print(data_frame['Sex'].replace('male', np.nan))

# 加载数据，设置缺失值
data_frame = pd.read_csv(url, na_values=[np.nan, 'NONE', -999])
print(data_frame)
