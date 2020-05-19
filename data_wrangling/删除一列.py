import pandas as pd

url = 'titanic.csv'
data_frame = pd.read_csv(url)

# 删除列
print(data_frame.drop('Age', axis=1).head(2))

# 删除多列
print(data_frame.drop(['Age', 'Sex'], axis=1).head(2))

# 删除一列
print(data_frame.drop(data_frame.columns[1], axis=1).head(2))

# 将数据帧视为不可变对象，创建一个新的数据帧
data_frame_name_dropped = data_frame.drop(data_frame.columns[0], axis=1)
