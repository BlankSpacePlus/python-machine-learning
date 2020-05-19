import pandas as pd

url = 'titanic.csv'
data_frame = pd.read_csv(url)

# 删除一些行，查看输出结果的前两行
print(data_frame[data_frame['Sex'] != 'male'].head(2))

# 使用布尔条件通过匹配唯一值的方式可以轻松删除一行
print(data_frame[data_frame['Name'] != 'Allison, Miss Helen Loraine'].head(2))

# 可以依据行的下标删除一行
print(data_frame[data_frame.index != 0].head(2))
