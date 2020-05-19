import pandas as pd

url = 'titanic.csv'
data_frame = pd.read_csv(url)

# 替换掉一些值，并查看两行数据
print(data_frame['Sex'].replace("female", 'Woman').head(2))

# 替换多个值
print(data_frame['Sex'].replace(["female", "male"], ["Woman", "Man"]).head(5))

# 替换一些值，并查看两行数据
print(data_frame['Sex'].replace(1, "One").head(2))

# 使用正则表达式替换一些值并查看两行数据
print(data_frame['Sex'].replace(r"1st", "First").head(2))
