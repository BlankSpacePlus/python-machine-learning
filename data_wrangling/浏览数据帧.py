import pandas as pd

url = 'titanic.csv'
data_frame = pd.read_csv(url)

# 选择第一行数据
print(data_frame.iloc[0])

# 选择2、3、4三行
print(data_frame.iloc[1:4])

# 获取从第1行到第5行为止的所有人
print(data_frame.iloc[:5])

# 设置索引
data_frame = data_frame.set_index(data_frame['Name'])

# 查看行
print(data_frame.loc['Allen, Miss Elisabeth Walton'])
