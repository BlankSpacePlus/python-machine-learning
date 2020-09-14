import pandas as pd

url = 'titanic.csv'
data_frame = pd.read_csv(url)

# 展示Sex列的值是female的前两行
print(data_frame[data_frame['Sex'] == 'female'].head(2))

# 过滤行
print(data_frame[(data_frame['Sex'] == 'female') & (data_frame['Age'] >= 65)])
