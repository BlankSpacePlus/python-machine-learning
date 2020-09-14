import pandas as pd

url = 'titanic.csv'

# 加载数据
data_frame = pd.read_csv(url)

# 查看前5行数据
print(data_frame.head(5))
