import pandas as pd

# 创建URL
url = 'titanic.csv'

# 将数据作为数据帧加载进来
data_frame = pd.read_csv(url)

# 查看前5行数据
print(data_frame.head(5))
