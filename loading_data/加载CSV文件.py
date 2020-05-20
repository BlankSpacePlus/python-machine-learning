import pandas as pd

# 创建URL
url = 'csv_load_test.csv'

# 加载数据集
data_frame = pd.read_csv(url)

# 查看前三行数据
print(data_frame.head(3))
