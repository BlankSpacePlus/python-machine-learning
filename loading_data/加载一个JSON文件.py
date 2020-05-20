import pandas as pd

# 创建一个URL
url = 'json_load_test.json'

# 加载数据
data_frame = pd.read_json(url, orient='columns')

# 查看前两行数据
print(data_frame.head(2))

