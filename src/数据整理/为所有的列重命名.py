import collections
import pandas as pd

url = 'titanic.csv'
data_frame = pd.read_csv(url)

# 创建字典
column_names = collections.defaultdict(str)

# 创建键
for name in data_frame.columns:
    column_names[name]

# 查看字典
print(column_names)

collections.defaultdict(str, {'Age': ' ', 'Name': ' ', 'PClass': ' ', 'Sex': ' ', 'SexCode': ' ', 'Survived': ' '})
