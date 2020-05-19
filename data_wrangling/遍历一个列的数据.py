import pandas as pd

url = 'titanic.csv'
data_frame = pd.read_csv(url)

# 以大写的形式打印前两行中的名字
for name in data_frame['Name'][0:2]:
    print(name.upper())

# 以大写的形式打印前两行的名字
print([name.upper() for name in data_frame['Name'][0:2]])
