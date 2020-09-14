import pandas as pd

url = 'titanic.csv'
data_frame = pd.read_csv(url)

# 对行分组，然后在每一组上应用函数
print(data_frame.groupby('Sex').apply(lambda x: x.count()))
