import pandas as pd

url = 'titanic.csv'
data_frame = pd.read_csv(url)


# 定义一个函数
def uppercase(x):
    return x.upper()


# 应用函数，查看两行
print(data_frame['Name'].apply(uppercase)[0:2])
