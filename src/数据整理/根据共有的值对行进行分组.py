import pandas as pd

url = 'titanic.csv'
data_frame = pd.read_csv(url)

# 根据Sex列的值来对行进行分组，并计算每一组的平均值
print(data_frame.groupby('Sex').mean())

# 这么写只能返回hashcode
print(data_frame.groupby('Sex'))

# 分组后计算行数
print(data_frame.groupby('Survived')['Name'].count())

# 对行进行分组，计算平均值
print(data_frame.groupby(['Sex', 'Survived'])['Age'].mean())
