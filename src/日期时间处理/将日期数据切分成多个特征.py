import pandas as pd

data_frame = pd.DataFrame()

data_frame['date'] = pd.date_range('1/1/2001', periods=150, freq='M')

# 创建年、月、日、时、分的特征
data_frame['year'] = data_frame['date'].dt.year
data_frame['month'] = data_frame['date'].dt.month
data_frame['day'] = data_frame['date'].dt.day
data_frame['hour'] = data_frame['date'].dt.hour
data_frame['minute'] = data_frame['date'].dt.minute

print(data_frame)
