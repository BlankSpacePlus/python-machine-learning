import pandas as pd

# 创建数据帧
data_frame = pd.DataFrame()

# 增加列
data_frame['Name'] = ['小明', '小红']
data_frame['Age'] = [13, 15]
data_frame['Sex'] = ['男♂', '女♀']

# 查看数据帧
print(data_frame)

# 底部添加新的元组
print(data_frame.append(pd.Series(['小强', 14, '男♂'], index=['Name', 'Age', 'Sex']), ignore_index=True))
