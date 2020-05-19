import pandas as pd

# 创建数据帧
data_a = {'id': ['1', '2', '3'], 'first': ['Alex', 'Amy', 'Allen'], 'last': ['Bonder', 'Ackerman', 'Ali']}
data_frame_a = pd.DataFrame(data_a, columns=['id', 'first', 'last'])
data_b = {'id': ['4', '5', '6'], 'first': ['Billy', 'Brian', 'Bran'], 'last': ['Bonder', 'Black', 'Balwer']}
data_frame_b = pd.DataFrame(data_b, columns=['id', 'first', 'last'])

# 沿着行的方向连接两个数据帧
print(pd.concat([data_frame_a, data_frame_b], axis=0))

# 沿着列的方向连接两个数据帧
print(pd.concat([data_frame_a, data_frame_b], axis=1))

# 创建一行
row = pd.Series([10, 'Chris', 'Chillon'], index=['id', 'first', 'last'])

# 使用append附加一行
print(data_frame_a.append(row, ignore_index=True))
