import pandas as pd

url = 'titanic.csv'
data_frame = pd.read_csv(url)

# 删除重复行，查看输出结果的前两行
print(data_frame.drop_duplicates().head(2))

'''
上述方案其实并没有删除任何行，我们可以查看一下行数，是相同的
drop_duplicates()只默认删除那些所有列都完美匹配的行，dataFrame的每一列都是唯一的
'''
print("是否没有删除任何行：", len(data_frame) == len(data_frame.drop_duplicates()))

# 使用subset参数
print(data_frame.drop_duplicates(subset=['Sex']))

# 使用keep参数
print(data_frame.drop_duplicates(subset=['Sex'], keep='last'))
