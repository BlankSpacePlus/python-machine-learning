import pandas as pd
from sqlalchemy import create_engine

# 创建一个数据库连接
database_connection = create_engine('sqlite:///sqlite_load_test.db')

# 加载数据
data_frame = pd.read_sql_query('SELECT * FROM COMPANY', database_connection)

# 查看前两行数据
# print(data_frame.head(2))
print(data_frame)