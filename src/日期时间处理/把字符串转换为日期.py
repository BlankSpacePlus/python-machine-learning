import numpy as np
import pandas as pd

# 创建时间字符串
date_strings = np.array(['03-04-2005 11:35 PM', '23-05-2010 12:01 AM', '04-09-2009 09:09 PM'])

'''
时间格式：
| format  |   描述                  |     例子  |
| %Y      | 完整的年份               |    2020  |
| %m      | 月，首位空缺时需用0填充   |     05   |
| %d      | 日，首位空缺时需用0填充   |     21   |
| %I      | 时，首位空缺时需用0填充   |     14   |
| %p      | AM 或 PM                |     PM   | 
| %M      | 分，首位空缺时需用0填充   |     40   | 
| %S      | 秒，首位空缺时需用0填充   |     30   | 
'''

# 转换成datetime类型的数据
print([pd.to_datetime(date, format='%d-%m-%Y %I:%M %p') for date in date_strings])

# 转换成datetime类型的数据（加一个errors参数来处理错误）
# errors="coerce"意味着当转换出现错误时不会抛出异常（默认行为），但是会将导致这个错误的值设置为NaT（也就是缺失错误）
print([pd.to_datetime(date, format='%d-%m-%Y %I:%M %p', errors="coerce") for date in date_strings])

