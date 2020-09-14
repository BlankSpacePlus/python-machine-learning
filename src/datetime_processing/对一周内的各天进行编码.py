import pandas as pd

# 创建日期
dates = pd.Series(pd.date_range("2/2/2002", periods=3, freq="M"))

# 查看星期几
print(dates.dt.day_name())

# 只显示数值
print(dates.dt.weekday)
