import pandas as pd

# 创建datetime
print(pd.Timestamp('2017-05-01 06:00:00', tz='Europe/London'))

# 创建datetime
date = pd.Timestamp('2017-05-01')
# 设置时区
date_in_london = date.tz_localize('Europe/London')
# 转换时区
date_in_london.tz_convert('Asia/Tokyo')

# 创建三个日期
dates = pd.Series(pd.date_range('2/2/2002', periods=3, freq='M'))
# 设置时区
dates.dt.tz_localize('Asia/Tokyo')
# 查看时间
print(dates)
