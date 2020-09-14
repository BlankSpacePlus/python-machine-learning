import pandas as pd

# 创建数据帧
data_frame = pd.DataFrame()

# 创建数据
data_frame["dates"] = pd.date_range("1/1/2001", periods=5, freq="D")
data_frame["stock_price"] = [1.1, 2.2, 3.3, 4.4, 5.5]

# 让值滞后一行(使用历史数据做预测 被称为 滞后一个特征)
data_frame["previous_days_stock_price"] = data_frame["stock_price"].shift(1)

print(data_frame)
