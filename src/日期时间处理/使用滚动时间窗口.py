import pandas as pd

'''
滚动（有时称为移动）时间窗口的概念很简单。
假设我们对股票价格的观测值是以月为单位的。
  如果拥有一个确定的月份数量的时间窗口，并且在所有的观察值中移动这个窗口，那么计算时间窗口中所有的观察值的统计量是很有价值的。
例：有一个宽度为3个月的时间窗口，要求滚动平均值，算法：
  STEP1 求1月、2月、3月的平均值
  STEP2 求2月、3月、4月的平均值
  STEP3 求3月、4月、5月的平均值
  STEP...以此类推
也就是说，我们用一个宽度为3个月的时间窗口走完了所有的观察值，每走一步就会计算这个窗口内所有观察值的平均值。
滚动平均值通常用于对时间序列数据做平滑处理，因为使用整个时间窗口的平均值能够削弱短期波动的影响。
我们可以使用pandas中rolling的window参数指定窗口的大小，还可以计算max()、mean()、count()、corr()
'''

# 创建datetime
time_index = pd.date_range("01/01/2010", periods=5, freq="M")

# 创建数据帧，设置索引
data_frame = pd.DataFrame(index=time_index)

# 创建特征
data_frame["Stock_Price"] = [1, 2, 3, 4, 5]

# 计算滚动平均值
print(data_frame.rolling(window=2).mean())
