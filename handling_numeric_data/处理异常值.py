import pandas as pd
import numpy as np

'''
异常值处理方法一：丢弃异常值
'''
# 创建数据帧
houses = pd.DataFrame()
houses['Price'] = [534433, 392333, 293222, 4322032]
houses['Bathrooms'] = [2, 3.5, 2, 116]
houses['Square_Feet'] = [1500, 2500, 1500, 48000]

# 筛选观察值
print(houses[houses['Bathrooms'] < 20])

'''
异常值处理方法二：标记为异常值并作为数据的一个特征
'''
# 基于布尔条件语句来创建特征
houses["Outlier"] = np.where(houses["Bathrooms"] < 20, 0, 1)

# 查看数据
print(houses)

'''
异常值处理方法三：对有异常值的特征进行转换，降低异常值的影响
'''
# 对特征取对数值
houses["Log_Of_Square_Feet"] = [np.log(x) for x in houses["Square_Feet"]]

# 查看数据
print(houses)
