import pandas as pd

# 创建特征
data_frame = pd.DataFrame({"Score": ["Low", "Low", "Medium", "Medium", "High"]})

# 创建映射器
scale_mapper = {"Low": 1, "Medium": 2, "High": 3}

# 使用映射器
print(data_frame["Score"].replace(scale_mapper))
