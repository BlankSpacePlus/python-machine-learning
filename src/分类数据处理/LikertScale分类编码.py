import pandas as pd

data_frame = pd.DataFrame({"Score": ["Low", "Low", "Medium", "Medium", "High", "Barely More Than Medium"]})

scale_mapper = {"Low": 1, "Medium": 2, "Barely More Than Medium": 3, "High": 4}

print(data_frame["Score"].replace(scale_mapper))

# 上面的low与medium之间的距离和medium与Barely More Than Medium之间的距离相同是不合理的
scale_mapper = {"Low": 1, "Medium": 2, "Barely More Than Medium": 2.1, "High": 3}
print(data_frame["Score"].replace(scale_mapper))
