import pandas as pd

# 创建一个URL
url = 'xlsx_load_test.xlsx'

# 加载数据
data_frame = pd.read_excel(url, sheet_name=0, header=0)

# 查看前两行
print(data_frame.head(2))
