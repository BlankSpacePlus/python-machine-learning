import pandas as pd

# 创建数据帧
employee_data = {'employee_id': ['1', '2', '3', '4'], 'name': ['Amy Jones', 'Allen Keys', 'Alice Bees', 'Tim Horton']}
data_frame_employees = pd.DataFrame(employee_data, columns=['employee_id', 'name'])

sales_data = {'employee_id': ['3', '4', '5', '6'], 'total_sales': [23456, 2512, 2345, 1455]}
data_frame_sales = pd.DataFrame(sales_data, columns=['employee_id', 'total_sales'])

# 合并数据帧（inner join）
print(pd.merge(data_frame_employees, data_frame_sales, on='employee_id'))

# 合并数据帧（outer join）
print(pd.merge(data_frame_employees, data_frame_sales, on='employee_id', how='outer'))

# 合并数据帧（left inner join）
print(pd.merge(data_frame_employees, data_frame_sales, on='employee_id', how='left'))

# 合并数据帧（right inner join）
print(pd.merge(data_frame_employees, data_frame_sales, on='employee_id', how='right'))
