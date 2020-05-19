import pandas as pd

url = 'titanic.csv'
data_frame = pd.read_csv(url)

print(data_frame.rename(columns={'PClass': 'Passenger Class'}).head(2))

print(data_frame.rename(columns={'PClass': 'Passenger Class', 'Sex': 'Gender'}).head(2))
