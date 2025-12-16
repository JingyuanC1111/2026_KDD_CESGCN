import pandas as pd

df = pd.read_csv('CDC_DATA/target-hospital-admissions_20251206.csv')
removable = ['US']
df['date'] = pd.to_datetime(df['date']).dt.date
df = df[~df['location'].isin(removable)]
df['location'] = df['location'].astype(int).astype(str).str.zfill(2)
available_data = df.pivot(index='location', columns='date', values='value')

available_data = available_data.T
available_data.fillna(0, inplace=True)
available_data = available_data.iloc[1:,:]
available_data.to_csv('CDC_DATA/transferred_hospital_admission_20251206.csv')