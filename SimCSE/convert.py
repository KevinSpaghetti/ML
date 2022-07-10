import pandas as pd

df = pd.read_pickle('meddra_data.pkl')['ENG']

df.to_csv('./all_meddras.csv', index=False)
print(df.head())
