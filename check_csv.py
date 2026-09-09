import pandas as pd
df = pd.read_csv('model/model_comparison.csv')
print(df.columns.tolist())
print(df.head())