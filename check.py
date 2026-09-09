import pandas as pd

df = pd.read_csv('data/dropout_dataset_10000.csv')
print(f'Dataset loaded: {df.shape[0]:,} rows x {df.shape[1]} columns')
print(f'Dropout rate:   {df["Dropout_Status"].mean()*100:.1f}%')
print(f'Columns:        {list(df.columns)}')
print()
print('Step 1 complete. Ready for Step 2.')