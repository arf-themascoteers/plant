import pandas as pd
import numpy as np

df = pd.read_csv("data/info_modified.csv")
num_train_rows = df['is_train'].sum()
print("Number of rows where 'is_train' is 1:", num_train_rows)
num_nonzero_train_rows = (df['is_train'] != 1).sum()
print("Number of rows where 'is_train' is not 0:", num_nonzero_train_rows)

print(len(df))







