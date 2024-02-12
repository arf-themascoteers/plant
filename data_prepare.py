import pandas as pd
import numpy as np

df = pd.read_csv("data/info.csv")
df['is_train'] = np.where(np.random.rand(len(df)) < 0.9, 1, 0)

df.to_csv("data/info_modified.csv", index=False)
