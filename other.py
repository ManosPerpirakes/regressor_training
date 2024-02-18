import pandas as pd
df = pd.read_csv("results.csv")
print(df.head())
df = df.drop(['Unnamed: 0'], axis=1)
print(df.head())
df.to_csv("results.csv", index=False)