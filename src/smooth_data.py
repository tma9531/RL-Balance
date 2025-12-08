import pandas as pd

df = pd.read_csv("src/data/terrain_balance.csv")
df["smoothed"] = df["Value"].rolling(window=5, center=True).mean()
df.to_csv("src/data/smoothed_terrain_balance.csv", index=False)
