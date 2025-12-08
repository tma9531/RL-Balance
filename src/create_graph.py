import pandas as pd
import matplotlib.pyplot as plt

# Read your CSV file
df = pd.read_csv("src/data/smoothed_terrain_balance.csv")

# Plot a column vs another column
plt.plot(df["Step"], df["smoothed"])

plt.xlabel("Step")
plt.ylabel("Value")
plt.title("Value vs Step")
plt.show()
