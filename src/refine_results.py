import pandas as pd

# Load CSV
df = pd.read_csv("../temp/BTC_1_min_results.txt")

# Remove duplicates (keep missing data intact)
df = df.drop_duplicates()

# Filter out zero-profit, zero-commission rows
df = df[~((df["total_profit"] == 0) & (df["commission"] == 0) & (df["net_profit"] == 0))]

# Find row with highest net_profit, breaking ties by lowest commission
best_row = df.sort_values(by=["net_profit", "commission"], ascending=[False, True]).iloc[0]

# Display the result
print("Best configuration:")
print(best_row)