"""
Part A - Step 1: Profiling and Saving Offline Fallback
"""
import os
import pandas as pd
import seaborn as sns

# 1. Load dataset (fetch or fallback)
csv_path = "analytics/titanic.csv"
if not os.path.exists(csv_path):
    print("Fetching dataset via sns.load_dataset('titanic')...")
    df_raw = sns.load_dataset("titanic")
    # Save offline fallback immediately
    df_raw.to_csv(csv_path, index=False)
    print(f"Saved committed fallback to {csv_path}")
else:
    print(f"Loading from committed fallback: {csv_path}")
    df_raw = pd.read_csv(csv_path)

print("\n--- DATA SHAPE ---")
print(f"Rows: {df_raw.shape[0]}, Columns: {df_raw.shape}")

print("\n--- DATA INFO ---")
df_raw.info()

print("\n--- MISSING VALUE PERCENTAGES ---")
missing = df_raw.isnull().mean() * 100
missing_cols = missing[missing > 0].sort_values(ascending=False)
for col, pct in missing_cols.items():
    print(f"{col:15s}: {pct:6.2f}% missing ({df_raw[col].isnull().sum()} rows)")

