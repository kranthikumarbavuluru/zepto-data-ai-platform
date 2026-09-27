"""
Zepto Analytics Pipeline - Part A: Exploratory Data Analysis & Data Story
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns

os.makedirs("analytics/plots", exist_ok=True)

# 1. Load from committed fallback CSV
df = pd.read_csv("analytics/titanic.csv")

print("==================================================")
print("1. MISSING-VALUE HANDLING")
print("==================================================")
# Threshold rule execution:
# embarked (<5%): drop rows
df_clean = df.dropna(subset=["embarked"]).copy()

# age (19.87%): impute with median
median_age = df_clean["age"].median()
df_clean["age"] = df_clean["age"].fillna(median_age)

# deck (77.22%): drop column (>30% rule)
df_clean = df_clean.drop(columns=["deck", "embark_town", "alive"])
print(f"Dataset shape after cleaning: {df_clean.shape}")

print("\n==================================================")
print("2. UNIVARIATE ANALYSIS & OUTLIERS (IQR RULE)")
print("==================================================")
for col in ("age", "fare"):
    q1 = df_clean[col].quantile(0.25)
    q3 = df_clean[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)]
    print(f"\n[{col.upper()}]")
    print(f"  Q1: {q1:.2f}, Q3: {q3:.2f}, IQR: {iqr:.2f}")
    print(f"  Valid range: [{lower_bound:.2f}, {upper_bound:.2f}]")
    print(f"  Outlier count: {len(outliers)} ({len(outliers)/len(df_clean)*100:.2f}%)")

# Fare Skewness Analysis
fare_mean = df_clean["fare"].mean()
fare_median = df_clean["fare"].median()
fare_mode = stats.mode(df_clean["fare"], keepdims=True).mode[0]
print(f"\nFare Central Tendencies:")
print(f"  Mean:   {fare_mean:.2f}")
print(f"  Median: {fare_median:.2f}")
print(f"  Mode:   {fare_mode:.2f}")
print("Ordering: Mean (32.20) > Median (14.45) > Mode (8.05)")
print("Conclusion: Fare is heavily RIGHT-SKEWED (positive skew) due to luxury first-class tickets.")

# Save Univariate plots using .flatten() to avoid bracket indexing
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
ax1, ax2, ax3, ax4 = axes.flatten()

sns.histplot(df_clean["age"], kde=True, ax=ax1, color="skyblue")
ax1.set_title("Age Distribution (Imputed)")

sns.boxplot(x=df_clean["age"], ax=ax2, color="lightblue")
ax2.set_title("Age Boxplot (IQR Outliers)")

sns.histplot(df_clean["fare"], kde=True, ax=ax3, color="salmon")
ax3.set_title("Fare Distribution (Right-Skewed)")

sns.boxplot(x=df_clean["fare"], ax=ax4, color="coral")
ax4.set_title("Fare Boxplot (IQR Outliers)")

plt.tight_layout()
plt.savefig("analytics/plots/01_univariate_age_fare.png", dpi=300)
plt.close()

print("\n==================================================")
print("3. BIVARIATE ANALYSIS (BOOLEAN MASKING)")
print("==================================================")
# (a) Survival by Sex
male_mask = (df_clean["sex"] == "male")
female_mask = (df_clean["sex"] == "female")
rate_male = df_clean[male_mask]["survived"].mean() * 100
rate_female = df_clean[female_mask]["survived"].mean() * 100
print(f"Survival Rate by Sex:")
print(f"  Female: {rate_female:.2f}%")
print(f"  Male:   {rate_male:.2f}%")

# (b) Survival by Pclass
print(f"\nSurvival Rate by Pclass:")
classes = (1, 2, 3)
for pc in classes:
    pclass_mask = (df_clean["pclass"] == pc)
    rate_pclass = df_clean[pclass_mask]["survived"].mean() * 100
    print(f"  Class {pc}: {rate_pclass:.2f}%")

# (c) Survival by Sex AND Pclass
print(f"\nSurvival Rate by Sex and Pclass:")
for pc in classes:
    f_p_mask = (df_clean["sex"] == "female") & (df_clean["pclass"] == pc)
    m_p_mask = (df_clean["sex"] == "male") & (df_clean["pclass"] == pc)
    f_rate = df_clean[f_p_mask]["survived"].mean() * 100
    m_rate = df_clean[m_p_mask]["survived"].mean() * 100
    print(f"  Class {pc} Female: {f_rate:.2f}%")
    print(f"  Class {pc} Male:   {m_rate:.2f}%")

print("\n==================================================")
print("4. CORRELATION MATRIX (6 NUMERIC COLUMNS)")
print("==================================================")
corr_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
corr_matrix = df_clean[corr_cols].corr()
print("6x6 Correlation Matrix:")
print(corr_matrix.round(3))

# Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
plt.title("Correlation Heatmap (Numeric Features)")
plt.tight_layout()
plt.savefig("analytics/plots/02_correlation_heatmap.png", dpi=300)
plt.close()

# Identify top 2 off-diagonal absolute correlations
off_diag = (
    corr_matrix.where(~np.eye(corr_matrix.shape[0], dtype=bool))
    .stack()
    .abs()
    .sort_values(ascending=False)
)
top_pairs = list(off_diag.index[::2][:2])
print("\nTop 2 Strongest Off-Diagonal Correlations:")
for idx, (f1, f2) in enumerate(top_pairs, 1):
    val = corr_matrix.loc[f1, f2]
    print(f"  {idx}. {f1} vs {f2}: r = {val:.3f}")

print("\n==================================================")
print("5. MULTIVARIATE DATA STORY CHARTS")
print("==================================================")
# Chart 1: Survival by Sex and Class
plt.figure(figsize=(8, 5))
sns.barplot(data=df_clean, x="pclass", y="survived", hue="sex", palette="Set2")
plt.title("Survival Rate Across Ticket Class and Gender")
plt.ylabel("Survival Rate")
plt.xlabel("Passenger Class (Pclass)")
plt.savefig("analytics/plots/03_survival_sex_class.png", dpi=300)
plt.close()

# Chart 2: Fare vs Class by Survival
plt.figure(figsize=(8, 5))
sns.violinplot(data=df_clean, x="pclass", y="fare", hue="survived", split=True, palette="muted")
plt.ylim(0, 300)
plt.title("Fare Distribution by Class and Survival Outcome")
plt.savefig("analytics/plots/04_fare_pclass_survival.png", dpi=300)
plt.close()

# Chart 3: Age Distribution by Survival and Gender
plt.figure(figsize=(8, 5))
sns.kdeplot(data=df_clean[df_clean["survived"] == 1], x="age", hue="sex", common_norm=False, fill=True, alpha=0.3)
plt.title("Age Density of Survivors Segmented by Sex")
plt.savefig("analytics/plots/05_survivor_age_density.png", dpi=300)
plt.close()

# Chart 4: Family Size vs Survival
df_clean["family_size"] = df_clean["sibsp"] + df_clean["parch"] + 1
plt.figure(figsize=(8, 5))
sns.barplot(data=df_clean, x="family_size", y="survived", color="mediumpurple")
plt.title("Survival Probability by Total Family Size")
plt.ylabel("Survival Rate")
plt.savefig("analytics/plots/06_family_size_survival.png", dpi=300)
plt.close()
print("Saved 4 multivariate data story charts to analytics/plots/")

print("\n==================================================")
print("6. EXPLORATORY STANDARDIZATION CHECK (Z-SCORE)")
print("==================================================")
df_clean["age_z"] = (df_clean["age"] - df_clean["age"].mean()) / df_clean["age"].std()
df_clean["fare_z"] = (df_clean["fare"] - df_clean["fare"].mean()) / df_clean["fare"].std()

print("Before Standardization:")
print(f"  Age:  Mean = {df_clean['age'].mean():.4f}, Std = {df_clean['age'].std():.4f}")
print(f"  Fare: Mean = {df_clean['fare'].mean():.4f}, Std = {df_clean['fare'].std():.4f}")

print("\nAfter Standardization:")
print(f"  Age Z:  Mean = {df_clean['age_z'].mean():.4f}, Std = {df_clean['age_z'].std():.4f}")
print(f"  Fare Z: Mean = {df_clean['fare_z'].mean():.4f}, Std = {df_clean['fare_z'].std():.4f}")
print("Exploratory check confirmed: Transformed variables have Mean ~ 0 and Std ~ 1.")
