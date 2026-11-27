# ============================================================
# PHASE 3: FEATURE ENGINEERING, SUBSETTING & MULTICOLLINEARITY
# Dataset: sindh-education-stats_2.csv
# Covers: Steps 11 to 14
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------
# 3.1 LOAD PHASE 2 OUTPUT
# ------------------------------------------------------------

df = pd.read_csv("P2_cleaned_data.csv")

print("=" * 65)
print("PHASE 3: FEATURE ENGINEERING & MULTICOLLINEARITY ANALYSIS")
print("=" * 65)

# ------------------------------------------------------------
# 3.2 GENERATE NEW ANALYTICAL FEATURES (Step 11)
# ------------------------------------------------------------

print("\n--- 3.2 FEATURE GENERATION ---")

# 1. Reconstructed count from Population and No/100000 (deliberately collinear with Number)
if {"Population", "No/100000"}.issubset(df.columns):
    df["Expected_Count_From_Rate"] = ((df["Population"] / 100000.0) * df["No/100000"]).round(2)

# 2. Percentage share of district population represented by 'Number'
if {"Number", "Population"}.issubset(df.columns):
    df["Population_Share_Pct"] = ((df["Number"] / df["Population"].replace(0, np.nan)) * 100).round(4)

# 3. Log-transformed count to reduce right-skewness
if "Number" in df.columns:
    df["Log_Number"] = np.log1p(df["Number"]).round(4)

print("Engineered features added. Current columns:")
print(df.columns.tolist())

# ------------------------------------------------------------
# 3.3 CREATE ANALYTICAL DATA SUBSET (Step 12)
# ------------------------------------------------------------

print("\n--- 3.3 FEATURE SUBSETTING ---")

# Exclude high-cardinality regional identifier 'District' from continuous modeling subset
drop_cols = [col for col in ["District"] if col in df.columns]
df_subset = df.drop(columns=drop_cols).copy()

print("Dropped identifier/grouping column:", drop_cols)
print("Subset Shape:", df_subset.shape)

# ------------------------------------------------------------
# 3.4 CORRELATION ANALYSIS (Step 13)
# ------------------------------------------------------------

print("\n--- 3.4 CORRELATION ANALYSIS ---")

numeric_df = df_subset.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr()
corr_matrix.to_csv("P3_correlation_matrix.csv")

plt.figure(figsize=(10, 7))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Heatmap — Sindh Education Numerical Features")
plt.tight_layout()
plt.savefig("P3_correlation_heatmap.png", dpi=150)
plt.show()

corr_pairs = corr_matrix.unstack()
corr_pairs = corr_pairs[corr_pairs.index.get_level_values(0) < corr_pairs.index.get_level_values(1)]

print("\nStrongest Positive Correlations:")
print(corr_pairs.sort_values(ascending=False).head(5))

print("\nStrongest Negative Correlations:")
print(corr_pairs.sort_values(ascending=True).head(5))

# ------------------------------------------------------------
# 3.5 MULTICOLLINEARITY & REDUNDANT FEATURE REMOVAL (Step 14)
# ------------------------------------------------------------

print("\n--- 3.5 MULTICOLLINEARITY CHECK (|r| > 0.85) ---")

high_corr = corr_pairs[corr_pairs.abs() > 0.85]
print(high_corr)

# Retain primary 'Number' and 'No/100000'; remove redundant derived features
redundant_cols = [
    col for col in ["Expected_Count_From_Rate", "Population_Share_Pct"]
    if col in df_subset.columns
]

df_subset = df_subset.drop(columns=redundant_cols)
print("Removed redundant/multicollinear columns:", redundant_cols)

df_subset.to_csv("P3_engineered_subset.csv", index=False)
print("\nPHASE 3 COMPLETED SUCCESSFULLY!")