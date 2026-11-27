# ============================================================
# PHASE 1: DATA LOADING & STRUCTURAL INSPECTION
# Dataset: sindh-education-stats_2.csv
# Covers: Steps 1 to 5
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

# ------------------------------------------------------------
# 1.1 LOAD DATASET (Step 1 & Step 2)
# ------------------------------------------------------------

file_path = Path("sindh-education-stats_2.csv")
if not file_path.exists():
    file_path = Path("sindh-education-stats.csv")

df = pd.read_csv(file_path)

print("=" * 65)
print("1.1 DATASET LOADED SUCCESSFULLY")
print("=" * 65)

print("Source File:", file_path.name)
print("Number of Rows:", df.shape[0])
print("Number of Columns:", df.shape[1])
print("Total Cells:", df.size)

print("\nFIRST FIVE RECORDS:")
print(df.head())

print("\nLAST FIVE RECORDS:")
print(df.tail())

# ------------------------------------------------------------
# 1.2 DATASET STRUCTURE & DATA TYPE VALIDATION (Step 3)
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("1.2 DATASET STRUCTURE & DATA TYPE VALIDATION")
print("=" * 65)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nRaw Data Types:")
print(df.dtypes)

# Clean comma-separated strings in 'Population' (e.g., '1,804,516' -> 1804516)
if "Population" in df.columns:
    df["Population"] = pd.to_numeric(
        df["Population"].astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce"
    )

for num_col in ["Number", "No/100000"]:
    if num_col in df.columns:
        df[num_col] = pd.to_numeric(df[num_col], errors="coerce")

print("\nValidated Data Types:")
print(df.dtypes)

print("\nSchema Info:")
df.info()

print("\nDescriptive Summary:")
print(df.describe(include="all"))

# ------------------------------------------------------------
# 1.3 DUPLICATE RECORD INSPECTION & REMOVAL (Step 4)
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("1.3 DUPLICATE DETECTION & REMOVAL")
print("=" * 65)

duplicate_count = df.duplicated().sum()
print("Exact Duplicate Rows Detected:", duplicate_count)

# Export duplicate records for audit before removal
duplicate_records = df[df.duplicated(keep=False)].copy()
duplicate_records.to_csv("P1_duplicate_records.csv", index=False)

# Remove exact duplicates
df = df.drop_duplicates().reset_index(drop=True)
print("Duplicates Removed:", duplicate_count)
print("Shape After Duplicate Removal:", df.shape)

# ------------------------------------------------------------
# 1.4 NULL-VALUE ANALYSIS (Step 5)
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("1.4 NULL-VALUE ANALYSIS")
print("=" * 65)

null_report = pd.DataFrame({
    "Missing_Count": df.isnull().sum(),
    "Missing_Percentage": (df.isnull().mean() * 100).round(2)
}).sort_values(by="Missing_Percentage", ascending=False)

print(null_report)
null_report.to_csv("P1_missing_value_report.csv")

# Save intermediate output for Phase 2
df.to_csv("P1_inspected_data.csv", index=False)

print("\nPHASE 1 COMPLETED SUCCESSFULLY!")