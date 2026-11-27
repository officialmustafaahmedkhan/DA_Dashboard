# ============================================================
# PHASE 2: DATA CLEANING, IMPUTATION & OUTLIER TREATMENT
# Dataset: sindh-education-stats_2.csv
# Covers: Steps 6 to 10
# ============================================================

import numpy as np
import pandas as pd

# ------------------------------------------------------------
# 2.1 LOAD PHASE 1 OUTPUT
# ------------------------------------------------------------

df = pd.read_csv("P1_inspected_data.csv")

print("=" * 65)
print("PHASE 2: DATA CLEANING, IMPUTATION & OUTLIER TREATMENT")
print("=" * 65)

# ------------------------------------------------------------
# 2.2 CLEAN INCONSISTENT CATEGORICAL VALUES (Step 6)
# ------------------------------------------------------------

print("\n--- 2.2 CATEGORICAL STANDARDIZATION ---")

cat_cols = ["District", "Location", "School Type", "Category", "Composition"]

for col in cat_cols:
    if col in df.columns:
        print(f"\nBefore cleaning ({col}) sample:", df[col].unique()[:8])
        df[col] = df[col].astype("string").str.strip()
        df[col] = df[col].replace({"nan": pd.NA, "None": pd.NA, "": pd.NA})

# Standardize casing per domain convention
if "Location" in df.columns:
    df["Location"] = df["Location"].str.title()

if "School Type" in df.columns:
    df["School Type"] = df["School Type"].str.title()

if "Category" in df.columns:
    df["Category"] = df["Category"].str.upper()

if "Composition" in df.columns:
    df["Composition"] = df["Composition"].str.upper()

if "District" in df.columns:
    df["District"] = df["District"].str.title().replace({
        "Matira": "Matiari",
        "Tando Allah Yar": "Tando Allahyar"
    })

for col in cat_cols:
    if col in df.columns:
        print(f"After cleaning ({col}) unique ({df[col].nunique()}):", df[col].unique()[:8])

# ------------------------------------------------------------
# 2.3 CONVERT AND VALIDATE DATE DATA (Step 7)
# ------------------------------------------------------------

print("\n--- 2.3 DATE VALIDATION ---")

date_candidates = [c for c in df.columns if "date" in c.lower()]
if date_candidates:
    for date_col in date_candidates:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        invalid_dates = df[date_col].isnull().sum()
        print(f"Invalid/Missing Dates Converted to NaT in '{date_col}': {invalid_dates}")
        df[date_col] = df[date_col].fillna(df[date_col].median())
else:
    print("No date column present in 'sindh-education-stats_2.csv' — skipping date coercion.")

# ------------------------------------------------------------
# 2.4 IMPUTE MISSING VALUES (Step 8)
# ------------------------------------------------------------

print("\n--- 2.4 MISSING VALUE IMPUTATION ---")

num_cols = ["Number", "Population", "No/100000"]
num_cols = [c for c in num_cols if c in df.columns]

# Median imputation for numerical columns
for col in num_cols:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].median())

# Mode imputation for categorical columns
for col in cat_cols:
    if col in df.columns and df[col].isnull().any():
        mode_vals = df[col].mode(dropna=True)
        if not mode_vals.empty:
            df[col] = df[col].fillna(mode_vals.iloc[0])

print("Remaining Null Cells After Imputation:", df.isnull().sum().sum())

# ------------------------------------------------------------
# 2.5 DETECT & CAP OUTLIERS USING IQR (Step 9 & Step 10)
# ------------------------------------------------------------

print("\n--- 2.5 OUTLIER DETECTION & CAPPING (IQR METHOD) ---")

outlier_report = []

for col in num_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
    outlier_count = int(outlier_mask.sum())

    outlier_report.append({
        "Column": col,
        "Q1": round(q1, 2),
        "Q3": round(q3, 2),
        "IQR": round(iqr, 2),
        "Lower_Bound": round(lower_bound, 2),
        "Upper_Bound": round(upper_bound, 2),
        "Outliers_Capped": outlier_count
    })

    # Cap values at IQR boundaries
    df[col] = np.clip(df[col], lower_bound, upper_bound)
    print(f"{col}: {outlier_count} outliers capped to [{lower_bound:.2f}, {upper_bound:.2f}]")

pd.DataFrame(outlier_report).to_csv("P2_outlier_capping_report.csv", index=False)

df.to_csv("P2_cleaned_data.csv", index=False)
print("\nPHASE 2 COMPLETED SUCCESSFULLY!")