# ============================================================
# PHASE 6: FINAL QUALITY VALIDATION & CLEAN DATASET EXPORT
# Dataset: sindh-education-stats_2.csv
# Covers: Step 22 (Final Validation, PCA Verification & Export)
# ============================================================

from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

# ------------------------------------------------------------
# 6.1 CONFIGURATION & LOAD PHASE 4 / PHASE 5 ARTIFACTS
# ------------------------------------------------------------

print("=" * 65)
print("PHASE 6: FINAL QUALITY VALIDATION & CLEAN DATASET EXPORT")
print("=" * 65)

OUTPUT_DIR = Path("Phase6_Outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

clean_dataset_path = Path("P4_encoded_clean_dataset.csv")
train_pca_path = Path("P5_X_train_pca_95.npy")
test_pca_path = Path("P5_X_test_pca_95.npy")
pca_var_path = Path("P5_pca_explained_variance.csv")

# Verify required intermediate files exist
required_files = [clean_dataset_path, train_pca_path, test_pca_path]
missing_files = [str(f) for f in required_files if not f.exists()]

if missing_files:
    raise FileNotFoundError(
        f"Missing required files from Phase 4/5: {missing_files}\n"
        "Please run P1 through P5 before executing Phase 6."
    )

df_clean = pd.read_csv(clean_dataset_path)
X_train_pca_95 = np.load(train_pca_path)
X_test_pca_95 = np.load(test_pca_path)

pca_var_df = pd.read_csv(pca_var_path) if pca_var_path.exists() else None

print("\nArtifacts loaded successfully.")
print("Encoded Clean Dataset Shape:", df_clean.shape)
print("PCA-Reduced Train Set Shape:", X_train_pca_95.shape)
print("PCA-Reduced Test Set Shape :", X_test_pca_95.shape)

# ------------------------------------------------------------
# 6.2 SCHEMA, NULL-VALUE & DUPLICATE INTEGRITY CHECK
# ------------------------------------------------------------

print("\n--- 6.2 SCHEMA, NULL & DUPLICATE AUDIT ---")

null_counts = df_clean.isnull().sum()
total_nulls = int(null_counts.sum())
total_duplicates = int(df_clean.duplicated().sum())
non_numeric_cols = df_clean.select_dtypes(exclude=[np.number]).columns.tolist()

print("Total Missing Cells       :", total_nulls)
print("Exact Duplicate Rows      :", total_duplicates)
print("Non-Numeric Columns Left  :", non_numeric_cols)

if total_nulls > 0:
    print("\nColumns with Remaining Nulls:")
    print(null_counts[null_counts > 0])

# Check PCA arrays for NaN or Inf values
train_finite = bool(np.isfinite(X_train_pca_95).all())
test_finite = bool(np.isfinite(X_test_pca_95).all())
print("PCA Train Array Finite    :", train_finite)
print("PCA Test Array Finite     :", test_finite)

# ------------------------------------------------------------
# 6.3 FINAL DATA TYPES & DESCRIPTIVE SUMMARY
# ------------------------------------------------------------

print("\n--- 6.3 FINAL DATA TYPES & DESCRIPTIVE STATISTICS ---")

print("\nData Types:")
print(df_clean.dtypes)

desc_stats = df_clean.describe().T
desc_stats["Median"] = df_clean.median(numeric_only=True)
desc_stats["Skewness"] = df_clean.skew(numeric_only=True)
desc_stats = desc_stats.round(4)

print("\nFinal Descriptive Statistics:")
print(desc_stats)

desc_stats.to_csv(OUTPUT_DIR / "P6_final_descriptive_statistics.csv")

# ------------------------------------------------------------
# 6.4 BUILD COMPREHENSIVE QUALITY VALIDATION REPORT
# ------------------------------------------------------------

print("\n--- 6.4 QUALITY VALIDATION SUMMARY ---")

retained_variance = "N/A"
n_components_selected = X_train_pca_95.shape[1]

if pca_var_df is not None and "Cumulative_Variance" in pca_var_df.columns:
    retained_variance = f"{pca_var_df['Cumulative_Variance'].iloc[n_components_selected - 1] * 100:.2f}%"

quality_report = pd.DataFrame({
    "Validation_Metric": [
        "Timestamp",
        "Final Clean Dataset Rows",
        "Final Clean Dataset Columns",
        "Remaining Missing Cells",
        "Remaining Duplicate Rows",
        "All Columns Numeric (ML-Ready)",
        "PCA Components Retained (95% Threshold)",
        "Total Variance Retained by PCA",
        "PCA Train Array Shape",
        "PCA Test Array Shape",
        "PCA Arrays NaN/Inf Free"
    ],
    "Status_or_Value": [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        df_clean.shape[0],
        df_clean.shape[1],
        total_nulls,
        total_duplicates,
        len(non_numeric_cols) == 0,
        n_components_selected,
        retained_variance,
        str(X_train_pca_95.shape),
        str(X_test_pca_95.shape),
        train_finite and test_finite
    ]
})

print(quality_report.to_string(index=False))
quality_report.to_csv(OUTPUT_DIR / "P6_final_quality_report.csv", index=False)

# ------------------------------------------------------------
# 6.5 EXPORT FINAL CLEAN DATASET & PCA-TRANSFORMED DATASETS
# ------------------------------------------------------------

print("\n--- 6.5 EXPORTING FINAL DELIVERABLES ---")

# 1. Export clean, encoded tabular dataset
final_csv_path = Path("sindh_education_cleaned_pca_ready.csv")
df_clean.to_csv(final_csv_path, index=False)
df_clean.to_csv(OUTPUT_DIR / final_csv_path.name, index=False)

# 2. Export PCA-transformed train and test sets as labeled CSVs for easy inspection
pca_cols = [f"PC{i+1}" for i in range(n_components_selected)]

pd.DataFrame(X_train_pca_95, columns=pca_cols).to_csv(
    OUTPUT_DIR / "P6_X_train_pca_95.csv", index=False
)
pd.DataFrame(X_test_pca_95, columns=pca_cols).to_csv(
    OUTPUT_DIR / "P6_X_test_pca_95.csv", index=False
)

# 3. Export plain-text run summary
summary_lines = [
    "PHASE 6: FINAL QUALITY VALIDATION & EXPORT SUMMARY",
    "=" * 55,
    f"Generated At            : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    f"Final Clean Rows        : {df_clean.shape[0]}",
    f"Final Clean Columns     : {df_clean.shape[1]}",
    f"Remaining Nulls         : {total_nulls}",
    f"Remaining Duplicates    : {total_duplicates}",
    f"PCA Components (95%)    : {n_components_selected}",
    f"Variance Retained       : {retained_variance}",
    "",
    "Exported Files:",
    f"1. {final_csv_path}",
    f"2. {OUTPUT_DIR / 'P6_final_quality_report.csv'}",
    f"3. {OUTPUT_DIR / 'P6_final_descriptive_statistics.csv'}",
    f"4. {OUTPUT_DIR / 'P6_X_train_pca_95.csv'}",
    f"5. {OUTPUT_DIR / 'P6_X_test_pca_95.csv'}",
]

(OUTPUT_DIR / "P6_run_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

print("\n" + "=" * 65)
print("PHASE 6 COMPLETED SUCCESSFULLY!")
print("=" * 65)
print(f"Primary Clean Dataset : {final_csv_path}")
print(f"All Phase 6 Reports   : {OUTPUT_DIR.resolve()}")