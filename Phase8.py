# ============================================================
# PHASE 8: STATISTICAL HYPOTHESIS TESTING
# Dataset: sindh-education-stats_2.csv (Cleaned Output from P2)
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon, pearsonr, spearmanr, chi2_contingency

print("=" * 65)
print("PHASE 8: STATISTICAL HYPOTHESIS TESTING")
print("=" * 65)

OUTPUT_DIR = Path("Phase8_Stats_Outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("P2_cleaned_data.csv")
test_results = []

def record_test(name, stat, p_val):
    sig = "Significant (Reject H0)" if p_val < 0.05 else "Not Significant (Fail to Reject H0)"
    test_results.append({
        "Statistical_Test": name,
        "Test_Statistic": round(float(stat), 4),
        "P_Value": round(float(p_val), 6),
        "Conclusion_at_Alpha_0.05": sig
    })

# ------------------------------------------------------------
# 8.1 PAIRED T-TEST & WILCOXON: URBAN VS RURAL INSTITUTIONS
# ------------------------------------------------------------

ur_df = df[
    (df["Category"] == "INSTITUTIONS") &
    (df["Location"].isin(["Urban", "Rural"])) &
    (df["Composition"] == "TOTAL")
].pivot_table(index=["District", "School Type"], columns="Location", values="Number", aggfunc="sum").dropna()

t_stat, p_val_t = ttest_rel(ur_df["Urban"], ur_df["Rural"])
w_stat, p_val_w = wilcoxon(ur_df["Urban"], ur_df["Rural"])

record_test("Paired t-test: Urban vs Rural Institutions", t_stat, p_val_t)
record_test("Wilcoxon Signed-Rank: Urban vs Rural Institutions", w_stat, p_val_w)

# ------------------------------------------------------------
# 8.2 PAIRED T-TEST: MALE VS FEMALE ENROLMENT
# ------------------------------------------------------------

mf_df = df[
    (df["Category"] == "ENROLMENT") &
    (df["Location"] == "Total") &
    (df["Composition"].isin(["MALE", "FEMALE"]))
].pivot_table(index=["District", "School Type"], columns="Composition", values="Number", aggfunc="sum").dropna()

t_mf, p_mf = ttest_rel(mf_df["MALE"], mf_df["FEMALE"])
record_test("Paired t-test: Male vs Female Enrolment", t_mf, p_mf)

# ------------------------------------------------------------
# 8.3 CORRELATION SIGNIFICANCE: POPULATION VS INSTITUTIONS
# ------------------------------------------------------------

tot_inst = df[
    (df["Category"] == "INSTITUTIONS") &
    (df["Location"] == "Total") &
    (df["Composition"] == "TOTAL")
].groupby("District")["Number"].sum()

dist_pop = df.groupby("District")["Population"].first()
merged_dist = pd.concat([dist_pop, tot_inst], axis=1).dropna()

r_pearson, p_pearson = pearsonr(merged_dist["Population"], merged_dist["Number"])
r_spearman, p_spearman = spearmanr(merged_dist["Population"], merged_dist["Number"])

record_test("Pearson Correlation: Population vs Total Institutions", r_pearson, p_pearson)
record_test("Spearman Correlation: Population vs Total Institutions", r_spearman, p_spearman)

# ------------------------------------------------------------
# 8.4 CHI-SQUARE TEST: DISTRICT VS SCHOOL TYPE INDEPENDENCE
# ------------------------------------------------------------

contingency = df[
    (df["Category"] == "INSTITUTIONS") &
    (df["Location"] == "Total") &
    (df["Composition"] == "TOTAL")
].pivot_table(index="District", columns="School Type", values="Number", aggfunc="sum", fill_value=0)

chi2, p_chi2, dof, _ = chi2_contingency(contingency)
record_test(f"Chi-Square Test (dof={dof}): District vs School Type", chi2, p_chi2)

# Export Results Table
results_df = pd.DataFrame(test_results)
print("\n--- 8.5 HYPOTHESIS TESTING SUMMARY TABLE ---")
print(results_df.to_string(index=False))
results_df.to_csv(OUTPUT_DIR / "P8_hypothesis_test_results.csv", index=False)

print("\nPHASE 8 COMPLETED SUCCESSFULLY!")