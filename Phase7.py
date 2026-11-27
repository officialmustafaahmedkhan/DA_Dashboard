# ============================================================
# PHASE 7: EXPLORATORY DATA ANALYSIS (EDA) & VISUALIZATIONS
# Dataset: sindh-education-stats_2.csv (Cleaned Output from P2)
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 65)
print("PHASE 7: EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 65)

OUTPUT_DIR = Path("Phase7_EDA_Outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("P2_cleaned_data.csv")
print("Loaded Cleaned Dataset Shape:", df.shape)

# ------------------------------------------------------------
# 7.1 GROUPBY SUMMARY TABLES (DISTRICT & CATEGORY)
# ------------------------------------------------------------

print("\n--- 7.1 TOP 10 DISTRICTS BY TOTAL INSTITUTIONS ---")
institutions = df[
    (df["Category"] == "INSTITUTIONS") &
    (df["Location"] == "Total") &
    (df["Composition"] == "TOTAL")
]

district_inst = (
    institutions.groupby("District")["Number"]
    .sum()
    .sort_values(ascending=False)
)
print(district_inst.head(10))
district_inst.to_csv(OUTPUT_DIR / "P7_district_institutions_summary.csv", header=["Total_Institutions"])

# ------------------------------------------------------------
# 7.2 VISUALIZATION 1: URBAN VS RURAL COMPARISON
# ------------------------------------------------------------

urban_rural = df[
    (df["Category"] == "INSTITUTIONS") &
    (df["Location"].isin(["Urban", "Rural"])) &
    (df["Composition"] == "TOTAL")
]

ur_pivot = urban_rural.pivot_table(
    index="School Type", columns="Location", values="Number", aggfunc="sum"
)
print("\n--- 7.2 URBAN VS RURAL INSTITUTIONS BY SCHOOL TYPE ---")
print(ur_pivot)

ur_pivot.plot(kind="bar", figsize=(9, 5), color=["#1f77b4", "#ff7f0e"])
plt.title("Urban vs. Rural Institutions by School Type")
plt.xlabel("School Type")
plt.ylabel("Total Institutions (Capped)")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "P7_urban_vs_rural_institutions.png", dpi=150)
plt.show()

# ------------------------------------------------------------
# 7.3 VISUALIZATION 2: MALE VS FEMALE ENROLMENT SHARE
# ------------------------------------------------------------

enrolment = df[
    (df["Category"] == "ENROLMENT") &
    (df["Location"] == "Total") &
    (df["Composition"].isin(["MALE", "FEMALE"]))
]

gender_pivot = enrolment.pivot_table(
    index="School Type", columns="Composition", values="Number", aggfunc="sum"
)
gender_pivot["Female_Share_%"] = (
    gender_pivot["FEMALE"] / (gender_pivot["MALE"] + gender_pivot["FEMALE"]) * 100
).round(2)

print("\n--- 7.3 MALE VS FEMALE ENROLMENT BY SCHOOL TYPE ---")
print(gender_pivot)
gender_pivot.to_csv(OUTPUT_DIR / "P7_gender_enrolment_summary.csv")

gender_pivot[["MALE", "FEMALE"]].plot(kind="bar", figsize=(9, 5), color=["#2ca02c", "#d62728"])
plt.title("Male vs. Female Enrolment by School Type")
plt.xlabel("School Type")
plt.ylabel("Recorded Enrolment")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "P7_male_vs_female_enrolment.png", dpi=150)
plt.show()

# ------------------------------------------------------------
# 7.4 VISUALIZATION 3: BOXPLOT ACROSS CATEGORIES
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Category", y="Number", palette="Set2")
plt.title("Distribution of 'Number' Across Education Categories")
plt.xlabel("Education Category")
plt.ylabel("Recorded Count (IQR Capped)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "P7_category_boxplot.png", dpi=150)
plt.show()

print("\nPHASE 7 COMPLETED SUCCESSFULLY!")