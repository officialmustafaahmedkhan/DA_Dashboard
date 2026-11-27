# ============================================================
# PHASE 10: UNSUPERVISED K-MEANS CLUSTERING & FINAL REPORT
# Dataset: sindh-education-stats_2.csv
# ============================================================

from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

print("=" * 65)
print("PHASE 10: DISTRICT K-MEANS CLUSTERING & FINAL REPORT")
print("=" * 65)

OUTPUT_DIR = Path("Phase10_Final_Outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("P2_cleaned_data.csv")

# ------------------------------------------------------------
# 10.1 AGGREGATE DISTRICT-LEVEL PROFILES
# ------------------------------------------------------------

inst_dist = df[
    (df["Category"] == "INSTITUTIONS") & (df["Location"] == "Total") & (df["Composition"] == "TOTAL")
].groupby("District")["Number"].sum().rename("Total_Institutions")

enrol_dist = df[
    (df["Category"] == "ENROLMENT") & (df["Location"] == "Total") & (df["Composition"] == "TOTAL")
].groupby("District")["Number"].sum().rename("Total_Enrolment")

staff_dist = df[
    (df["Category"] == "TEACHING STAFF") & (df["Location"] == "Total") & (df["Composition"] == "TOTAL")
].groupby("District")["Number"].sum().rename("Total_Teaching_Staff")

pop_dist = df.groupby("District")["Population"].first().rename("Population")

district_df = pd.concat([inst_dist, enrol_dist, staff_dist, pop_dist], axis=1).dropna()

# Scale district features
scaler = StandardScaler()
X_dist_scaled = scaler.fit_transform(district_df)

# ------------------------------------------------------------
# 10.2 FIND OPTIMAL CLUSTERS (K=2 TO 5) VIA SILHOUETTE SCORE
# ------------------------------------------------------------

best_k, best_score, best_labels = 2, -1, None
sil_records = []

for k in range(2, 6):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_dist_scaled)
    score = silhouette_score(X_dist_scaled, labels)
    sil_records.append({"K": k, "Silhouette_Score": round(score, 4)})
    if score > best_score:
        best_k, best_score, best_labels = k, score, labels

print("\n--- 10.2 SILHOUETTE SCORES ---")
print(pd.DataFrame(sil_records).to_string(index=False))

district_df["Cluster"] = best_labels
district_df.to_csv(OUTPUT_DIR / "P10_district_clusters.csv")

print(f"\nSelected Optimal Clusters: K = {best_k} (Silhouette = {best_score:.4f})")
print("\nCluster Mean Profiles:")
print(district_df.groupby("Cluster").mean().round(2))

# ------------------------------------------------------------
# 10.3 2D PCA VISUALIZATION OF DISTRICT CLUSTERS
# ------------------------------------------------------------

pca_2d = PCA(n_components=2)
coords = pca_2d.fit_transform(X_dist_scaled)

plt.figure(figsize=(10, 7))
for c in sorted(district_df["Cluster"].unique()):
    mask = district_df["Cluster"] == c
    plt.scatter(coords[mask, 0], coords[mask, 1], label=f"Cluster {c}", s=70, alpha=0.85)

for idx, district_name in enumerate(district_df.index):
    plt.annotate(district_name, (coords[idx, 0], coords[idx, 1]), fontsize=8, alpha=0.85)

plt.title(f"Sindh Districts K-Means Clustering (K={best_k}) Projected via 2D PCA")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "P10_district_clusters_pca.png", dpi=150)
plt.show()

# ------------------------------------------------------------
# 10.4 GENERATE FINAL LAB SUMMARY REPORT (MARKDOWN)
# ------------------------------------------------------------

report_text = f"""# Sindh Education Statistics — Complete Data Analytics Lab Report
**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Pipeline Summary
1. **Preprocessing (`P1`–`P6`):** Cleaned `{len(df):,}` records across `{df['District'].nunique()}` districts, standardized categorical labels, capped IQR outliers, engineered `Log_Number`, and reduced dimensionality to **5 PCA components** retaining **95.96% variance**.
2. **Exploratory Analysis (`P7`):** Generated district institution rankings, Urban vs. Rural comparisons, and Male vs. Female enrolment share breakdowns.
3. **Statistical Testing (`P8`):** Executed Paired t-tests, Wilcoxon signed-rank tests, Pearson/Spearman correlations, and Chi-Square independence tests (`Phase8_Stats_Outputs/P8_hypothesis_test_results.csv`).
4. **Supervised ML Classification (`P9`):** Trained Logistic Regression, KNN, Decision Tree, and Random Forest classifiers on PCA features (`Phase9_ML_Outputs/P9_model_comparison.csv`).
5. **Unsupervised Clustering (`P10`):** Segmented Sindh districts into **{best_k} clusters** (Silhouette Score = **{best_score:.4f}**) based on infrastructure, enrolment, teaching staff, and population.
"""

(OUTPUT_DIR / "P10_Final_Lab_Report.md").write_text(report_text, encoding="utf-8")

print("\n" + "=" * 65)
print("PHASE 10 COMPLETED! ENTIRE END-TO-END ANALYTICS PIPELINE FINISHED.")
print("=" * 65)