# ============================================================
# PHASE 5: PRINCIPAL COMPONENT ANALYSIS (PCA)
# Dataset: sindh-education-stats_2.csv
# Covers: Steps 19 to 21
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ------------------------------------------------------------
# 5.1 LOAD SCALED DATA FROM PHASE 4
# ------------------------------------------------------------

X_train_scaled = np.load("P4_X_train_scaled.npy")
X_test_scaled = np.load("P4_X_test_scaled.npy")

print("=" * 65)
print("PHASE 5: PRINCIPAL COMPONENT ANALYSIS (PCA)")
print("=" * 65)

# ------------------------------------------------------------
# 5.2 FIT FULL PCA & INSPECT EXPLAINED VARIANCE (Step 19)
# ------------------------------------------------------------

print("\n--- 5.2 FULL PCA DECOMPOSITION ---")

pca = PCA()
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

explained_var = pca.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var)

variance_df = pd.DataFrame({
    "Component": [f"PC{i+1}" for i in range(len(explained_var))],
    "Explained_Variance_Ratio": explained_var.round(4),
    "Cumulative_Variance": cumulative_var.round(4)
})

print(variance_df.to_string(index=False))
variance_df.to_csv("P5_pca_explained_variance.csv", index=False)

# ------------------------------------------------------------
# 5.3 PLOT EXPLAINED VARIANCE & SCREE CURVE (Step 20)
# ------------------------------------------------------------

print("\n--- 5.3 SCREE & CUMULATIVE VARIANCE PLOT ---")

components = range(1, len(cumulative_var) + 1)

plt.figure(figsize=(9, 5))
plt.bar(components, explained_var, alpha=0.45, color="teal", label="Individual Variance")
plt.plot(components, cumulative_var, marker="o", linestyle="--", color="navy", label="Cumulative Variance")
plt.axhline(y=0.95, color="red", linestyle=":", linewidth=1.5, label="95% Variance Threshold")

plt.title("PCA Scree & Cumulative Explained Variance — Sindh Education")
plt.xlabel("Number of Principal Components")
plt.ylabel("Explained Variance Ratio")
plt.xticks(components)
plt.legend(loc="best")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("P5_pca_explained_variance_plot.png", dpi=150)
plt.show()

min_components_95 = int(np.argmax(cumulative_var >= 0.95) + 1)
print(f"Estimated minimum components for >= 95% variance: {min_components_95}")

# ------------------------------------------------------------
# 5.4 RETAIN 95% VARIANCE WITH PCA (Step 21)
# ------------------------------------------------------------

print("\n--- 5.4 DIMENSIONALITY REDUCTION (95% VARIANCE) ---")

pca_95 = PCA(n_components=0.95)
X_train_pca_95 = pca_95.fit_transform(X_train_scaled)
X_test_pca_95 = pca_95.transform(X_test_scaled)

print("Components Selected by Scikit-Learn:", pca_95.n_components_)
print("Total Variance Retained:", round(float(pca_95.explained_variance_ratio_.sum()), 4))
print("Reduced X_train Shape:", X_train_pca_95.shape)
print("Reduced X_test Shape:", X_test_pca_95.shape)

np.save("P5_X_train_pca_95.npy", X_train_pca_95)
np.save("P5_X_test_pca_95.npy", X_test_pca_95)

print("\nPHASE 5 COMPLETED SUCCESSFULLY!")