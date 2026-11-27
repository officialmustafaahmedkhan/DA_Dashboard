# ============================================================
# PHASE 9: SUPERVISED ML CLASSIFICATION ON PCA FEATURES
# Dataset: sindh-education-stats_2.csv (Uses P4 & P5 Outputs)
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

print("=" * 65)
print("PHASE 9: SUPERVISED ML MODELING & EVALUATION")
print("=" * 65)

OUTPUT_DIR = Path("Phase9_ML_Outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# 9.1 LOAD PCA-TRANSFORMED FEATURES & TARGET ARRAYS
# ------------------------------------------------------------

X_train = np.load("P5_X_train_pca_95.npy")
X_test = np.load("P5_X_test_pca_95.npy")
y_train = np.load("P4_y_train.npy")
y_test = np.load("P4_y_test.npy")

class_names = ["ENROLMENT", "INSTITUTIONS", "TEACHING STAFF"]

print("X_train (PCA 95%) Shape:", X_train.shape)
print("X_test  (PCA 95%) Shape:", X_test.shape)

# ------------------------------------------------------------
# 9.2 TRAIN & COMPARE MULTIPLE CLASSIFIERS
# ------------------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

comparison_rows = []
trained_models = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    trained_models[name] = (model, y_pred)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    comparison_rows.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "Precision_Weighted": round(prec, 4),
        "Recall_Weighted": round(rec, 4),
        "F1_Score_Weighted": round(f1, 4)
    })

comparison_df = pd.DataFrame(comparison_rows).sort_values(by="Accuracy", ascending=False)
print("\n--- 9.2 MODEL PERFORMANCE COMPARISON ---")
print(comparison_df.to_string(index=False))
comparison_df.to_csv(OUTPUT_DIR / "P9_model_comparison.csv", index=False)

# ------------------------------------------------------------
# 9.3 DETAILED REPORT & CONFUSION MATRIX FOR BEST MODEL
# ------------------------------------------------------------

best_model_name = comparison_df.iloc[0]["Model"]
_, best_preds = trained_models[best_model_name]

print(f"\n--- 9.3 CLASSIFICATION REPORT ({best_model_name}) ---")
print(classification_report(y_test, best_preds, target_names=class_names))

cm = confusion_matrix(y_test, best_preds)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title(f"Confusion Matrix — {best_model_name} (PCA Features)")
plt.xlabel("Predicted Category")
plt.ylabel("Actual Category")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "P9_best_model_confusion_matrix.png", dpi=150)
plt.show()

print("\nPHASE 9 COMPLETED SUCCESSFULLY!")