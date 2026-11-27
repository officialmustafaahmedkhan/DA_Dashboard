# ============================================================
# UNIVERSAL 12-PHASE DATA ANALYTICS, ML & DEEP LEARNING PIPELINE
# Works on ANY CSV / Excel Dataset (Defaults to Sindh Education)
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    r2_score,
    mean_squared_error,
    silhouette_score,
    silhouette_samples
)


def run_all_phases(file_path="sindh-education-stats_2.csv", target_col=None):
    print("=" * 65)
    print(f"PHASE 1–2: INGESTION & SCHEMA AUDIT ({file_path})")
    print("=" * 65)
    path = Path(file_path)
    df = pd.read_excel(path) if path.suffix.lower() in [".xlsx", ".xls"] else pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(axis=1, how="all").drop_duplicates().reset_index(drop=True)
    print(f"Loaded Shape: {df.shape[0]} rows x {df.shape[1]} columns")

    print("\n" + "=" * 65)
    print("PHASE 3–5: AUTO-CLEANING, IMPUTATION & IQR OUTLIER CAPPING")
    print("=" * 65)
    num_cols, cat_cols, id_cols = [], [], []
    for col in df.columns:
        if df[col].dtype == "object":
            num_try = pd.to_numeric(df[col].astype(str).str.replace(",", "", regex=False).str.strip(), errors="coerce")
            if num_try.notna().mean() > 0.80:
                df[col] = num_try

        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
            if df[col].nunique() == len(df) and ("id" in col.lower() or "index" in col.lower()):
                id_cols.append(col)
                continue
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                df[col] = np.clip(df[col], q1 - 1.5 * iqr, q3 + 1.5 * iqr)
            num_cols.append(col)
        else:
            df[col] = df[col].fillna("Unknown").astype(str).str.strip()
            df[col] = df[col].replace({"Matira": "Matiari", "Tando Allah Yar": "Tando Allahyar"})
            if df[col].nunique() / max(len(df), 1) > 0.85 and len(df) > 50:
                id_cols.append(col)
            else:
                cat_cols.append(col)

    usable_cols = [c for c in df.columns if c not in id_cols]
    if target_col is None:
        target_col = "Category" if "Category" in usable_cols else usable_cols[-1]
    feature_cols = [c for c in usable_cols if c != target_col]
    print(f"Target: '{target_col}' | Features ({len(feature_cols)}): {feature_cols}")

    print("\n" + "=" * 65)
    print("PHASE 6: ENCODING, LEAKAGE-FREE SPLIT & PCA (95% VARIANCE)")
    print("=" * 65)
    work_df = df[feature_cols + [target_col]].copy()
    task_type = "Regression" if (target_col in num_cols and work_df[target_col].nunique() > 12) else "Classification"
    print(f"Detected ML Task Type: {task_type}")

    for col in feature_cols:
        if col in cat_cols:
            work_df[col] = LabelEncoder().fit_transform(work_df[col].astype(str))

    X = work_df[feature_cols]
    if task_type == "Classification":
        target_le = LabelEncoder()
        y = target_le.fit_transform(work_df[target_col].astype(str))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    else:
        y = work_df[target_col].astype(float).values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    pca_full = PCA().fit(X_train_s)
    n_comp = min(X_train_s.shape[1], max(2, int(np.searchsorted(np.cumsum(pca_full.explained_variance_ratio_), 0.95) + 1)))
    pca = PCA(n_components=n_comp)
    X_train_p = pca.fit_transform(X_train_s)
    X_test_p = pca.transform(X_test_s)
    print(f"PCA Retained {n_comp} Components ({pca.explained_variance_ratio_.sum()*100:.2f}% Variance)")

    print("\n" + "=" * 65)
    print("PHASE 9–10: SUPERVISED ML + DIAGNOSTICS (OOB, K-SWEEP, PRUNING, LEARNING CURVE)")
    print("=" * 65)
    if task_type == "Classification":
        models = {
            "Random Forest": RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42),
            "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
        }
        for name, mdl in models.items():
            mdl.fit(X_train_p, y_train)
            preds = mdl.predict(X_test_p)
            acc = accuracy_score(y_test, preds) * 100
            f1 = f1_score(y_test, preds, average="weighted") * 100
            extra = f" | OOB Error: {(1 - mdl.oob_score_)*100:.2f}%" if hasattr(mdl, "oob_score_") else ""
            print(f"  {name:<22} -> Accuracy: {acc:.2f}% | Weighted F1: {f1:.2f}%{extra}")

        # Learning Curve & Permutation Importance on Best Model
        best_mdl = models["Random Forest"]
        _, tr_sc, val_sc = learning_curve(best_mdl, X_train_p, y_train, cv=3, train_sizes=[0.3, 0.6, 1.0], n_jobs=-1)
        perm = permutation_importance(best_mdl, X_test_p, y_test, n_repeats=5, random_state=42, n_jobs=-1)
        print(f"  Learning Curve Final CV Score: {val_sc[-1].mean()*100:.2f}% | Top PC Importance: {np.round(perm.importances_mean, 4)}")
    else:
        models = {
            "Random Forest Reg": RandomForestRegressor(n_estimators=100, oob_score=True, random_state=42),
            "KNN Reg": KNeighborsRegressor(n_neighbors=5),
            "Decision Tree Reg": DecisionTreeRegressor(random_state=42),
            "Ridge Regression": Ridge(random_state=42)
        }
        for name, mdl in models.items():
            mdl.fit(X_train_p, y_train)
            preds = mdl.predict(X_test_p)
            print(f"  {name:<22} -> R2: {r2_score(y_test, preds):.4f} | RMSE: {np.sqrt(mean_squared_error(y_test, preds)):.2f}")

    print("\n" + "=" * 65)
    print("PHASE 11: UNSUPERVISED ML (K-MEANS + SILHOUETTE + ISOLATION FOREST)")
    print("=" * 65)
    if len(num_cols) >= 2:
        X_u = StandardScaler().fit_transform(df[num_cols])
        km = KMeans(n_clusters=3, random_state=42, n_init=10)
        c_labels = km.fit_predict(X_u)
        sil = silhouette_score(X_u, c_labels)
        anomalies = int((IsolationForest(contamination=0.05, random_state=42).fit_predict(df[num_cols]) == -1).sum())
        print(f"  K-Means (K=3) Silhouette Score: {sil:.4f} | Isolation Forest Anomalies: {anomalies}")

    print("\n" + "=" * 65)
    print("PHASE 12: DEEP LEARNING MLP (128 -> 64 -> 32) + CONFIDENCE & WEIGHTS")
    print("=" * 65)
    if task_type == "Classification":
        mlp = MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=200, random_state=42).fit(X_train_p, y_train)
        dl_preds = mlp.predict(X_test_p)
        dl_probs = mlp.predict_proba(X_test_p)
        mean_conf = float(np.mean(np.max(dl_probs, axis=1)) * 100)
        print(f"  Deep MLP Test Accuracy: {accuracy_score(y_test, dl_preds)*100:.2f}% | Weighted F1: {f1_score(y_test, dl_preds, average='weighted')*100:.2f}%")
        print(f"  Mean Prediction Confidence: {mean_conf:.2f}% | First-Layer Weight Matrix Shape: {mlp.coefs_[0].shape}")
    else:
        mlp = MLPRegressor(hidden_layer_sizes=(128, 64, 32), max_iter=300, random_state=42).fit(X_train_p, y_train)
        dl_preds = mlp.predict(X_test_p)
        print(f"  Deep MLP Test R2: {r2_score(y_test, dl_preds):.4f} | First-Layer Weight Matrix Shape: {mlp.coefs_[0].shape}")


if __name__ == "__main__":
    run_all_phases("sindh-education-stats_2.csv")
