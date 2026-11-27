# ============================================================
# UNIVERSAL DATA ANALYTICS, ML & DEEP LEARNING INTELLIGENCE HUB
# Error-Proof Ingestion + AutoML + Deep Learning + Diagnostics
# ============================================================

from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingRegressor,
    IsolationForest
)
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    precision_recall_curve,
    confusion_matrix,
    roc_curve,
    auc,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    silhouette_score,
    silhouette_samples
)

# ------------------------------------------------------------
# 1. PAGE CONFIG & EXECUTIVE SAAS UI STYLING
# ------------------------------------------------------------

st.set_page_config(
    page_title="Universal AI & Data Intelligence Hub",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "active_section" not in st.session_state:
    st.session_state.active_section = "Executive Overview"

SECTIONS = [
    ("🏠", "Executive Overview"),
    ("🧹", "Cleaning & PCA"),
    ("📊", "Deep EDA & Stats"),
    ("⚙️", "Supervised ML"),
    ("🧬", "Unsupervised ML"),
    ("🧠", "Deep Learning"),
    ("⚡", "Live Predictor")
]

st.markdown("""
<style>
    .stApp { background-color: #EEF3FA !important; color: #0F172A !important; }
    .block-container {
        padding-top: 1.0rem !important; padding-bottom: 1.2rem !important;
        padding-left: 1.5rem !important; padding-right: 1.5rem !important;
        max-width: 100% !important;
    }
    header[data-testid="stHeader"] { background: transparent !important; height: 1.8rem !important; }

    /* DARK NAVY SIDEBAR */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #071A4B 0%, #0A2361 70%, #06153D 100%) !important;
        border-right: none !important;
    }
    section[data-testid="stSidebar"] > div:first-child { padding-top: 0.9rem !important; }
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stCaption { color: #CBD5E1 !important; }

    section[data-testid="stSidebar"] div[role="radiogroup"] { gap: 4px !important; }
    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        background-color: transparent !important; padding: 8px 14px !important;
        border-radius: 10px !important; margin: 0 !important; border: 1px solid transparent !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background-color: rgba(255, 255, 255, 0.08) !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"],
    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
        background: linear-gradient(90deg, #1677FF 0%, #0E5FD8 100%) !important;
        box-shadow: 0 4px 12px rgba(22, 119, 255, 0.4) !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label p {
        color: #FFFFFF !important; font-weight: 600 !important; font-size: 0.88rem !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child { display: none !important; }

    section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
        background-color: #0B2563 !important; border: 1px solid #264896 !important; border-radius: 8px !important;
    }
    section[data-testid="stSidebar"] div[data-baseweb="select"] span,
    section[data-testid="stSidebar"] div[data-baseweb="select"] div { color: #FFFFFF !important; }
    section[data-testid="stSidebar"] svg { fill: #FFFFFF !important; }

    section[data-testid="stSidebar"] section[data-testid="stFileUploaderDropzone"] {
        background-color: #0B2563 !important; border: 1px dashed #38BDF8 !important;
        border-radius: 10px !important; padding: 8px !important;
    }
    section[data-testid="stSidebar"] section[data-testid="stFileUploaderDropzone"] * {
        color: #E2E8F0 !important; font-size: 0.78rem !important;
    }

    div[data-testid="stAppViewContainer"] div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important; border: 1px solid #CBD5E1 !important; border-radius: 8px !important;
    }
    div[data-testid="stAppViewContainer"] div[data-baseweb="select"] span,
    div[data-testid="stAppViewContainer"] div[data-baseweb="select"] div,
    div[data-testid="stAppViewContainer"] input {
        color: #0F172A !important; background-color: #FFFFFF !important;
    }

    .sidebar-health {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.16) 0%, rgba(6, 78, 59, 0.35) 100%);
        border: 1px solid rgba(52, 211, 153, 0.45); border-radius: 12px;
        padding: 11px 14px; margin-top: 10px;
    }
    .health-top { display: flex; align-items: center; gap: 10px; }
    .health-check {
        width: 32px; height: 32px; border-radius: 50%; background-color: #10B981;
        color: #FFFFFF; display: flex; align-items: center; justify-content: center;
        font-weight: 900; font-size: 1.0rem; flex-shrink: 0;
    }
    .health-sub { color: #6EE7B7 !important; font-size: 0.72rem; font-weight: 700; }
    .health-main { color: #FFFFFF !important; font-size: 1.0rem; font-weight: 800; }
    .health-foot {
        color: #A7F3D0 !important; font-size: 0.72rem; margin-top: 6px;
        padding-top: 5px; border-top: 1px solid rgba(52, 211, 153, 0.25);
    }

    .title-block { border-left: 5px solid #14B8A6; padding-left: 12px; }
    .main-hub-title { font-size: 1.7rem; font-weight: 800; color: #0B2259; line-height: 1.15; margin: 0; }
    .main-hub-sub { font-size: 0.86rem; color: #475569; margin-top: 2px; }
    .updated-pill {
        background-color: #FFFFFF; border: 1px solid #CBD5E1; border-radius: 10px;
        padding: 7px 12px; font-size: 0.78rem; font-weight: 600; color: #1E293B;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04); text-align: center;
    }
    .avatar-pill {
        background-color: #0B2259; color: #FFFFFF; font-weight: 800; font-size: 0.78rem;
        width: 36px; height: 36px; border-radius: 50%;
        display: inline-flex; align-items: center; justify-content: center;
    }

    .grad-kpi {
        border-radius: 14px; padding: 15px 16px; color: #FFFFFF !important;
        position: relative; overflow: hidden; box-shadow: 0 6px 16px rgba(15, 23, 42, 0.10);
        min-height: 114px; display: flex; align-items: center; gap: 12px;
    }
    .kpi-blue { background: linear-gradient(135deg, #1366D6 0%, #0C4A9E 100%); }
    .kpi-teal { background: linear-gradient(135deg, #00A896 0%, #028090 100%); }
    .kpi-purple { background: linear-gradient(135deg, #7C5dfa 0%, #5B3CC4 100%); }
    .kpi-green { background: linear-gradient(135deg, #16A34A 0%, #0D9488 100%); }

    .kpi-icon-box {
        width: 46px; height: 46px; border-radius: 12px; background: rgba(255, 255, 255, 0.18);
        display: flex; align-items: center; justify-content: center; font-size: 1.45rem; flex-shrink: 0;
    }
    .kpi-content { z-index: 2; }
    .kpi-lbl { font-size: 0.84rem; font-weight: 600; color: rgba(255, 255, 255, 0.92) !important; }
    .kpi-val {
        font-size: clamp(1.3rem, 1.8vw, 1.85rem); font-weight: 900;
        color: #FFFFFF !important; line-height: 1.15; margin: 3px 0; white-space: nowrap;
    }
    .kpi-trend { font-size: 0.74rem; font-weight: 600; color: #D1FAE5 !important; }
    .kpi-watermark { position: absolute; right: 10px; bottom: -6px; font-size: 3.2rem; opacity: 0.14; pointer-events: none; }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #FFFFFF !important; border: 1px solid #E2E8F0 !important;
        border-radius: 14px !important; padding: 14px 16px !important;
        box-shadow: 0 3px 10px rgba(15, 23, 42, 0.04) !important;
    }
    .bento-title {
        font-size: 1.0rem; font-weight: 800; color: #0B2259 !important;
        margin-bottom: 4px; display: flex; align-items: center; gap: 7px;
    }

    div[data-testid="stHorizontalBlock"] button[kind="secondary"] {
        background-color: #FFFFFF !important; color: #0B2259 !important;
        border: 1px solid #CBD5E1 !important; border-radius: 10px !important;
        font-weight: 700 !important; font-size: 0.81rem !important; padding: 0.40rem 0.55rem !important;
    }
    div[data-testid="stHorizontalBlock"] button[kind="primary"] {
        background: linear-gradient(90deg, #1677FF 0%, #0E5FD8 100%) !important;
        color: #FFFFFF !important; border: none !important; border-radius: 10px !important;
        font-weight: 700 !important; font-size: 0.81rem !important; padding: 0.40rem 0.55rem !important;
    }

    .footer-banner {
        background: linear-gradient(90deg, #071A4B 0%, #0E3A8A 65%, #0D9488 100%);
        border-radius: 14px; padding: 13px 20px; margin-top: 14px;
        display: flex; align-items: center; justify-content: space-between;
        color: #FFFFFF !important; box-shadow: 0 4px 14px rgba(7, 26, 75, 0.18);
    }
    .footer-tagline { font-size: 0.98rem; font-weight: 700; color: #FFFFFF !important; }
    .footer-skyline { font-size: 0.82rem; color: #BAE6FD !important; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

PLOTLY_CONFIG = {"displayModeBar": False}


def style_chart(fig, height=280):
    fig.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color="#0F172A", size=11, family="Inter, sans-serif"),
        legend=dict(font=dict(color="#0F172A", size=11), orientation="h", y=1.14, x=0.5, xanchor="center"),
        margin=dict(t=24, l=8, r=8, b=8),
        height=height
    )
    fig.update_xaxes(showgrid=False, zeroline=False, tickfont=dict(color="#0F172A", size=11), title_font=dict(color="#0F172A", size=11))
    fig.update_yaxes(showgrid=True, gridcolor="#E2E8F0", zeroline=False, tickfont=dict(color="#0F172A", size=11), title_font=dict(color="#0F172A", size=11))
    return fig


# ------------------------------------------------------------
# 2. ERROR-PROOF DATA INGESTION & AUTO-CLEANING PIPELINE
# ------------------------------------------------------------

@st.cache_data
def load_raw_dataset(uploaded_bytes, file_name):
    if uploaded_bytes is not None:
        if file_name.lower().endswith((".xlsx", ".xls")):
            return pd.read_excel(uploaded_bytes), file_name
        return pd.read_csv(uploaded_bytes), file_name

    base_dir = Path(__file__).resolve().parent
    for pattern in ["sindh-education-stats_2.csv", "sindh-education-stats.csv", "P2_cleaned_data.csv", "*.csv"]:
        matches = list(base_dir.rglob(pattern))
        if matches:
            return pd.read_csv(matches[0]), matches[0].name

    return None, None


@st.cache_data
def run_universal_cleaning(raw_df):
    df = raw_df.copy()
    initial_rows = len(df)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(axis=1, how="all")

    dup_count = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)

    audit_records = []
    num_cols, cat_cols, id_cols = [], [], []

    for col in df.columns:
        if df[col].dtype == "object":
            cleaned_str = df[col].astype(str).str.replace(",", "", regex=False).str.strip()
            numeric_try = pd.to_numeric(cleaned_str, errors="coerce")
            if numeric_try.notna().mean() > 0.80:
                df[col] = numeric_try

        missing_cnt = int(df[col].isna().sum())

        if pd.api.types.is_numeric_dtype(df[col]):
            med_val = df[col].median()
            df[col] = df[col].fillna(med_val if pd.notna(med_val) else 0)

            if df[col].nunique() == len(df) and ("id" in col.lower() or "index" in col.lower()):
                id_cols.append(col)
                audit_records.append({"Column": col, "Detected Type": "ID (Excluded)", "Missing Imputed": missing_cnt, "Outliers Capped": 0})
                continue

            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            outliers_capped = 0
            if iqr > 0:
                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                outliers_capped = int(((df[col] < lower) | (df[col] > upper)).sum())
                df[col] = np.clip(df[col], lower, upper)

            num_cols.append(col)
            audit_records.append({"Column": col, "Detected Type": "Numeric", "Missing Imputed": missing_cnt, "Outliers Capped": outliers_capped})
        else:
            df[col] = df[col].fillna("Unknown").astype(str).str.strip()
            df[col] = df[col].replace({"Matira": "Matiari", "Tando Allah Yar": "Tando Allahyar"})

            unique_ratio = df[col].nunique() / max(len(df), 1)
            if unique_ratio > 0.85 and len(df) > 50:
                id_cols.append(col)
                audit_records.append({"Column": col, "Detected Type": "High-Cardinality ID", "Missing Imputed": missing_cnt, "Outliers Capped": 0})
            else:
                cat_cols.append(col)
                audit_records.append({"Column": col, "Detected Type": "Categorical", "Missing Imputed": missing_cnt, "Outliers Capped": 0})

    audit_df = pd.DataFrame(audit_records)
    summary = {
        "initial_rows": initial_rows,
        "final_rows": len(df),
        "duplicates_removed": dup_count,
        "total_missing_fixed": int(audit_df["Missing Imputed"].sum()),
        "total_outliers_capped": int(audit_df["Outliers Capped"].sum()),
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "id_cols": id_cols
    }
    return df, audit_df, summary


# ------------------------------------------------------------
# 3. UNIVERSAL AUTOML, DEEP LEARNING & CACHED DIAGNOSTICS
# ------------------------------------------------------------

@st.cache_resource
def train_universal_ai(df, target_col, feature_cols, num_cols, cat_cols):
    work_df = df[feature_cols + [target_col]].copy()
    if len(work_df) > 4000:
        work_df = work_df.sample(4000, random_state=42).reset_index(drop=True)

    is_numeric_target = target_col in num_cols
    n_unique_target = work_df[target_col].nunique()
    task_type = "Regression" if (is_numeric_target and n_unique_target > 12) else "Classification"

    encoders = {}
    for col in feature_cols:
        if col in cat_cols:
            le = LabelEncoder()
            work_df[col] = le.fit_transform(work_df[col].astype(str))
            encoders[col] = le

    X = work_df[feature_cols]
    y_raw = work_df[target_col]

    if task_type == "Classification":
        target_le = LabelEncoder()
        y = target_le.fit_transform(y_raw.astype(str))
        encoders[target_col] = target_le
        class_names = target_le.classes_

        class_counts = pd.Series(y).value_counts()
        valid_classes = class_counts[class_counts >= 3].index
        mask = pd.Series(y).isin(valid_classes)
        X, y = X[mask].reset_index(drop=True), y[mask]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    else:
        y = y_raw.astype(float).values
        class_names = None
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    pca_full = PCA().fit(X_train_s)
    n_comp = min(X_train_s.shape[1], max(2, int(np.searchsorted(np.cumsum(pca_full.explained_variance_ratio_), 0.95) + 1)))
    pca_opt = PCA(n_components=n_comp)
    X_train_p = pca_opt.fit_transform(X_train_s)
    X_test_p = pca_opt.transform(X_test_s)

    leaderboard_rows, fitted_models = [], {}

    if task_type == "Classification":
        models = {
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=min(5, len(X_train_p))),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
        }
        for name, mdl in models.items():
            mdl.fit(X_train_p, y_train)
            preds = mdl.predict(X_test_p)
            leaderboard_rows.append({
                "Model": name,
                "Accuracy (%)": round(accuracy_score(y_test, preds) * 100, 2),
                "Weighted F1 (%)": round(f1_score(y_test, preds, average="weighted") * 100, 2)
            })
            fitted_models[name] = mdl

        leaderboard = pd.DataFrame(leaderboard_rows).sort_values("Accuracy (%)", ascending=False)

        oob_trees = [15, 30, 50, 75, 100, 125]
        oob_errors = []
        for nt in oob_trees:
            rf_tmp = RandomForestClassifier(n_estimators=nt, oob_score=True, random_state=42, n_jobs=-1).fit(X_train_p, y_train)
            oob_errors.append(round((1 - rf_tmp.oob_score_) * 100, 2))

        k_vals = [k for k in range(1, 20, 2) if k < len(X_train_p)]
        knn_train_acc, knn_val_acc = [], []
        for k in k_vals:
            kn = KNeighborsClassifier(n_neighbors=k).fit(X_train_p, y_train)
            knn_train_acc.append(round(accuracy_score(y_train, kn.predict(X_train_p)) * 100, 2))
            knn_val_acc.append(round(accuracy_score(y_test, kn.predict(X_test_p)) * 100, 2))

        path = DecisionTreeClassifier(random_state=42).cost_complexity_pruning_path(X_train_p, y_train)
        alphas = np.unique(path.ccp_alphas)[::max(1, len(path.ccp_alphas) // 10)][:10]
        dt_val_scores, dt_depths = [], []
        for a in alphas:
            dt_tmp = DecisionTreeClassifier(random_state=42, ccp_alpha=a).fit(X_train_p, y_train)
            dt_val_scores.append(round(accuracy_score(y_test, dt_tmp.predict(X_test_p)) * 100, 2))
            dt_depths.append(dt_tmp.get_depth())

        lr_raw = LogisticRegression(max_iter=1000, random_state=42).fit(X_train_s, y_train)
        coef_magnitudes = np.mean(np.abs(lr_raw.coef_), axis=0)
        coef_df = pd.DataFrame({"Feature": feature_cols, "Mean |Weight|": coef_magnitudes}).sort_values("Mean |Weight|")

        dl_model = MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=1, warm_start=True, random_state=42)
        losses, val_scores = [], []
        unique_cls = np.unique(y_train)
        for _ in range(50):
            dl_model.partial_fit(X_train_p, y_train, classes=unique_cls)
            losses.append(dl_model.loss_)
            val_scores.append(accuracy_score(y_test, dl_model.predict(X_test_p)) * 100)

        dl_preds = dl_model.predict(X_test_p)
        dl_primary = round(accuracy_score(y_test, dl_preds) * 100, 2)

    else:
        models = {
            "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
            "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42),
            "KNN Regressor": KNeighborsRegressor(n_neighbors=min(5, len(X_train_p))),
            "Ridge Regression": Ridge(random_state=42)
        }
        for name, mdl in models.items():
            mdl.fit(X_train_p, y_train)
            preds = mdl.predict(X_test_p)
            leaderboard_rows.append({
                "Model": name,
                "R² Score": round(r2_score(y_test, preds), 4),
                "RMSE": round(float(np.sqrt(mean_squared_error(y_test, preds))), 2),
                "MAE": round(float(mean_absolute_error(y_test, preds)), 2)
            })
            fitted_models[name] = mdl

        leaderboard = pd.DataFrame(leaderboard_rows).sort_values("R² Score", ascending=False)

        oob_trees = [15, 30, 50, 75, 100, 125]
        oob_errors = []
        for nt in oob_trees:
            rf_tmp = RandomForestRegressor(n_estimators=nt, oob_score=True, random_state=42, n_jobs=-1).fit(X_train_p, y_train)
            oob_errors.append(round((1 - rf_tmp.oob_score_) * 100, 2))

        k_vals = [k for k in range(1, 20, 2) if k < len(X_train_p)]
        knn_train_acc, knn_val_acc = [], []
        for k in k_vals:
            kn = KNeighborsRegressor(n_neighbors=k).fit(X_train_p, y_train)
            knn_train_acc.append(round(r2_score(y_train, kn.predict(X_train_p)) * 100, 2))
            knn_val_acc.append(round(r2_score(y_test, kn.predict(X_test_p)) * 100, 2))

        path = DecisionTreeRegressor(random_state=42).cost_complexity_pruning_path(X_train_p, y_train)
        alphas = np.unique(path.ccp_alphas)[::max(1, len(path.ccp_alphas) // 10)][:10]
        dt_val_scores, dt_depths = [], []
        for a in alphas:
            dt_tmp = DecisionTreeRegressor(random_state=42, ccp_alpha=a).fit(X_train_p, y_train)
            dt_val_scores.append(round(r2_score(y_test, dt_tmp.predict(X_test_p)) * 100, 2))
            dt_depths.append(dt_tmp.get_depth())

        ridge_raw = Ridge(random_state=42).fit(X_train_s, y_train)
        coef_df = pd.DataFrame({"Feature": feature_cols, "Mean |Weight|": np.abs(ridge_raw.coef_)}).sort_values("Mean |Weight|")

        dl_model = MLPRegressor(hidden_layer_sizes=(128, 64, 32), max_iter=1, warm_start=True, random_state=42)
        losses, val_scores = [], []
        for _ in range(60):
            dl_model.partial_fit(X_train_p, y_train)
            losses.append(dl_model.loss_)
            val_scores.append(r2_score(y_test, dl_model.predict(X_test_p)) * 100)

        dl_preds = dl_model.predict(X_test_p)
        dl_primary = round(r2_score(y_test, dl_preds), 4)

    return {
        "task_type": task_type,
        "encoders": encoders,
        "scaler": scaler,
        "pca_full": pca_full,
        "pca_opt": pca_opt,
        "X_train_p": X_train_p,
        "y_train": y_train,
        "X_test_p": X_test_p,
        "y_test": y_test,
        "class_names": class_names,
        "sup_models": fitted_models,
        "leaderboard": leaderboard,
        "diag": {
            "oob_trees": oob_trees,
            "oob_errors": oob_errors,
            "k_vals": k_vals,
            "knn_train": knn_train_acc,
            "knn_val": knn_val_acc,
            "alphas": [round(float(a), 5) for a in alphas],
            "dt_scores": dt_val_scores,
            "dt_depths": dt_depths,
            "coef_df": coef_df
        },
        "dl": {
            "model": dl_model,
            "primary": dl_primary,
            "losses": losses,
            "val_scores": val_scores
        }
    }


@st.cache_data
def compute_model_curves(_model, X_tr, y_tr, X_te, y_te, model_key):
    train_sizes, train_sc, val_sc = learning_curve(
        _model, X_tr, y_tr, cv=3, train_sizes=np.linspace(0.2, 1.0, 5), n_jobs=-1
    )
    perm = permutation_importance(_model, X_te, y_te, n_repeats=5, random_state=42, n_jobs=-1)
    pc_names = [f"PC{i+1}" for i in range(X_te.shape[1])]
    perm_df = pd.DataFrame({
        "Component": pc_names,
        "Importance": perm.importances_mean
    }).sort_values("Importance")
    return train_sizes, train_sc.mean(axis=1) * 100, val_sc.mean(axis=1) * 100, perm_df


# ------------------------------------------------------------
# 4. DARK NAVY SIDEBAR
# ------------------------------------------------------------

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding-bottom: 6px;">
        <svg width="56" height="48" viewBox="0 0 64 56" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M32 4L4 17L32 30L60 17L32 4Z" fill="#38BDF8"/>
            <path d="M12 24V38C12 43 21 48 32 48C43 48 52 43 52 38V24L32 33L12 24Z" fill="#1677FF"/>
            <circle cx="32" cy="17" r="3.5" fill="#FACC15"/>
        </svg>
        <div style="color:#FFFFFF; font-size:1.12rem; font-weight:800; line-height:1.2; margin-top:2px;">
            Universal Data & AI<br>Intelligence Hub
        </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📂 Upload Any CSV / Excel", type=["csv", "xlsx", "xls"])
    raw_df, active_filename = load_raw_dataset(uploaded_file, uploaded_file.name if uploaded_file else None)

    if raw_df is None:
        st.info("👆 Upload any CSV or Excel file above to start the AI & Analytics Hub.")
        st.stop()

    df, audit_df, clean_summary = run_universal_cleaning(raw_df)
    num_cols = clean_summary["num_cols"]
    cat_cols = clean_summary["cat_cols"]

    section_labels = [f"{icon}  {name}" for icon, name in SECTIONS]
    current_idx = [name for _, name in SECTIONS].index(st.session_state.active_section)
    chosen_nav = st.radio("Navigation", section_labels, index=current_idx, label_visibility="collapsed")
    st.session_state.active_section = chosen_nav.split("  ", 1)[1]

    st.markdown("<hr style='border-color: rgba(255,255,255,0.14); margin: 8px 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='color:#FFFFFF; font-weight:700; font-size:0.86rem; margin-bottom:4px;'>⚙️ Dynamic Pipeline Config</div>", unsafe_allow_html=True)

    all_usable_cols = [c for c in df.columns if c not in clean_summary["id_cols"]]
    default_target = "Category" if "Category" in all_usable_cols else all_usable_cols[-1]
    target_col = st.selectbox("🎯 Target Column (Predict)", all_usable_cols, index=all_usable_cols.index(default_target))

    default_metric = "Number" if "Number" in num_cols else (num_cols[0] if num_cols else None)
    metric_col = st.selectbox("📈 Primary Numeric Metric", num_cols, index=num_cols.index(default_metric)) if num_cols else None

    default_group = "District" if "District" in cat_cols else (cat_cols[0] if cat_cols else None)
    group_col = st.selectbox("🏷️ Primary Group / Category", cat_cols, index=cat_cols.index(default_group)) if cat_cols else None

    filtered_df = df.copy()
    if group_col:
        unique_groups = sorted(df[group_col].unique().tolist())
        selected_group_val = st.selectbox(f"Filter by {group_col}", [f"All ({len(unique_groups)})"] + unique_groups)
        if not selected_group_val.startswith("All ("):
            filtered_df = filtered_df[filtered_df[group_col] == selected_group_val]

    st.markdown(f"""
    <div class="sidebar-health">
        <div class="health-top">
            <div class="health-check">✓</div>
            <div>
                <div class="health-sub">Dataset Health • Auto-Cleaned</div>
                <div class="health-main">{len(df):,} Clean Rows</div>
            </div>
        </div>
        <div class="health-foot">✓ {clean_summary['total_missing_fixed']} Nulls Fixed • {clean_summary['total_outliers_capped']} Outliers Capped</div>
    </div>
    """, unsafe_allow_html=True)

feature_cols = [c for c in all_usable_cols if c != target_col]
if len(feature_cols) < 2:
    st.error("Dataset needs at least 2 feature columns besides the target column.")
    st.stop()

ai = train_universal_ai(df, target_col, feature_cols, num_cols, cat_cols)

# ------------------------------------------------------------
# 5. TOP COMMAND HEADER BAR & 4 KPI CARDS
# ------------------------------------------------------------

hdr_left, hdr_right = st.columns([1.45, 1])
with hdr_left:
    st.markdown(f"""
    <div class="title-block">
        <div class="main-hub-title">Universal Data & AI Intelligence Hub</div>
        <div class="main-hub-sub">Active Dataset: <b>{active_filename}</b> • Task Mode: <b>{ai['task_type']}</b> (Target: <code>{target_col}</code>)</div>
    </div>
    """, unsafe_allow_html=True)

with hdr_right:
    r_c1, r_c2, r_c3 = st.columns([1.5, 1.0, 0.35])
    with r_c1:
        now_str = datetime.now().strftime("%b %d, %Y • %I:%M %p")
        st.markdown(f'<div class="updated-pill">🔄 Active: {now_str}</div>', unsafe_allow_html=True)
    with r_c2:
        csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇ Download Cleaned CSV",
            data=csv_bytes,
            file_name=f"cleaned_{Path(active_filename).stem}.csv",
            mime="text/csv",
            width="stretch"
        )
    with r_c3:
        st.markdown('<div class="avatar-pill">AI</div>', unsafe_allow_html=True)

total_records = len(filtered_df)
metric_total = filtered_df[metric_col].sum() if metric_col else total_records
metric_avg = filtered_df[metric_col].mean() if metric_col else 0
best_sup_score = ai["leaderboard"].iloc[0, 1]
best_sup_label = "Accuracy" if ai["task_type"] == "Classification" else "R² Score"
dl_score_fmt = f"{ai['dl']['primary']}%" if ai["task_type"] == "Classification" else f"{ai['dl']['primary']}"
sup_score_fmt = f"{best_sup_score}%" if ai["task_type"] == "Classification" else f"{best_sup_score}"

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f"""
    <div class="grad-kpi kpi-blue">
        <div class="kpi-icon-box">📁</div>
        <div class="kpi-content">
            <div class="kpi-lbl">Active Clean Records</div>
            <div class="kpi-val">{total_records:,}</div>
            <div class="kpi-trend">✓ {len(feature_cols)} Features • {ai['pca_opt'].n_components_} PCs</div>
        </div>
        <div class="kpi-watermark">📊</div>
    </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="grad-kpi kpi-teal">
        <div class="kpi-icon-box">📈</div>
        <div class="kpi-content">
            <div class="kpi-lbl">Total {metric_col if metric_col else 'Rows'}</div>
            <div class="kpi-val">{metric_total:,.0f}</div>
            <div class="kpi-trend">● Mean: {metric_avg:,.1f} per record</div>
        </div>
        <div class="kpi-watermark">💹</div>
    </div>
    """, unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class="grad-kpi kpi-purple">
        <div class="kpi-icon-box">⚙️</div>
        <div class="kpi-content">
            <div class="kpi-lbl">Best Supervised ML ({best_sup_label})</div>
            <div class="kpi-val">{sup_score_fmt}</div>
            <div class="kpi-trend">🏆 {ai['leaderboard'].iloc[0]['Model']}</div>
        </div>
        <div class="kpi-watermark">🤖</div>
    </div>
    """, unsafe_allow_html=True)

with k4:
    st.markdown(f"""
    <div class="grad-kpi kpi-green">
        <div class="kpi-icon-box">🧠</div>
        <div class="kpi-content">
            <div class="kpi-lbl">Deep Neural Network ({best_sup_label})</div>
            <div class="kpi-val">{dl_score_fmt}</div>
            <div class="kpi-trend">⚡ MLP (128 → 64 → 32)</div>
        </div>
        <div class="kpi-watermark">🧬</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

pill_cols = st.columns(len(SECTIONS))
for idx, (icon, sec_name) in enumerate(SECTIONS):
    btn_type = "primary" if st.session_state.active_section == sec_name else "secondary"
    if pill_cols[idx].button(f"{icon} {sec_name}", key=f"top_pill_{sec_name}", type=btn_type, width="stretch"):
        st.session_state.active_section = sec_name
        st.rerun()

st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)

# ------------------------------------------------------------
# 6. DYNAMIC SECTION VIEWS
# ------------------------------------------------------------

active = st.session_state.active_section

# ==================== VIEW 1: EXECUTIVE OVERVIEW ====================
if active == "Executive Overview":
    r1_left, r1_right = st.columns([1.05, 1])

    with r1_left:
        with st.container(border=True):
            st.markdown('<div class="bento-title">📊 Hierarchical Breakdown (Sunburst)</div>', unsafe_allow_html=True)
            if len(cat_cols) >= 2 and metric_col:
                c_a, c_b = cat_cols[0], cat_cols[1]
                sun_df = filtered_df.groupby([c_a, c_b])[metric_col].sum().reset_index()
                fig_sun = px.sunburst(
                    sun_df, path=[c_a, c_b], values=metric_col,
                    color=c_a, color_discrete_sequence=["#1366D6", "#00A896", "#7C5dfa", "#16A34A", "#F59E0B"]
                )
                fig_sun.update_traces(textinfo="label+percent parent", insidetextfont=dict(color="#FFFFFF", size=11))
                fig_sun.update_layout(paper_bgcolor="#FFFFFF", margin=dict(t=6, l=6, r=6, b=6), height=255)
                st.plotly_chart(fig_sun, width="stretch", config=PLOTLY_CONFIG)
            elif len(cat_cols) == 1 and metric_col:
                fig_pie = px.pie(filtered_df, names=cat_cols[0], values=metric_col, hole=0.45)
                st.plotly_chart(style_chart(fig_pie, 255), width="stretch", config=PLOTLY_CONFIG)
            else:
                fig_hist = px.histogram(filtered_df, x=metric_col, color_discrete_sequence=["#1366D6"])
                st.plotly_chart(style_chart(fig_hist, 255), width="stretch", config=PLOTLY_CONFIG)

    with r1_right:
        with st.container(border=True):
            st.markdown(f'<div class="bento-title">🏛️ {metric_col} Comparison Across Categories</div>', unsafe_allow_html=True)
            if len(cat_cols) >= 2 and metric_col:
                grp_col1 = cat_cols[1] if len(cat_cols) > 1 else cat_cols[0]
                grp_col2 = cat_cols[0] if cat_cols[0] != grp_col1 else None
                bar_df = filtered_df.groupby([grp_col1] + ([grp_col2] if grp_col2 else []))[metric_col].sum().reset_index()
                fig_bar = px.bar(
                    bar_df, x=grp_col1, y=metric_col, color=grp_col2, barmode="group",
                    color_discrete_sequence=["#1366D6", "#00A896", "#7C5dfa", "#F472B6"]
                )
                fig_bar.update_layout(yaxis_title="", xaxis_title="")
                st.plotly_chart(style_chart(fig_bar, 255), width="stretch", config=PLOTLY_CONFIG)
            elif metric_col:
                fig_box = px.box(filtered_df, y=metric_col, color_discrete_sequence=["#00A896"])
                st.plotly_chart(style_chart(fig_box, 255), width="stretch", config=PLOTLY_CONFIG)

    r2_c1, r2_c2, r2_c3 = st.columns([1.05, 0.9, 1.15])

    with r2_c1:
        with st.container(border=True):
            st.markdown('<div class="bento-title">📦 Numeric Spread (IQR Capped Boxplot)</div>', unsafe_allow_html=True)
            if metric_col:
                x_box = cat_cols[1] if len(cat_cols) > 1 else (cat_cols[0] if cat_cols else None)
                fig_bx = px.box(
                    filtered_df, x=x_box, y=metric_col, color=x_box,
                    color_discrete_sequence=["#1366D6", "#00A896", "#7C5dfa", "#F472B6"]
                )
                fig_bx.update_layout(showlegend=False, yaxis_title="", xaxis_title="")
                st.plotly_chart(style_chart(fig_bx, 235), width="stretch", config=PLOTLY_CONFIG)

    with r2_c2:
        with st.container(border=True):
            st.markdown(f'<div class="bento-title">🍩 Target Distribution ({target_col})</div>', unsafe_allow_html=True)
            if ai["task_type"] == "Classification":
                t_counts = filtered_df[target_col].value_counts().head(6).reset_index()
                t_counts.columns = [target_col, "Count"]
                fig_donut = px.pie(
                    t_counts, names=target_col, values="Count", hole=0.60,
                    color_discrete_sequence=["#1366D6", "#00A896", "#7C5dfa", "#16A34A", "#F59E0B", "#E11D48"]
                )
                fig_donut.update_traces(textinfo="percent", textfont=dict(color="#FFFFFF", size=10))
                fig_donut.update_layout(
                    paper_bgcolor="#FFFFFF", margin=dict(t=8, l=4, r=4, b=8), height=235,
                    legend=dict(font=dict(color="#0F172A", size=10)),
                    annotations=[dict(text=f"<b>{len(filtered_df):,}</b><br>Rows", x=0.5, y=0.5, font_size=11, showarrow=False, font_color="#0B2259")]
                )
                st.plotly_chart(fig_donut, width="stretch", config=PLOTLY_CONFIG)
            else:
                fig_t_hist = px.histogram(filtered_df, x=target_col, color_discrete_sequence=["#7C5dfa"])
                st.plotly_chart(style_chart(fig_t_hist, 235), width="stretch", config=PLOTLY_CONFIG)

    with r2_c3:
        with st.container(border=True):
            st.markdown(f'<div class="bento-title">🏆 Top 15 {group_col if group_col else "Records"} Ranking</div>', unsafe_allow_html=True)
            if group_col and metric_col:
                rank_df = df.groupby(group_col)[metric_col].sum().sort_values(ascending=False).head(15).reset_index()
                fig_line = px.area(rank_df, x=group_col, y=metric_col, markers=True, color_discrete_sequence=["#1366D6"])
                fig_line.update_traces(line=dict(width=2.5), fillcolor="rgba(19, 102, 214, 0.12)")
                fig_line.update_layout(yaxis_title="", xaxis_title="")
                fig_line.update_xaxes(tickangle=-35, tickfont=dict(size=9))
                st.plotly_chart(style_chart(fig_line, 235), width="stretch", config=PLOTLY_CONFIG)
            elif len(num_cols) >= 2:
                fig_sc = px.scatter(filtered_df, x=num_cols[0], y=num_cols[1], color_discrete_sequence=["#1366D6"])
                st.plotly_chart(style_chart(fig_sc, 235), width="stretch", config=PLOTLY_CONFIG)

# ==================== VIEW 2: CLEANING & PCA ====================
elif active == "Cleaning & PCA":
    c_left, c_right = st.columns([1.15, 1])

    with c_left:
        with st.container(border=True):
            st.markdown('<div class="bento-title">🧹 Automated Data Cleaning & Schema Audit Log</div>', unsafe_allow_html=True)
            st.dataframe(audit_df, width="stretch", hide_index=True, height=330)

    with c_right:
        with st.container(border=True):
            st.markdown('<div class="bento-title">🧬 Dynamic PCA Scree & Cumulative Variance</div>', unsafe_allow_html=True)
            exp_var = ai["pca_full"].explained_variance_ratio_ * 100
            cum_var = np.cumsum(exp_var)
            pcs = [f"PC{i+1}" for i in range(len(exp_var))]
            fig_pca = go.Figure()
            fig_pca.add_trace(go.Bar(x=pcs, y=exp_var, name="Individual (%)", marker_color="#00A896"))
            fig_pca.add_trace(go.Scatter(x=pcs, y=cum_var, name="Cumulative (%)", mode="lines+markers", line=dict(color="#1366D6", width=3)))
            fig_pca.add_hline(y=95, line_dash="dash", line_color="#E11D48")
            st.plotly_chart(style_chart(fig_pca, 330), width="stretch", config=PLOTLY_CONFIG)

    with st.container(border=True):
        st.markdown('<div class="bento-title">📋 Cleaned Dataset Preview (Top 20 Rows)</div>', unsafe_allow_html=True)
        st.dataframe(filtered_df.head(20), width="stretch", hide_index=True)

# ==================== VIEW 3: DEEP EDA & STATS ====================
elif active == "Deep EDA & Stats":
    e1, e2 = st.columns(2)
    with e1:
        with st.container(border=True):
            st.markdown('<div class="bento-title">🔥 Dynamic Feature Correlation Matrix</div>', unsafe_allow_html=True)
            if len(num_cols) >= 2:
                corr_mat = df[num_cols].corr().round(3)
                fig_corr = px.imshow(corr_mat, text_auto=True, color_continuous_scale="Blues", zmin=-1, zmax=1)
                fig_corr.update_coloraxes(colorbar_tickfont=dict(color="#0F172A", size=11))
                st.plotly_chart(style_chart(fig_corr, 370), width="stretch", config=PLOTLY_CONFIG)
            else:
                st.info("At least 2 numeric columns are needed for a correlation matrix.")

    with e2:
        with st.container(border=True):
            st.markdown('<div class="bento-title">📐 Statistical Summary (Mean, Std, IQR)</div>', unsafe_allow_html=True)
            if num_cols:
                st.dataframe(filtered_df[num_cols].describe().round(2).T, width="stretch", height=370)

# ==================== VIEW 4: SUPERVISED ML ====================
elif active == "Supervised ML":
    with st.container(border=True):
        st.markdown(f'<div class="bento-title">⚙️ Supervised {ai["task_type"]} Leaderboard ({ai["pca_opt"].n_components_} PCA Components)</div>', unsafe_allow_html=True)
        st.dataframe(ai["leaderboard"], width="stretch", hide_index=True)

    sel_sup = st.selectbox("Select Supervised Model to Inspect Diagnostics:", ai["leaderboard"]["Model"].tolist())
    active_sup = ai["sup_models"][sel_sup]
    tr_sizes, tr_sc, val_sc, perm_df = compute_model_curves(
        active_sup, ai["X_train_p"], ai["y_train"], ai["X_test_p"], ai["y_test"], f"{active_filename}_{target_col}_{sel_sup}"
    )

    s1, s2 = st.columns(2)
    if ai["task_type"] == "Classification":
        class_names = ai["class_names"]
        y_bin = label_binarize(ai["y_test"], classes=list(range(len(class_names))))
        if len(class_names) == 2:
            y_bin = np.hstack([1 - y_bin, y_bin])
        probs = active_sup.predict_proba(ai["X_test_p"])
        palette = ["#1366D6", "#00A896", "#7C5dfa", "#E11D48", "#F59E0B", "#16A34A"]

        with s1:
            with st.container(border=True):
                st.markdown(f'<div class="bento-title">📈 Multi-Class ROC Curve — {sel_sup}</div>', unsafe_allow_html=True)
                fig_roc = go.Figure()
                for i, cname in enumerate(class_names[:6]):
                    fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
                    fig_roc.add_trace(go.Scatter(
                        x=fpr, y=tpr, mode="lines", name=f"{cname} (AUC={auc(fpr, tpr):.3f})",
                        line=dict(color=palette[i % len(palette)], width=3)
                    ))
                fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(color="#94A3B8", dash="dash")))
                st.plotly_chart(style_chart(fig_roc, 300), width="stretch", config=PLOTLY_CONFIG)

        with s2:
            with st.container(border=True):
                st.markdown(f'<div class="bento-title">🎯 Precision-Recall Curve — {sel_sup}</div>', unsafe_allow_html=True)
                fig_pr = go.Figure()
                for i, cname in enumerate(class_names[:6]):
                    prec_c, rec_c, _ = precision_recall_curve(y_bin[:, i], probs[:, i])
                    fig_pr.add_trace(go.Scatter(
                        x=rec_c, y=prec_c, mode="lines", name=f"{cname} (AP={auc(rec_c, prec_c):.3f})",
                        line=dict(color=palette[i % len(palette)], width=3)
                    ))
                fig_pr.update_layout(xaxis_title="Recall", yaxis_title="Precision")
                st.plotly_chart(style_chart(fig_pr, 300), width="stretch", config=PLOTLY_CONFIG)
    else:
        y_pred_reg = active_sup.predict(ai["X_test_p"])
        with s1:
            with st.container(border=True):
                st.markdown(f'<div class="bento-title">📐 Actual vs. Predicted — {sel_sup}</div>', unsafe_allow_html=True)
                reg_df = pd.DataFrame({"Actual": ai["y_test"], "Predicted": y_pred_reg})
                fig_reg = px.scatter(reg_df, x="Actual", y="Predicted", opacity=0.65, color_discrete_sequence=["#1366D6"])
                min_v, max_v = min(ai["y_test"].min(), y_pred_reg.min()), max(ai["y_test"].max(), y_pred_reg.max())
                fig_reg.add_trace(go.Scatter(x=[min_v, max_v], y=[min_v, max_v], mode="lines", name="Ideal Fit", line=dict(color="#E11D48", dash="dash")))
                st.plotly_chart(style_chart(fig_reg, 300), width="stretch", config=PLOTLY_CONFIG)

        with s2:
            with st.container(border=True):
                st.markdown(f'<div class="bento-title">📉 Residuals vs. Predicted — {sel_sup}</div>', unsafe_allow_html=True)
                res_df = pd.DataFrame({"Predicted": y_pred_reg, "Residual": ai["y_test"] - y_pred_reg})
                fig_res = px.scatter(res_df, x="Predicted", y="Residual", opacity=0.65, color_discrete_sequence=["#7C5dfa"])
                fig_res.add_hline(y=0, line_dash="dash", line_color="#E11D48")
                st.plotly_chart(style_chart(fig_res, 300), width="stretch", config=PLOTLY_CONFIG)

    s3, s4 = st.columns(2)
    with s3:
        with st.container(border=True):
            st.markdown(f'<div class="bento-title">📚 Learning Curve (Bias vs. Variance) — {sel_sup}</div>', unsafe_allow_html=True)
            fig_lc = go.Figure()
            fig_lc.add_trace(go.Scatter(x=tr_sizes, y=tr_sc, mode="lines+markers", name="Train Score (%)", line=dict(color="#1366D6", width=3)))
            fig_lc.add_trace(go.Scatter(x=tr_sizes, y=val_sc, mode="lines+markers", name="CV Val Score (%)", line=dict(color="#00A896", width=3)))
            fig_lc.update_layout(xaxis_title="Training Samples", yaxis_title="Score (%)")
            st.plotly_chart(style_chart(fig_lc, 300), width="stretch", config=PLOTLY_CONFIG)

    with s4:
        with st.container(border=True):
            if "Random Forest" in sel_sup:
                st.markdown('<div class="bento-title">🌲 Out-of-Bag (OOB) Error Curve vs. Trees</div>', unsafe_allow_html=True)
                fig_sig = px.line(x=ai["diag"]["oob_trees"], y=ai["diag"]["oob_errors"], markers=True, labels={"x": "Number of Trees", "y": "OOB Error (%)"})
                fig_sig.update_traces(line_color="#E11D48", line_width=3)
            elif "KNN" in sel_sup or "Neighbors" in sel_sup:
                st.markdown('<div class="bento-title">🔍 Accuracy / R² vs. Neighbors (K Sweep)</div>', unsafe_allow_html=True)
                fig_sig = go.Figure()
                fig_sig.add_trace(go.Scatter(x=ai["diag"]["k_vals"], y=ai["diag"]["knn_train"], mode="lines+markers", name="Train (%)", line=dict(color="#1366D6", width=2.5)))
                fig_sig.add_trace(go.Scatter(x=ai["diag"]["k_vals"], y=ai["diag"]["knn_val"], mode="lines+markers", name="Test (%)", line=dict(color="#00A896", width=3)))
                fig_sig.update_layout(xaxis_title="Neighbors (K)", yaxis_title="Score (%)")
            elif "Decision Tree" in sel_sup:
                st.markdown('<div class="bento-title">✂️ Cost-Complexity Pruning Curve (ccp_alpha)</div>', unsafe_allow_html=True)
                fig_sig = px.line(x=ai["diag"]["alphas"], y=ai["diag"]["dt_scores"], markers=True, labels={"x": "Pruning Penalty (ccp_alpha)", "y": "Validation Score (%)"})
                fig_sig.update_traces(line_color="#7C5dfa", line_width=3)
            else:
                st.markdown('<div class="bento-title">⚖️ Learned Feature Weight Magnitudes (|β|)</div>', unsafe_allow_html=True)
                fig_sig = px.bar(ai["diag"]["coef_df"], x="Mean |Weight|", y="Feature", orientation="h", color_discrete_sequence=["#1366D6"])
            st.plotly_chart(style_chart(fig_sig, 300), width="stretch", config=PLOTLY_CONFIG)

    s5, s6 = st.columns(2)
    with s5:
        with st.container(border=True):
            st.markdown(f'<div class="bento-title">🌟 Permutation Importance (PCA Components) — {sel_sup}</div>', unsafe_allow_html=True)
            fig_perm = px.bar(perm_df, x="Importance", y="Component", orientation="h", color_discrete_sequence=["#00A896"])
            st.plotly_chart(style_chart(fig_perm, 290), width="stretch", config=PLOTLY_CONFIG)

    with s6:
        if ai["task_type"] == "Classification":
            with st.container(border=True):
                st.markdown(f'<div class="bento-title">🔢 Confusion Matrix — {sel_sup}</div>', unsafe_allow_html=True)
                cm = confusion_matrix(ai["y_test"], active_sup.predict(ai["X_test_p"]))
                fig_cm = px.imshow(cm, text_auto=True, x=ai["class_names"], y=ai["class_names"], color_continuous_scale="Blues")
                fig_cm.update_coloraxes(colorbar_tickfont=dict(color="#0F172A", size=11))
                st.plotly_chart(style_chart(fig_cm, 290), width="stretch", config=PLOTLY_CONFIG)

# ==================== VIEW 5: UNSUPERVISED ML ====================
elif active == "Unsupervised ML":
    if len(num_cols) < 2:
        st.warning("At least 2 numeric columns are required for Unsupervised Clustering & Anomaly Detection.")
    else:
        if group_col:
            clust_df = df.groupby(group_col)[num_cols].mean().dropna()
        else:
            clust_df = df[num_cols].dropna().head(300)
            clust_df.index = [f"Row_{i}" for i in range(len(clust_df))]

        X_u_scaled = StandardScaler().fit_transform(clust_df)
        max_k = min(6, max(2, len(clust_df) - 1))

        k_val = st.slider("Select K-Means Cluster Count (K):", 2, max_k, min(3, max_k))
        km = KMeans(n_clusters=k_val, random_state=42, n_init=10)
        labels_u = km.fit_predict(X_u_scaled)
        clust_df["Cluster"] = labels_u.astype(str)
        sil = silhouette_score(X_u_scaled, labels_u) if len(set(labels_u)) > 1 else 0.0

        u1, u2 = st.columns([1.25, 1])
        with u1:
            with st.container(border=True):
                st.markdown(f'<div class="bento-title">🧬 2D PCA Cluster Projection (K={k_val} | Silhouette = {sil:.3f})</div>', unsafe_allow_html=True)
                coords = PCA(n_components=2).fit_transform(X_u_scaled)
                clust_df["PCA1"], clust_df["PCA2"] = coords[:, 0], coords[:, 1]
                plot_u = clust_df.reset_index()
                fig_cl = px.scatter(
                    plot_u, x="PCA1", y="PCA2", color="Cluster",
                    hover_name=plot_u.columns[0],
                    text=plot_u.columns[0] if len(plot_u) <= 35 else None,
                    color_discrete_sequence=["#1366D6", "#00A896", "#7C5dfa", "#16A34A", "#F59E0B", "#E11D48"]
                )
                if len(plot_u) <= 35:
                    fig_cl.update_traces(textposition="top center", textfont=dict(color="#0F172A", size=9))
                st.plotly_chart(style_chart(fig_cl, 330), width="stretch", config=PLOTLY_CONFIG)

        with u2:
            with st.container(border=True):
                st.markdown('<div class="bento-title">🔪 Silhouette Analysis Knife Plot</div>', unsafe_allow_html=True)
                sil_vals = silhouette_samples(X_u_scaled, labels_u) if len(set(labels_u)) > 1 else np.zeros(len(labels_u))
                sil_df = pd.DataFrame({"Sample": range(len(sil_vals)), "Silhouette": sil_vals, "Cluster": clust_df["Cluster"]})
                sil_df = sil_df.sort_values(["Cluster", "Silhouette"]).reset_index(drop=True)
                sil_df["Order"] = sil_df.index
                fig_sil = px.bar(
                    sil_df, x="Silhouette", y="Order", color="Cluster", orientation="h",
                    color_discrete_sequence=["#1366D6", "#00A896", "#7C5dfa", "#16A34A", "#F59E0B", "#E11D48"]
                )
                fig_sil.add_vline(x=sil, line_dash="dash", line_color="#E11D48")
                fig_sil.update_yaxes(showticklabels=False, title="Samples")
                st.plotly_chart(style_chart(fig_sil, 330), width="stretch", config=PLOTLY_CONFIG)

        u3, u4 = st.columns(2)
        with u3:
            with st.container(border=True):
                st.markdown('<div class="bento-title">🌡️ Standardized Cluster Profile Heatmap</div>', unsafe_allow_html=True)
                scaled_profile = pd.DataFrame(X_u_scaled, columns=num_cols)
                scaled_profile["Cluster"] = clust_df["Cluster"].values
                prof_means = scaled_profile.groupby("Cluster").mean().round(2)
                fig_hm = px.imshow(prof_means, text_auto=True, color_continuous_scale="RdBu_r", zmin=-2, zmax=2)
                fig_hm.update_coloraxes(colorbar_tickfont=dict(color="#0F172A", size=11))
                st.plotly_chart(style_chart(fig_hm, 280), width="stretch", config=PLOTLY_CONFIG)

        with u4:
            with st.container(border=True):
                st.markdown('<div class="bento-title">🚨 Isolation Forest Anomaly Detection (5%)</div>', unsafe_allow_html=True)
                iso = IsolationForest(contamination=0.05, random_state=42)
                df_anom = df.copy()
                df_anom["Status"] = np.where(iso.fit_predict(df[num_cols]) == -1, "Anomaly", "Normal")
                fig_iso = px.scatter(
                    df_anom, x=num_cols[0], y=num_cols[1], color="Status",
                    color_discrete_map={"Normal": "#94A3B8", "Anomaly": "#E11D48"}
                )
                st.plotly_chart(style_chart(fig_iso, 280), width="stretch", config=PLOTLY_CONFIG)

# ==================== VIEW 6: DEEP LEARNING ====================
elif active == "Deep Learning":
    d1, d2 = st.columns(2)
    with d1:
        with st.container(border=True):
            metric_tag = f"Acc: {ai['dl']['primary']}%" if ai["task_type"] == "Classification" else f"R²: {ai['dl']['primary']}"
            st.markdown(f'<div class="bento-title">🧠 Deep Neural Network Convergence ({metric_tag})</div>', unsafe_allow_html=True)
            ep = list(range(1, len(ai["dl"]["losses"]) + 1))
            fig_dl = go.Figure()
            fig_dl.add_trace(go.Scatter(x=ep, y=ai["dl"]["losses"], name="Training Loss", line=dict(color="#E11D48", width=2.5)))
            fig_dl.add_trace(go.Scatter(x=ep, y=ai["dl"]["val_scores"], name="Val Score (%)", yaxis="y2", line=dict(color="#1366D6", width=2.5)))
            fig_dl.update_layout(yaxis=dict(title="Loss"), yaxis2=dict(title="Score (%)", overlaying="y", side="right"))
            st.plotly_chart(style_chart(fig_dl, 300), width="stretch", config=PLOTLY_CONFIG)

    if ai["task_type"] == "Classification":
        class_names = ai["class_names"]
        dl_probs = ai["dl"]["model"].predict_proba(ai["X_test_p"])
        dl_preds = ai["dl"]["model"].predict(ai["X_test_p"])

        with d2:
            with st.container(border=True):
                st.markdown('<div class="bento-title">📈 Deep MLP Multi-Class ROC Curves</div>', unsafe_allow_html=True)
                y_bin = label_binarize(ai["y_test"], classes=list(range(len(class_names))))
                if len(class_names) == 2:
                    y_bin = np.hstack([1 - y_bin, y_bin])
                fig_dl_roc = go.Figure()
                palette = ["#1366D6", "#00A896", "#7C5dfa", "#E11D48", "#F59E0B", "#16A34A"]
                for i, cname in enumerate(class_names[:6]):
                    fpr, tpr, _ = roc_curve(y_bin[:, i], dl_probs[:, i])
                    fig_dl_roc.add_trace(go.Scatter(
                        x=fpr, y=tpr, mode="lines", name=f"{cname} (AUC={auc(fpr, tpr):.3f})",
                        line=dict(color=palette[i % len(palette)], width=3)
                    ))
                fig_dl_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(color="#94A3B8", dash="dash")))
                st.plotly_chart(style_chart(fig_dl_roc, 300), width="stretch", config=PLOTLY_CONFIG)

        d3, d4 = st.columns(2)
        with d3:
            with st.container(border=True):
                st.markdown('<div class="bento-title">🎯 Deep MLP Confusion Matrix</div>', unsafe_allow_html=True)
                dl_cm = confusion_matrix(ai["y_test"], dl_preds)
                fig_dl_cm = px.imshow(
                    dl_cm, text_auto=True, x=class_names, y=class_names,
                    color_continuous_scale="Teal", labels=dict(x="Predicted", y="Actual", color="Count")
                )
                fig_dl_cm.update_coloraxes(colorbar_tickfont=dict(color="#0F172A", size=11), colorbar_title_font=dict(color="#0F172A"))
                st.plotly_chart(style_chart(fig_dl_cm, 290), width="stretch", config=PLOTLY_CONFIG)

        with d4:
            with st.container(border=True):
                st.markdown('<div class="bento-title">📊 Class-Wise Precision, Recall & F1-Score (%)</div>', unsafe_allow_html=True)
                prec, rec, f1_cls, _ = precision_recall_fscore_support(ai["y_test"], dl_preds, zero_division=0)
                metrics_df = pd.DataFrame({
                    "Category": np.tile(class_names, 3),
                    "Metric": np.repeat(["Precision", "Recall", "F1-Score"], len(class_names)),
                    "Score (%)": np.concatenate([prec * 100, rec * 100, f1_cls * 100]).round(1)
                })
                fig_prf = px.bar(
                    metrics_df, x="Category", y="Score (%)", color="Metric", barmode="group",
                    color_discrete_map={"Precision": "#1366D6", "Recall": "#00A896", "F1-Score": "#7C5dfa"}
                )
                fig_prf.update_traces(texttemplate="%{y:.1f}%", textposition="outside", textfont=dict(size=9, color="#0F172A"))
                fig_prf.update_yaxes(range=[0, 112])
                st.plotly_chart(style_chart(fig_prf, 290), width="stretch", config=PLOTLY_CONFIG)

        d5, d6 = st.columns(2)
        with d5:
            with st.container(border=True):
                st.markdown('<div class="bento-title">🔔 Prediction Confidence Distribution</div>', unsafe_allow_html=True)
                max_conf = np.max(dl_probs, axis=1) * 100
                fig_conf = px.histogram(x=max_conf, nbins=20, labels={"x": "Max Predicted Probability (%)", "y": "Test Samples"}, color_discrete_sequence=["#1366D6"])
                st.plotly_chart(style_chart(fig_conf, 270), width="stretch", config=PLOTLY_CONFIG)

        with d6:
            with st.container(border=True):
                st.markdown('<div class="bento-title">🕸️ First Hidden Layer Synaptic Weights (Top 32 Neurons)</div>', unsafe_allow_html=True)
                w0 = ai["dl"]["model"].coefs_[0][:, :32]
                pc_labels = [f"PC{i+1}" for i in range(w0.shape[0])]
                fig_w = px.imshow(w0, y=pc_labels, color_continuous_scale="RdBu_r", aspect="auto", labels=dict(x="Hidden Neuron (1–32)", y="Input PC", color="Weight"))
                fig_w.update_coloraxes(colorbar_tickfont=dict(color="#0F172A", size=11))
                st.plotly_chart(style_chart(fig_w, 270), width="stretch", config=PLOTLY_CONFIG)
    else:
        with d2:
            with st.container(border=True):
                st.markdown('<div class="bento-title">📐 Deep MLP Regressor: Actual vs. Predicted</div>', unsafe_allow_html=True)
                dl_y_pred = ai["dl"]["model"].predict(ai["X_test_p"])
                reg_dl_df = pd.DataFrame({"Actual": ai["y_test"], "Predicted": dl_y_pred})
                fig_dl_reg = px.scatter(reg_dl_df, x="Actual", y="Predicted", opacity=0.65, color_discrete_sequence=["#00A896"])
                st.plotly_chart(style_chart(fig_dl_reg, 300), width="stretch", config=PLOTLY_CONFIG)

# ==================== VIEW 7: AUTO-GENERATED LIVE PREDICTOR ====================
elif active == "Live Predictor":
    with st.container(border=True):
        st.markdown(f'<div class="bento-title">⚡ Auto-Generated Live AI Inference Form (Target: {target_col})</div>', unsafe_allow_html=True)

        user_inputs = {}
        cols_grid = st.columns(4)
        for i, col in enumerate(feature_cols):
            with cols_grid[i % 4]:
                if col in cat_cols:
                    options = sorted(df[col].unique().tolist())
                    user_inputs[col] = st.selectbox(col, options, key=f"dyn_in_{col}")
                else:
                    med_val = float(df[col].median())
                    user_inputs[col] = st.number_input(col, value=med_val, key=f"dyn_in_{col}")

        all_models = {"Deep Neural Network (MLP)": ai["dl"]["model"], **ai["sup_models"]}
        chosen_model_name = st.selectbox("Select AI Model for Inference:", list(all_models.keys()))

        if st.button("🚀 Run Dynamic Prediction", type="primary"):
            row_df = pd.DataFrame([user_inputs])
            for col in feature_cols:
                if col in ai["encoders"]:
                    row_df[col] = ai["encoders"][col].transform(row_df[col].astype(str))

            row_p = ai["pca_opt"].transform(ai["scaler"].transform(row_df))
            active_clf = all_models[chosen_model_name]

            if ai["task_type"] == "Classification":
                pred_idx = active_clf.predict(row_p)[0]
                pred_label = ai["encoders"][target_col].inverse_transform([pred_idx])[0]
                probs = active_clf.predict_proba(row_p)[0]
                st.success(f"**Predicted {target_col} ({chosen_model_name}):** `{pred_label}` — Confidence: **{max(probs)*100:.2f}%**")

                p_df = pd.DataFrame({"Class": ai["class_names"], "Confidence (%)": (probs * 100).round(2)})
                fig_p = px.bar(p_df, x="Class", y="Confidence (%)", color="Class", text="Confidence (%)",
                               color_discrete_sequence=["#1366D6", "#00A896", "#7C5dfa", "#16A34A"])
                fig_p.update_layout(showlegend=False)
                st.plotly_chart(style_chart(fig_p, 260), width="stretch", config=PLOTLY_CONFIG)
            else:
                pred_val = active_clf.predict(row_p)[0]
                st.success(f"**Predicted {target_col} ({chosen_model_name}):** `{pred_val:,.2f}`")

# ------------------------------------------------------------
# 7. BOTTOM GRADIENT FOOTER BANNER
# ------------------------------------------------------------

st.markdown(f"""
<div class="footer-banner">
    <div class="footer-tagline">💡 Universal Auto-Analytics & AI Engine • Currently Analyzing: {active_filename}</div>
    <div class="footer-skyline">⚙️ Auto-Cleaning • PCA ({ai['pca_opt'].n_components_} PCs) • Supervised • Unsupervised • Deep Learning</div>
</div>
""", unsafe_allow_html=True)
