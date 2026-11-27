# ============================================================
# COMPLETE DATA ANALYTICS, SUPERVISED ML, UNSUPERVISED ML & DL DASHBOARD
# Dataset: Sindh Education Statistics (sindh-education-stats_2.csv)
# Run via terminal: streamlit run d:/DA_Updated/app.py
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# Supervised ML Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Unsupervised ML Models
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

# Deep Learning Model
from sklearn.neural_network import MLPClassifier

# Evaluation Metrics
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    silhouette_score
)

# ------------------------------------------------------------
# 1. PAGE CONFIG & CUSTOM ZERO-TRUNCATION KPI CARDS (BLACK TEXT)
# ------------------------------------------------------------

st.set_page_config(
    page_title="Sindh Education Analytics & AI Hub",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.15rem;
        font-weight: 800;
        background: linear-gradient(90deg, #0f4c81, #00a896);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.15rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #64748b;
        margin-bottom: 1.2rem;
    }
    /* Custom Full-Width KPI Cards: No Ellipsis (...), Strictly Black Text */
    .kpi-card {
        background-color: #ffffff;
        border: 2px solid #cbd5e1;
        border-radius: 12px;
        padding: 14px 14px;
        box-shadow: 0 3px 8px rgba(0, 0, 0, 0.08);
        text-align: left;
        min-height: 98px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .kpi-title {
        color: #000000 !important;
        font-size: 0.88rem;
        font-weight: 700;
        margin-bottom: 6px;
        white-space: nowrap;
    }
    .kpi-value {
        color: #000000 !important;
        font-size: 1.48rem;
        font-weight: 900;
        line-height: 1.2;
        letter-spacing: -0.5px;
        white-space: nowrap;
    }
    /* Also force Black Text on Standard st.metric Cards */
    div[data-testid="stMetric"] {
        background-color: #ffffff !important;
        border: 2px solid #cbd5e1 !important;
        padding: 12px 14px !important;
        border-radius: 10px !important;
    }
    div[data-testid="stMetric"] label,
    div[data-testid="stMetricLabel"] p,
    div[data-testid="stMetricValue"] {
        color: #000000 !important;
        font-weight: 800 !important;
        font-size: 1.35rem !important;
    }
</style>
""", unsafe_allow_html=True)


def render_kpi_card(title, value):
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)


st.markdown('<div class="main-header">🎓 Sindh Education Intelligence: EDA, Supervised ML, Unsupervised ML & Deep Learning</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">End-to-End Analytics Platform • Zero-Truncation Metrics • Separate Supervised, Unsupervised & Deep Learning Suites</div>',
    unsafe_allow_html=True
)

# ------------------------------------------------------------
# 2. DATA LOADING & PREPROCESSING (CACHED)
# ------------------------------------------------------------

@st.cache_data
def load_and_clean_data():
    p2_path = Path("P2_cleaned_data.csv")
    if p2_path.exists():
        df = pd.read_csv(p2_path)
    else:
        raw_path = Path("sindh-education-stats_2.csv")
        if not raw_path.exists():
            raw_path = Path("sindh-education-stats.csv")
        df = pd.read_csv(raw_path)
        df["Population"] = pd.to_numeric(
            df["Population"].astype(str).str.replace(",", "", regex=False).str.strip(),
            errors="coerce"
        )
        for col in ["Number", "Population", "No/100000"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            df[col] = np.clip(df[col], q1 - 1.5 * iqr, q3 + 1.5 * iqr)

        df["Location"] = df["Location"].astype(str).str.strip().str.title()
        df["School Type"] = df["School Type"].astype(str).str.strip().str.title()
        df["Category"] = df["Category"].astype(str).str.strip().str.upper()
        df["Composition"] = df["Composition"].astype(str).str.strip().str.upper()
        df["District"] = df["District"].astype(str).str.strip().str.title().replace({
            "Matira": "Matiari", "Tando Allah Yar": "Tando Allahyar"
        })

    df["Log_Number"] = np.log1p(df["Number"])
    return df


@st.cache_resource
def train_ml_and_dl_suites(df):
    data = df.drop(columns=["District"]).copy()
    encoders = {}
    for col in ["Location", "School Type", "Category", "Composition"]:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        encoders[col] = le

    clf_features = ["Location", "School Type", "Composition", "Number", "Population", "No/100000", "Log_Number"]
    X = data[clf_features]
    y = data["Category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    pca_full = PCA().fit(X_train_s)
    pca_95 = PCA(n_components=0.95)
    X_train_p = pca_95.fit_transform(X_train_s)
    X_test_p = pca_95.transform(X_test_s)

    # --- 1. SUPERVISED ML CLASSIFIERS ---
    sup_models = {
        "Random Forest Classifier": RandomForestClassifier(n_estimators=120, random_state=42),
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree Classifier": DecisionTreeClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
    }

    sup_results = []
    fitted_sup = {}
    for name, model in sup_models.items():
        model.fit(X_train_p, y_train)
        preds = model.predict(X_test_p)
        sup_results.append({
            "Supervised ML Model": name,
            "Accuracy (%)": round(accuracy_score(y_test, preds) * 100, 2),
            "Weighted F1 (%)": round(f1_score(y_test, preds, average="weighted") * 100, 2)
        })
        fitted_sup[name] = model

    sup_leaderboard = pd.DataFrame(sup_results).sort_values(by="Accuracy (%)", ascending=False)

    # Feature importance model on un-PCA features for interpretability
    rf_raw = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
    feat_imp_df = pd.DataFrame({
        "Feature": clf_features,
        "Importance": rf_raw.feature_importances_
    }).sort_values(by="Importance", ascending=True)

    # --- 2. SUPERVISED REGRESSION (PREDICTING 'Number') ---
    reg_features = ["Location", "School Type", "Category", "Composition", "Population", "No/100000"]
    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(data[reg_features], data["Number"], test_size=0.20, random_state=42)
    rf_reg = RandomForestRegressor(n_estimators=100, random_state=42).fit(Xr_tr, yr_tr)
    yr_pred = rf_reg.predict(Xr_te)

    reg_bundle = {
        "r2": round(r2_score(yr_te, yr_pred), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(yr_te, yr_pred))), 2),
        "mae": round(float(mean_absolute_error(yr_te, yr_pred)), 2),
        "y_test": yr_te.values,
        "y_pred": yr_pred
    }

    # --- 3. DEEP LEARNING MODEL (MULTI-LAYER PERCEPTRON WITH EPOCH TRACKING) ---
    dl_mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        max_iter=1,
        warm_start=True,
        random_state=42
    )
    train_losses, val_accs = [], []
    classes = np.unique(y_train)
    for epoch in range(60):
        dl_mlp.partial_fit(X_train_p, y_train, classes=classes)
        train_losses.append(dl_mlp.loss_)
        val_accs.append(accuracy_score(y_test, dl_mlp.predict(X_test_p)) * 100)

    dl_preds = dl_mlp.predict(X_test_p)
    dl_metrics = {
        "model": dl_mlp,
        "accuracy": round(accuracy_score(y_test, dl_preds) * 100, 2),
        "f1": round(f1_score(y_test, dl_preds, average="weighted") * 100, 2),
        "losses": train_losses,
        "val_accs": val_accs
    }

    return {
        "encoders": encoders,
        "scaler": scaler,
        "pca_full": pca_full,
        "pca_95": pca_95,
        "X_test_p": X_test_p,
        "y_test": y_test,
        "sup_models": fitted_sup,
        "sup_leaderboard": sup_leaderboard,
        "feat_imp": feat_imp_df,
        "reg": reg_bundle,
        "dl": dl_metrics
    }


df = load_and_clean_data()
bundle = train_ml_and_dl_suites(df)

# ------------------------------------------------------------
# 3. SIDEBAR INTERACTIVE CONTROLS
# ------------------------------------------------------------

st.sidebar.header("🎛️ Filter Controls")

all_districts = sorted(df["District"].unique().tolist())
selected_district = st.sidebar.selectbox("📍 Select District", ["All Sindh (29 Districts)"] + all_districts)

all_schools = sorted(df["School Type"].unique().tolist())
selected_schools = st.sidebar.multiselect("🏫 School Levels", all_schools, default=all_schools)

selected_location = st.sidebar.radio("🌆 Area Breakdown", ["Total", "Urban", "Rural"], horizontal=True)

st.sidebar.markdown("---")
st.sidebar.info(
    f"**Model Summary:**\n"
    f"- **Supervised ML (RF):** `{bundle['sup_leaderboard'].iloc[0]['Accuracy (%)']}%`\n"
    f"- **Deep Learning (MLP):** `{bundle['dl']['accuracy']}%`\n"
    f"- **Regression $R^2$:** `{bundle['reg']['r2']}`\n"
    f"- **PCA Retained:** `{bundle['pca_95'].n_components_}` PCs (`{bundle['pca_95'].explained_variance_ratio_.sum()*100:.1f}%`)"
)

filtered_df = df[df["School Type"].isin(selected_schools)].copy()
if selected_district != "All Sindh (29 Districts)":
    filtered_df = filtered_df[filtered_df["District"] == selected_district]

# ------------------------------------------------------------
# 4. SEVEN DEDICATED TABS (17 INTERACTIVE CHARTS)
# ------------------------------------------------------------

tab_eda, tab_geo, tab_pca, tab_sup, tab_unsup, tab_dl, tab_pred = st.tabs([
    "📊 1. Executive EDA",
    "🗺️ 2. District & Gender",
    "🧬 3. PCA & Correlations",
    "🎯 4. Supervised ML",
    "🧩 5. Unsupervised ML",
    "🧠 6. Deep Learning",
    "🤖 7. Live Predictor"
])

# ============================================================
# TAB 1: EXECUTIVE EDA (KPI CARDS + CHARTS 1 & 2)
# ============================================================
with tab_eda:
    tot_df = filtered_df[(filtered_df["Location"] == selected_location) & (filtered_df["Composition"] == "TOTAL")]

    inst_cnt = int(tot_df[tot_df["Category"] == "INSTITUTIONS"]["Number"].sum())
    enrol_cnt = int(tot_df[tot_df["Category"] == "ENROLMENT"]["Number"].sum())
    staff_cnt = int(tot_df[tot_df["Category"] == "TEACHING STAFF"]["Number"].sum())
    pop_cnt = int(filtered_df.groupby("District")["Population"].first().sum())
    student_teacher = round(enrol_cnt / staff_cnt, 1) if staff_cnt > 0 else 0

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        render_kpi_card("🏛️ Institutions", f"{inst_cnt:,}")
    with k2:
        render_kpi_card("🎒 Enrolment", f"{enrol_cnt:,}")
    with k3:
        render_kpi_card("👩‍🏫 Teaching Staff", f"{staff_cnt:,}")
    with k4:
        render_kpi_card("👥 Population", f"{pop_cnt:,}")
    with k5:
        render_kpi_card("📐 Student : Teacher", f"{student_teacher} : 1")

    st.markdown("---")
    c_left, c_right = st.columns([1.15, 1])

    with c_left:
        st.subheader("Chart 1: Sunburst Hierarchy (Category → School → Gender)")
        sunburst_df = filtered_df[
            (filtered_df["Location"] == selected_location) &
            (filtered_df["Composition"] != "TOTAL")
        ]
        if not sunburst_df.empty:
            fig_sun = px.sunburst(
                sunburst_df,
                path=["Category", "School Type", "Composition"],
                values="Number",
                color="Category",
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            fig_sun.update_layout(margin=dict(t=10, l=10, r=10, b=10), height=420)
            st.plotly_chart(fig_sun, width="stretch")

    with c_right:
        st.subheader("Chart 2: Statistical Distribution Boxplot by Category")
        fig_box = px.box(
            filtered_df[filtered_df["Location"] == selected_location],
            x="Category", y="Number", color="Category", points="outliers",
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        fig_box.update_layout(showlegend=False, height=420, margin=dict(t=20, l=10, r=10, b=10))
        st.plotly_chart(fig_box, width="stretch")

# ============================================================
# TAB 2: DISTRICT, URBAN-RURAL & GENDER (CHARTS 3, 4 & 5)
# ============================================================
with tab_geo:
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Chart 3: Top 15 Districts by Recorded Institutions")
        dist_rank = (
            df[(df["Category"] == "INSTITUTIONS") & (df["Location"] == "Total") & (df["Composition"] == "TOTAL")]
            .groupby("District")["Number"].sum().sort_values(ascending=True).tail(15).reset_index()
        )
        fig_dist = px.bar(
            dist_rank, x="Number", y="District", orientation="h",
            color="Number", color_continuous_scale="Tealgrn", text_auto=".2s"
        )
        fig_dist.update_layout(height=440, coloraxis_showscale=False)
        st.plotly_chart(fig_dist, width="stretch")

    with col_b:
        st.subheader("Chart 4: Urban vs. Rural Infrastructure Gap")
        ur_data = filtered_df[
            (filtered_df["Category"] == "INSTITUTIONS") &
            (filtered_df["Location"].isin(["Urban", "Rural"])) &
            (filtered_df["Composition"] == "TOTAL")
        ].groupby(["School Type", "Location"])["Number"].sum().reset_index()

        fig_ur = px.bar(
            ur_data, x="School Type", y="Number", color="Location", barmode="group",
            text_auto=".2s", color_discrete_map={"Urban": "#0f4c81", "Rural": "#ff7f0e"}
        )
        fig_ur.update_layout(height=440)
        st.plotly_chart(fig_ur, width="stretch")

    st.markdown("---")
    st.subheader("Chart 5: Male vs. Female Enrolment Comparison by School Level")
    mf_data = filtered_df[
        (filtered_df["Category"] == "ENROLMENT") &
        (filtered_df["Location"] == selected_location) &
        (filtered_df["Composition"].isin(["MALE", "FEMALE"]))
    ].groupby(["School Type", "Composition"])["Number"].sum().reset_index()

    if not mf_data.empty:
        fig_mf = px.bar(
            mf_data, x="School Type", y="Number", color="Composition", barmode="group",
            text_auto=".2s", color_discrete_map={"MALE": "#2b5c8f", "FEMALE": "#e63946"}
        )
        fig_mf.update_layout(height=360)
        st.plotly_chart(fig_mf, width="stretch")

# ============================================================
# TAB 3: PCA & CORRELATIONS (CHARTS 6 & 7)
# ============================================================
with tab_pca:
    p_left, p_right = st.columns(2)

    with p_left:
        st.subheader("Chart 6: PCA Scree & Cumulative Explained Variance")
        exp_var = bundle["pca_full"].explained_variance_ratio_ * 100
        cum_var = np.cumsum(exp_var)
        pcs = [f"PC{i+1}" for i in range(len(exp_var))]

        fig_pca = go.Figure()
        fig_pca.add_trace(go.Bar(x=pcs, y=exp_var, name="Individual Variance (%)", marker_color="#00a896", opacity=0.75))
        fig_pca.add_trace(go.Scatter(
            x=pcs, y=cum_var, name="Cumulative Variance (%)",
            mode="lines+markers", line=dict(color="#0f4c81", width=3)
        ))
        fig_pca.add_hline(y=95, line_dash="dash", line_color="red", annotation_text="95% Threshold (5 PCs)")
        fig_pca.update_layout(yaxis_title="Variance Explained (%)", height=420, legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig_pca, width="stretch")

    with p_right:
        st.subheader("Chart 7: Feature Correlation Heatmap")
        num_corr = df[["Number", "Population", "No/100000", "Log_Number"]].corr().round(3)
        fig_corr = px.imshow(num_corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
        fig_corr.update_layout(height=420)
        st.plotly_chart(fig_corr, width="stretch")

# ============================================================
# TAB 4: DEDICATED SUPERVISED ML SECTION (CHARTS 8, 9, 10 & 11)
# ============================================================
with tab_sup:
    st.header("🎯 Supervised Machine Learning (Classification & Regression)")

    st.subheader("1. Supervised Classification Leaderboard (Trained on PCA Features)")
    st.dataframe(bundle["sup_leaderboard"], width="stretch", hide_index=True)

    sel_sup_model = st.selectbox(
        "Select Supervised Classifier to Inspect:",
        bundle["sup_leaderboard"]["Supervised ML Model"].tolist()
    )
    active_sup = bundle["sup_models"][sel_sup_model]
    X_te = bundle["X_test_p"]
    y_te = bundle["y_test"]
    class_names = bundle["encoders"]["Category"].classes_

    r_col, c_col = st.columns(2)
    with r_col:
        st.subheader(f"Chart 8: Supervised Multi-Class ROC — {sel_sup_model}")
        y_te_bin = label_binarize(y_te, classes=[0, 1, 2])
        y_probs = active_sup.predict_proba(X_te)

        fig_roc = go.Figure()
        palette = ["#0f4c81", "#e63946", "#00a896"]
        for i, cls_label in enumerate(class_names):
            fpr, tpr, _ = roc_curve(y_te_bin[:, i], y_probs[:, i])
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines",
                name=f"{cls_label} (AUC = {auc(fpr, tpr):.3f})",
                line=dict(color=palette[i], width=3)
            ))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Baseline", line=dict(color="gray", dash="dash")))
        fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=400)
        st.plotly_chart(fig_roc, width="stretch")

    with c_col:
        st.subheader(f"Chart 9: Supervised Confusion Matrix — {sel_sup_model}")
        cm = confusion_matrix(y_te, active_sup.predict(X_te))
        fig_cm = px.imshow(
            cm, text_auto=True, x=class_names, y=class_names, color_continuous_scale="Blues",
            labels=dict(x="Predicted", y="Actual", color="Count")
        )
        fig_cm.update_layout(height=400)
        st.plotly_chart(fig_cm, width="stretch")

    st.markdown("---")
    st.subheader("2. Supervised Regression & Feature Importance")
    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        render_kpi_card("📈 Regression R² Score", f"{bundle['reg']['r2']}")
    with rc2:
        render_kpi_card("📉 RMSE (Error)", f"{bundle['reg']['rmse']:,}")
    with rc3:
        render_kpi_card("📏 MAE (Error)", f"{bundle['reg']['mae']:,}")

    reg_left, reg_right = st.columns(2)
    with reg_left:
        st.subheader("Chart 10: Actual vs. Predicted Count (Random Forest Regressor)")
        reg_df = pd.DataFrame({"Actual": bundle["reg"]["y_test"], "Predicted": bundle["reg"]["y_pred"]})
        fig_reg = px.scatter(reg_df, x="Actual", y="Predicted", opacity=0.6, color_discrete_sequence=["#0f4c81"])
        fig_reg.add_trace(go.Scatter(
            x=[0, reg_df["Actual"].max()], y=[0, reg_df["Actual"].max()],
            mode="lines", name="Ideal Fit (y=x)", line=dict(color="red", dash="dash")
        ))
        fig_reg.update_layout(height=380)
        st.plotly_chart(fig_reg, width="stretch")

    with reg_right:
        st.subheader("Chart 11: Supervised Feature Importance Ranking")
        fig_imp = px.bar(
            bundle["feat_imp"], x="Importance", y="Feature", orientation="h",
            color="Importance", color_continuous_scale="Viridis", text_auto=".3f"
        )
        fig_imp.update_layout(height=380, coloraxis_showscale=False)
        st.plotly_chart(fig_imp, width="stretch")

# ============================================================
# TAB 5: DEDICATED UNSUPERVISED ML SECTION (CHARTS 12, 13 & 14)
# ============================================================
with tab_unsup:
    st.header("🧩 Unsupervised Machine Learning (K-Means Clustering & Anomaly Detection)")

    dist_agg = pd.concat([
        df[(df["Category"] == "INSTITUTIONS") & (df["Location"] == "Total") & (df["Composition"] == "TOTAL")].groupby("District")["Number"].sum().rename("Institutions"),
        df[(df["Category"] == "ENROLMENT") & (df["Location"] == "Total") & (df["Composition"] == "TOTAL")].groupby("District")["Number"].sum().rename("Enrolment"),
        df[(df["Category"] == "TEACHING STAFF") & (df["Location"] == "Total") & (df["Composition"] == "TOTAL")].groupby("District")["Number"].sum().rename("Staff"),
        df.groupby("District")["Population"].first().rename("Population")
    ], axis=1).dropna()

    X_d_scaled = StandardScaler().fit_transform(dist_agg)

    u_left, u_right = st.columns([1, 1.35])
    with u_left:
        k_clusters = st.slider("Select K-Means Cluster Count (K):", min_value=2, max_value=6, value=3)
        km = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
        dist_agg["Cluster"] = km.fit_predict(X_d_scaled).astype(str)
        sil = silhouette_score(X_d_scaled, dist_agg["Cluster"])

        render_kpi_card("🎯 Silhouette Score (Cluster Separation)", f"{sil:.4f}")
        st.markdown("<br>", unsafe_allow_html=True)
        st.write("**Mean District Profile per Cluster:**")
        st.dataframe(
            dist_agg.groupby("Cluster")[["Institutions", "Enrolment", "Staff", "Population"]].mean().round(1),
            width="stretch"
        )

    with u_right:
        st.subheader("Chart 12: K-Means Elbow Curve (Inertia vs. K)")
        inertias = [KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_d_scaled).inertia_ for k in range(2, 8)]
        fig_elbow = px.line(
            x=list(range(2, 8)), y=inertias, markers=True,
            labels={"x": "Number of Clusters (K)", "y": "Inertia (WCSS)"}
        )
        fig_elbow.update_layout(height=360)
        st.plotly_chart(fig_elbow, width="stretch")

    st.subheader(f"Chart 13: District Segmentation Bubble Map (K={k_clusters} via 2D PCA)")
    coords = PCA(n_components=2).fit_transform(X_d_scaled)
    dist_agg["PCA1"] = coords[:, 0]
    dist_agg["PCA2"] = coords[:, 1]
    fig_clust = px.scatter(
        dist_agg.reset_index(), x="PCA1", y="PCA2", color="Cluster", size="Population",
        hover_name="District", hover_data=["Institutions", "Enrolment", "Staff"], text="District",
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    fig_clust.update_traces(textposition="top center", textfont_size=9)
    fig_clust.update_layout(height=460)
    st.plotly_chart(fig_clust, width="stretch")

    st.markdown("---")
    st.subheader("Chart 14: Unsupervised Anomaly Detection (Isolation Forest)")
    iso = IsolationForest(contamination=0.05, random_state=42)
    df_anom = df.copy()
    df_anom["Status"] = np.where(iso.fit_predict(df[["Number", "Population", "No/100000"]]) == -1, "Anomaly (Outlier)", "Normal")
    fig_iso = px.scatter(
        df_anom, x="Population", y="Number", color="Status",
        hover_data=["District", "Category", "School Type"],
        color_discrete_map={"Normal": "#94a3b8", "Anomaly (Outlier)": "#e63946"}
    )
    fig_iso.update_layout(height=380)
    st.plotly_chart(fig_iso, width="stretch")

# ============================================================
# TAB 6: DEDICATED DEEP LEARNING SECTION (CHARTS 15, 16 & 17)
# ============================================================
with tab_dl:
    st.header("🧠 Deep Learning Suite — Multi-Layer Perceptron Neural Network (128 → 64 → 32)")

    dl1, dl2, dl3, dl4 = st.columns(4)
    with dl1:
        render_kpi_card("🧠 DL Test Accuracy", f"{bundle['dl']['accuracy']}%")
    with dl2:
        render_kpi_card("⚖️ DL Weighted F1", f"{bundle['dl']['f1']}%")
    with dl3:
        render_kpi_card("🏗️ Hidden Layers", "128 → 64 → 32")
    with dl4:
        render_kpi_card("⚡ Activation / Optimizer", "ReLU / Adam")

    st.markdown("---")
    dl_c1, dl_c2 = st.columns(2)

    with dl_c1:
        st.subheader("Chart 15: Neural Network Training Loss & Validation Accuracy")
        epochs_range = list(range(1, len(bundle["dl"]["losses"]) + 1))
        fig_learn = go.Figure()
        fig_learn.add_trace(go.Scatter(
            x=epochs_range, y=bundle["dl"]["losses"], name="Training Loss",
            line=dict(color="#e63946", width=2.5)
        ))
        fig_learn.add_trace(go.Scatter(
            x=epochs_range, y=bundle["dl"]["val_accs"], name="Validation Accuracy (%)",
            yaxis="y2", line=dict(color="#0f4c81", width=2.5)
        ))
        fig_learn.update_layout(
            xaxis_title="Training Epoch",
            yaxis=dict(title="Cross-Entropy Loss"),
            yaxis2=dict(title="Accuracy (%)", overlaying="y", side="right"),
            height=410,
            legend=dict(orientation="h", y=1.12)
        )
        st.plotly_chart(fig_learn, width="stretch")

    with dl_c2:
        st.subheader("Chart 16: Deep Neural Network Multi-Class ROC Curves")
        y_te_bin = label_binarize(bundle["y_test"], classes=[0, 1, 2])
        y_probs_dl = bundle["dl"]["model"].predict_proba(bundle["X_test_p"])
        class_names = bundle["encoders"]["Category"].classes_

        fig_dl_roc = go.Figure()
        palette = ["#0f4c81", "#e63946", "#00a896"]
        for i, cls_label in enumerate(class_names):
            fpr, tpr, _ = roc_curve(y_te_bin[:, i], y_probs_dl[:, i])
            fig_dl_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines",
                name=f"{cls_label} (AUC = {auc(fpr, tpr):.3f})",
                line=dict(color=palette[i], width=3)
            ))
        fig_dl_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(color="gray", dash="dash")))
        fig_dl_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=410)
        st.plotly_chart(fig_dl_roc, width="stretch")

    st.subheader("Chart 17: Deep Neural Network Confusion Matrix")
    dl_cm = confusion_matrix(bundle["y_test"], bundle["dl"]["model"].predict(bundle["X_test_p"]))
    fig_dl_cm = px.imshow(
        dl_cm, text_auto=True, x=class_names, y=class_names, color_continuous_scale="YlGnBu",
        labels=dict(x="Predicted Category", y="Actual Category", color="Samples")
    )
    fig_dl_cm.update_layout(height=380)
    st.plotly_chart(fig_dl_cm, width="stretch")

# ============================================================
# TAB 7: LIVE AI PREDICTOR
# ============================================================
with tab_pred:
    st.header("🤖 Live Multi-Paradigm AI Inference Studio")
    st.caption("Compare predictions between Deep Learning (MLP) and Supervised ML (Random Forest, KNN, Decision Tree, Logistic Regression).")

    i1, i2, i3, i4 = st.columns(4)
    in_loc = i1.selectbox("Location", bundle["encoders"]["Location"].classes_, key="p_loc")
    in_school = i2.selectbox("School Type", bundle["encoders"]["School Type"].classes_, key="p_sch")
    in_comp = i3.selectbox("Composition", bundle["encoders"]["Composition"].classes_, key="p_cmp")
    in_num = i4.number_input("Recorded Count (Number)", min_value=0.0, max_value=10000.0, value=320.0)

    all_inference_models = {"Deep Neural Network (MLP - DL)": bundle["dl"]["model"], **bundle["sup_models"]}

    i5, i6, i7 = st.columns([1, 1, 1.3])
    in_pop = i5.number_input("District Population", min_value=100000.0, max_value=5000000.0, value=1600000.0)
    in_rate = i6.number_input("Rate per 100k (No/100000)", min_value=0.0, max_value=1000.0, value=20.0)
    in_choice = i7.selectbox("Choose AI Model", list(all_inference_models.keys()))

    if st.button("🚀 Predict Category", type="primary"):
        row = pd.DataFrame([{
            "Location": bundle["encoders"]["Location"].transform([in_loc])[0],
            "School Type": bundle["encoders"]["School Type"].transform([in_school])[0],
            "Composition": bundle["encoders"]["Composition"].transform([in_comp])[0],
            "Number": in_num,
            "Population": in_pop,
            "No/100000": in_rate,
            "Log_Number": np.log1p(in_num)
        }])

        row_scaled = bundle["scaler"].transform(row)
        row_pca = bundle["pca_95"].transform(row_scaled)
        clf = all_inference_models[in_choice]

        pred_idx = clf.predict(row_pca)[0]
        probs = clf.predict_proba(row_pca)[0]
        pred_label = bundle["encoders"]["Category"].inverse_transform([pred_idx])[0]

        st.success(f"### Predicted Category ({in_choice}): **{pred_label}** (`{max(probs)*100:.2f}%` Confidence)")

        prob_df = pd.DataFrame({
            "Category": bundle["encoders"]["Category"].classes_,
            "Confidence (%)": (probs * 100).round(2)
        })
        fig_prob = px.bar(
            prob_df, x="Category", y="Confidence (%)", color="Category",
            text="Confidence (%)", color_discrete_sequence=["#0f4c81", "#00a896", "#e63946"]
        )
        fig_prob.update_layout(height=320, showlegend=False)
        st.plotly_chart(fig_prob, width="stretch")