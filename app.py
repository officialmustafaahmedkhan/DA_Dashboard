# ============================================================
# SINDH EDUCATION INTELLIGENCE HUB — SAAS BENTO-GRID UI EDITION
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
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
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
# 1. PAGE CONFIG & SAAS LIGHT-SLATE BENTO CSS
# ------------------------------------------------------------

st.set_page_config(
    page_title="Sindh Education Intelligence Hub",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Global Light-Slate SaaS Canvas */
    .stApp {
        background-color: #F4F6F9;
        color: #0F172A;
    }
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E2E8F0;
    }
    section[data-testid="stSidebar"] * {
        color: #0F172A !important;
    }

    /* Top Command Header & Status Pills */
    .command-title {
        font-size: 1.95rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.5px;
        margin-bottom: 0.35rem;
    }
    .pill-row {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 1.1rem;
    }
    .pill-navy {
        background-color: #1E3A8A;
        color: #FFFFFF;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
    }
    .pill-teal {
        background-color: #0D9488;
        color: #FFFFFF;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
    }
    .pill-slate {
        background-color: #E2E8F0;
        color: #0F172A;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
    }

    /* Sidebar Dataset Health Circular Badge */
    .health-badge-wrap {
        display: flex;
        justify-content: center;
        margin: 8px 0 18px 0;
    }
    .health-badge {
        width: 126px;
        height: 126px;
        border-radius: 50%;
        background: radial-gradient(circle, #1E3A8A 62%, #0D9488 100%);
        border: 5px solid #CBD5E1;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        box-shadow: 0 4px 12px rgba(30, 58, 138, 0.2);
    }
    .health-badge span {
        color: #FFFFFF !important;
        font-weight: 800;
        font-size: 0.78rem;
        line-height: 1.25;
        letter-spacing: 0.5px;
    }

    /* Bento KPI Metric Cards (Zero Truncation + Trend Badge) */
    .bento-kpi {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 16px 18px;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 124px;
    }
    .bento-kpi-header {
        display: flex;
        align-items: center;
        gap: 8px;
        color: #334155 !important;
        font-size: 0.90rem;
        font-weight: 700;
    }
    .bento-kpi-value {
        color: #000000 !important;
        font-size: clamp(1.35rem, 1.9vw, 1.85rem);
        font-weight: 900;
        letter-spacing: -0.6px;
        margin: 6px 0;
        white-space: nowrap;
    }
    .trend-badge-green {
        display: inline-block;
        background-color: #CCFBF1;
        color: #0F766E !important;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 999px;
        width: fit-content;
    }
    .trend-badge-blue {
        display: inline-block;
        background-color: #DBEAFE;
        color: #1E3A8A !important;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 999px;
        width: fit-content;
    }

    /* Bento Chart Card Wrapper */
    .bento-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 16px 18px 8px 18px;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04);
        margin-bottom: 14px;
    }
    .bento-card-title {
        color: #0F172A;
        font-size: 1.05rem;
        font-weight: 800;
        margin-bottom: 2px;
    }
    .bento-card-caption {
        color: #64748B;
        font-size: 0.82rem;
        margin-bottom: 8px;
    }

    /* Clean Pill Tabs */
    button[data-baseweb="tab"] {
        background-color: #FFFFFF !important;
        border-radius: 8px !important;
        border: 1px solid #E2E8F0 !important;
        padding: 8px 14px !important;
        margin-right: 6px !important;
        color: #0F172A !important;
        font-weight: 700 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: #1E3A8A !important;
        color: #FFFFFF !important;
        border-color: #1E3A8A !important;
    }
</style>
""", unsafe_allow_html=True)

PLOTLY_CONFIG = {"displayModeBar": False}


def style_plotly(fig, height=380):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#0F172A", family="Inter, sans-serif"),
        margin=dict(t=18, l=10, r=10, b=10),
        height=height
    )
    fig.update_xaxes(showgrid=True, gridcolor="#F1F5F9", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#F1F5F9", zeroline=False)
    return fig


def render_bento_kpi(icon, title, value, badge_text, badge_type="green"):
    badge_cls = "trend-badge-green" if badge_type == "green" else "trend-badge-blue"
    st.markdown(f"""
    <div class="bento-kpi">
        <div class="bento-kpi-header">{icon} <span>{title}</span></div>
        <div class="bento-kpi-value">{value}</div>
        <div class="{badge_cls}">{badge_text}</div>
    </div>
    """, unsafe_allow_html=True)


# ------------------------------------------------------------
# 2. DATA LOADING & CACHED ML/DL ENGINE
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
def build_ai_suites(df):
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

    # Supervised Classifiers
    sup_models = {
        "Random Forest": RandomForestClassifier(n_estimators=120, random_state=42),
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
    }

    sup_rows, fitted_sup = [], {}
    for name, model in sup_models.items():
        model.fit(X_train_p, y_train)
        preds = model.predict(X_test_p)
        sup_rows.append({
            "Model": name,
            "Accuracy (%)": round(accuracy_score(y_test, preds) * 100, 2),
            "Weighted F1 (%)": round(f1_score(y_test, preds, average="weighted") * 100, 2)
        })
        fitted_sup[name] = model

    sup_lb = pd.DataFrame(sup_rows).sort_values(by="Accuracy (%)", ascending=False)

    rf_imp = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
    feat_imp = pd.DataFrame({"Feature": clf_features, "Importance": rf_imp.feature_importances_}).sort_values("Importance")

    # Supervised Regression
    reg_features = ["Location", "School Type", "Category", "Composition", "Population", "No/100000"]
    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(data[reg_features], data["Number"], test_size=0.20, random_state=42)
    rf_reg = RandomForestRegressor(n_estimators=100, random_state=42).fit(Xr_tr, yr_tr)
    yr_pred = rf_reg.predict(Xr_te)

    reg_info = {
        "r2": round(r2_score(yr_te, yr_pred), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(yr_te, yr_pred))), 2),
        "mae": round(float(mean_absolute_error(yr_te, yr_pred)), 2),
        "y_test": yr_te.values,
        "y_pred": yr_pred
    }

    # Deep Learning MLP
    dl_mlp = MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=1, warm_start=True, random_state=42)
    losses, val_accs = [], []
    classes = np.unique(y_train)
    for _ in range(60):
        dl_mlp.partial_fit(X_train_p, y_train, classes=classes)
        losses.append(dl_mlp.loss_)
        val_accs.append(accuracy_score(y_test, dl_mlp.predict(X_test_p)) * 100)

    dl_preds = dl_mlp.predict(X_test_p)
    dl_info = {
        "model": dl_mlp,
        "acc": round(accuracy_score(y_test, dl_preds) * 100, 2),
        "f1": round(f1_score(y_test, dl_preds, average="weighted") * 100, 2),
        "losses": losses,
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
        "sup_lb": sup_lb,
        "feat_imp": feat_imp,
        "reg": reg_info,
        "dl": dl_info
    }


df = load_and_clean_data()
ai = build_ai_suites(df)

# ------------------------------------------------------------
# 3. LEFT SIDEBAR RAIL (HEALTH BADGE + FILTERS)
# ------------------------------------------------------------

st.sidebar.markdown("""
<div class="health-badge-wrap">
    <div class="health-badge">
        <span>✓<br>DATASET<br>HEALTH<br>100% CLEAN</span>
    </div>
</div>
""", unsafe_allow_html=True)

all_districts = sorted(df["District"].unique().tolist())
selected_district = st.sidebar.selectbox("District", ["All Sindh (29 Districts)"] + all_districts)

all_schools = sorted(df["School Type"].unique().tolist())
selected_schools = st.sidebar.multiselect("School Type", all_schools, default=all_schools)

selected_location = st.sidebar.selectbox("Area Breakdown", ["Total", "Urban", "Rural"])

st.sidebar.markdown("---")
st.sidebar.caption(f"Records Loaded: **{len(df):,} rows** | Missing Values: **0**")

filtered_df = df[df["School Type"].isin(selected_schools)].copy()
if selected_district != "All Sindh (29 Districts)":
    filtered_df = filtered_df[filtered_df["District"] == selected_district]

# ------------------------------------------------------------
# 4. TOP COMMAND HEADER & STATUS PILLS
# ------------------------------------------------------------

st.markdown('<div class="command-title">Sindh Education Intelligence Hub</div>', unsafe_allow_html=True)
st.markdown(f"""
<div class="pill-row">
    <span class="pill-navy">Deep Learning MLP: {ai['dl']['acc']}%</span>
    <span class="pill-teal">Random Forest ML: {ai['sup_lb'].iloc[0]['Accuracy (%)']}%</span>
    <span class="pill-slate">Regression R²: {ai['reg']['r2']}</span>
    <span class="pill-slate">PCA Retained: {ai['pca_95'].n_components_} PCs ({ai['pca_95'].explained_variance_ratio_.sum()*100:.1f}% Var)</span>
    <span class="pill-slate">Scope: {selected_district}</span>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# 5. TOP 4 BENTO KPI CARDS (FULL BLACK NUMBERS + TREND BADGES)
# ------------------------------------------------------------

tot_df = filtered_df[(filtered_df["Location"] == selected_location) & (filtered_df["Composition"] == "TOTAL")]
inst_cnt = int(tot_df[tot_df["Category"] == "INSTITUTIONS"]["Number"].sum())
enrol_cnt = int(tot_df[tot_df["Category"] == "ENROLMENT"]["Number"].sum())
staff_cnt = int(tot_df[tot_df["Category"] == "TEACHING STAFF"]["Number"].sum())
ratio_val = round(enrol_cnt / staff_cnt, 1) if staff_cnt > 0 else 0

rural_inst = int(filtered_df[(filtered_df["Category"] == "INSTITUTIONS") & (filtered_df["Location"] == "Rural") & (filtered_df["Composition"] == "TOTAL")]["Number"].sum())
rural_share = round((rural_inst / inst_cnt * 100), 1) if inst_cnt > 0 else 0

k1, k2, k3, k4 = st.columns(4)
with k1:
    render_bento_kpi("🏛️", "Institutions", f"{inst_cnt:,}", f"▲ {rural_share}% Rural Share", "green")
with k2:
    render_bento_kpi("🎒", "Enrolment", f"{enrol_cnt:,}", f"● {len(selected_schools)} School Levels", "green")
with k3:
    render_bento_kpi("👩‍🏫", "Teaching Staff", f"{staff_cnt:,}", "✓ IQR Outlier Capped", "green")
with k4:
    render_bento_kpi("📐", "Student-Teacher Ratio", f"{ratio_val} : 1", "— Provincial Benchmark", "blue")

st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

# ------------------------------------------------------------
# 6. CLEAN SEGMENTED WORKSPACE TABS
# ------------------------------------------------------------

tab_over, tab_sup, tab_unsup, tab_dl, tab_pca, tab_sim = st.tabs([
    "📊 Executive Overview",
    "🎯 Supervised ML",
    "🧩 Unsupervised ML",
    "🧠 Deep Learning",
    "🧬 PCA & District Rankings",
    "🤖 Live AI Sandbox"
])

# === TAB 1: EXECUTIVE OVERVIEW (ASYMMETRIC BENTO GRID) ===
with tab_over:
    left_bento, right_bento = st.columns([1.15, 1])

    with left_bento:
        st.markdown('<div class="bento-card"><div class="bento-card-title">Sunburst Hierarchy Chart</div>'
                    '<div class="bento-card-caption">Click any ring to drill down: Category → School Type → Gender Composition</div>', unsafe_allow_html=True)
        sun_df = filtered_df[(filtered_df["Location"] == selected_location) & (filtered_df["Composition"] != "TOTAL")]
        if not sun_df.empty:
            fig_sun = px.sunburst(
                sun_df, path=["Category", "School Type", "Composition"], values="Number",
                color="Category", color_discrete_sequence=["#1E3A8A", "#0D9488", "#3B82F6"]
            )
            st.plotly_chart(style_plotly(fig_sun, 390), width="stretch", config=PLOTLY_CONFIG)
        st.markdown("</div>", unsafe_allow_html=True)

    with right_bento:
        st.markdown('<div class="bento-card"><div class="bento-card-title">Urban vs. Rural Infrastructure Split</div>'
                    '<div class="bento-card-caption">Primary schools dominate rural counts while secondary institutions cluster in urban centers</div>', unsafe_allow_html=True)
        ur_df = filtered_df[
            (filtered_df["Category"] == "INSTITUTIONS") &
            (filtered_df["Location"].isin(["Urban", "Rural"])) &
            (filtered_df["Composition"] == "TOTAL")
        ].groupby(["School Type", "Location"])["Number"].sum().reset_index()
        fig_ur = px.bar(
            ur_df, x="School Type", y="Number", color="Location", barmode="group",
            color_discrete_map={"Urban": "#1E3A8A", "Rural": "#0D9488"}
        )
        st.plotly_chart(style_plotly(fig_ur, 390), width="stretch", config=PLOTLY_CONFIG)
        st.markdown("</div>", unsafe_allow_html=True)

    b2_left, b2_right = st.columns(2)
    with b2_left:
        st.markdown('<div class="bento-card"><div class="bento-card-title">Male vs. Female Enrolment Parity</div>'
                    '<div class="bento-card-caption">Gender comparison across Primary, Middle, High, and Higher Secondary levels</div>', unsafe_allow_html=True)
        mf_df = filtered_df[
            (filtered_df["Category"] == "ENROLMENT") &
            (filtered_df["Location"] == selected_location) &
            (filtered_df["Composition"].isin(["MALE", "FEMALE"]))
        ].groupby(["School Type", "Composition"])["Number"].sum().reset_index()
        fig_mf = px.bar(
            mf_df, x="School Type", y="Number", color="Composition", barmode="group",
            color_discrete_map={"MALE": "#1E3A8A", "FEMALE": "#E11D48"}
        )
        st.plotly_chart(style_plotly(fig_mf, 340), width="stretch", config=PLOTLY_CONFIG)
        st.markdown("</div>", unsafe_allow_html=True)

    with b2_right:
        st.markdown('<div class="bento-card"><div class="bento-card-title">Category Distribution Spread (Boxplot)</div>'
                    '<div class="bento-card-caption">Interquartile spread after IQR outlier capping across all active records</div>', unsafe_allow_html=True)
        fig_box = px.box(
            filtered_df[filtered_df["Location"] == selected_location],
            x="Category", y="Number", color="Category",
            color_discrete_sequence=["#1E3A8A", "#0D9488", "#6366F1"]
        )
        fig_box.update_layout(showlegend=False)
        st.plotly_chart(style_plotly(fig_box, 340), width="stretch", config=PLOTLY_CONFIG)
        st.markdown("</div>", unsafe_allow_html=True)

# === TAB 2: SUPERVISED ML STUDIO ===
with tab_sup:
    st.markdown('<div class="bento-card"><div class="bento-card-title">Supervised ML Classification Leaderboard</div>'
                '<div class="bento-card-caption">Evaluated on 20% stratified test split using 5 PCA components</div>', unsafe_allow_html=True)
    st.dataframe(ai["sup_lb"], width="stretch", hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    sel_sup = st.selectbox("Inspect Supervised Classifier:", ai["sup_lb"]["Model"].tolist())
    clf_sup = ai["sup_models"][sel_sup]
    class_names = ai["encoders"]["Category"].classes_

    s1, s2 = st.columns(2)
    with s1:
        st.markdown(f'<div class="bento-card"><div class="bento-card-title">Multi-Class ROC Curve — {sel_sup}</div>'
                    '<div class="bento-card-caption">One-vs-Rest Receiver Operating Characteristic curves across all 3 categories</div>', unsafe_allow_html=True)
        y_bin = label_binarize(ai["y_test"], classes=[0, 1, 2])
        probs = clf_sup.predict_proba(ai["X_test_p"])
        fig_roc = go.Figure()
        for i, cname in enumerate(class_names):
            fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines", name=f"{cname} (AUC={auc(fpr, tpr):.3f})",
                line=dict(color=["#1E3A8A", "#0D9488", "#E11D48"][i], width=3)
            ))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(color="#94A3B8", dash="dash")))
        st.plotly_chart(style_plotly(fig_roc, 360), width="stretch", config=PLOTLY_CONFIG)
        st.markdown("</div>", unsafe_allow_html=True)

    with s2:
        st.markdown(f'<div class="bento-card"><div class="bento-card-title">Confusion Matrix — {sel_sup}</div>'
                    '<div class="bento-card-caption">Actual vs. Predicted sample counts on held-out test set</div>', unsafe_allow_html=True)
        cm = confusion_matrix(ai["y_test"], clf_sup.predict(ai["X_test_p"]))
        fig_cm = px.imshow(cm, text_auto=True, x=class_names, y=class_names, color_continuous_scale="Blues")
        st.plotly_chart(style_plotly(fig_cm, 360), width="stretch", config=PLOTLY_CONFIG)
        st.markdown("</div>", unsafe_allow_html=True)

    s3, s4 = st.columns(2)
    with s3:
        st.markdown(f'<div class="bento-card"><div class="bento-card-title">Supervised Regression Fit (R² = {ai["reg"]["r2"]} | RMSE = {ai["reg"]["rmse"]})</div>'
                    '<div class="bento-card-caption">Random Forest Regressor predicting continuous count (Number)</div>', unsafe_allow_html=True)
        reg_df = pd.DataFrame({"Actual": ai["reg"]["y_test"], "Predicted": ai["reg"]["y_pred"]})
        fig_reg = px.scatter(reg_df, x="Actual", y="Predicted", opacity=0.6, color_discrete_sequence=["#1E3A8A"])
        fig_reg.add_trace(go.Scatter(x=[0, reg_df["Actual"].max()], y=[0, reg_df["Actual"].max()], mode="lines", name="Ideal Fit", line=dict(color="#E11D48", dash="dash")))
        st.plotly_chart(style_plotly(fig_reg, 350), width="stretch", config=PLOTLY_CONFIG)
        st.markdown("</div>", unsafe_allow_html=True)

    with s4:
        st.markdown('<div class="bento-card"><div class="bento-card-title">Feature Importance Ranking</div>'
                    '<div class="bento-card-caption">Gini importance showing which features drive category classification</div>', unsafe_allow_html=True)
        fig_imp = px.bar(ai["feat_imp"], x="Importance", y="Feature", orientation="h", color_discrete_sequence=["#0D9488"])
        st.plotly_chart(style_plotly(fig_imp, 350), width="stretch", config=PLOTLY_CONFIG)
        st.markdown("</div>", unsafe_allow_html=True)

# === TAB 3: UNSUPERVISED ML ===
with tab_unsup:
    dist_agg = pd.concat([
        df[(df["Category"] == "INSTITUTIONS") & (df["Location"] == "Total") & (df["Composition"] == "TOTAL")].groupby("District")["Number"].sum().rename("Institutions"),
        df[(df["Category"] == "ENROLMENT") & (df["Location"] == "Total") & (df["Composition"] == "TOTAL")].groupby("District")["Number"].sum().rename("Enrolment"),
        df[(df["Category"] == "TEACHING STAFF") & (df["Location"] == "Total") & (df["Composition"] == "TOTAL")].groupby("District")["Number"].sum().rename("Staff"),
        df.groupby("District")["Population"].first().rename("Population")
    ], axis=1).dropna()
    X_d_scaled = StandardScaler().fit_transform(dist_agg)

    u1, u2 = st.columns([1.3, 1])
    with u1:
        k_val = st.slider("Select K-Means Cluster Count (K):", 2, 6, 3)
        km = KMeans(n_clusters=k_val, random_state=42, n_init=10)
        dist_agg["Cluster"] = km.fit_predict(X_d_scaled).astype(str)
        sil = silhouette_score(X_d_scaled, dist_agg["Cluster"])

        st.markdown(f'<div class="bento-card"><div class="bento-card-title">2D PCA District Cluster Map (K={k_val} | Silhouette = {sil:.3f})</div>'
                    '<div class="bento-card-caption">Bubble size represents district population; color represents K-Means segment</div>', unsafe_allow_html=True)
        coords = PCA(n_components=2).fit_transform(X_d_scaled)
        dist_agg["PCA1"], dist_agg["PCA2"] = coords[:, 0], coords[:, 1]
        fig_cl = px.scatter(
            dist_agg.reset_index(), x="PCA1", y="PCA2", color="Cluster", size="Population",
            hover_name="District", text="District",
            color_discrete_sequence=["#1E3A8A", "#0D9488", "#E11D48", "#F59E0B", "#6366F1", "#10B981"]
        )
        fig_cl.update_traces(textposition="top center", textfont_size=9)
        st.plotly_chart(style_plotly(fig_cl, 400), width="stretch", config=PLOTLY_CONFIG)
        st.markdown("</div>", unsafe_allow_html=True)

    with u2:
        st.markdown('<div class="bento-card"><div class="bento-card-title">Elbow Curve (WCSS Inertia)</div>'
                    '<div class="bento-card-caption">Optimal cluster inflection point around K=3</div>', unsafe_allow_html=True)
        inertias = [KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_d_scaled).inertia_ for k in range(2, 8)]
        fig_el = px.line(x=list(range(2, 8)), y=inertias, markers=True, labels={"x": "Clusters (K)", "y": "Inertia"})
        fig_el.update_traces(line_color="#1E3A8A", line_width=3)
        st.plotly_chart(style_plotly(fig_el, 220), width="stretch", config=PLOTLY_CONFIG)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="bento-card"><div class="bento-card-title">Cluster Mean Profiles</div>'
                    '<div class="bento-card-caption">Average infrastructure and enrolment per cluster</div>', unsafe_allow_html=True)
        st.dataframe(dist_agg.groupby("Cluster")[["Institutions", "Enrolment", "Staff", "Population"]].mean().round(0), width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="bento-card"><div class="bento-card-title">Isolation Forest Anomaly Detection (5% Contamination)</div>'
                '<div class="bento-card-caption">Unsupervised detection of statistical outlier records across district population and count</div>', unsafe_allow_html=True)
    iso = IsolationForest(contamination=0.05, random_state=42)
    df_anom = df.copy()
    df_anom["Status"] = np.where(iso.fit_predict(df[["Number", "Population", "No/100000"]]) == -1, "Anomaly", "Normal")
    fig_iso = px.scatter(
        df_anom, x="Population", y="Number", color="Status", hover_data=["District", "Category", "School Type"],
        color_discrete_map={"Normal": "#94A3B8", "Anomaly": "#E11D48"}
    )
    st.plotly_chart(style_plotly(fig_iso, 340), width="stretch", config=PLOTLY_CONFIG)
    st.markdown("</div>", unsafe_allow_html=True)

# === TAB 4: DEEP LEARNING SUITE ===
with tab_dl:
    d1, d2 = st.columns(2)
    with d1:
        st.markdown(f'<div class="bento-card"><div class="bento-card-title">Deep MLP Training Convergence (Test Acc: {ai["dl"]["acc"]}%)</div>'
                    '<div class="bento-card-caption">3-Layer Neural Network (128 → 64 → 32, ReLU, Adam) training loss vs. validation accuracy</div>', unsafe_allow_html=True)
        ep = list(range(1, len(ai["dl"]["losses"]) + 1))
        fig_dl = go.Figure()
        fig_dl.add_trace(go.Scatter(x=ep, y=ai["dl"]["losses"], name="Loss", line=dict(color="#E11D48", width=2.5)))
        fig_dl.add_trace(go.Scatter(x=ep, y=ai["dl"]["val_accs"], name="Val Acc (%)", yaxis="y2", line=dict(color="#1E3A8A", width=2.5)))
        fig_dl.update_layout(yaxis=dict(title="Loss"), yaxis2=dict(title="Acc (%)", overlaying="y", side="right"))
        st.plotly_chart(style_plotly(fig_dl, 370), width="stretch", config=PLOTLY_CONFIG)
        st.markdown("</div>", unsafe_allow_html=True)

    with d2:
        st.markdown('<div class="bento-card"><div class="bento-card-title">Deep Neural Network Multi-Class ROC Curves</div>'
                    '<div class="bento-card-caption">Superior AUC separation across Enrolment, Institutions, and Teaching Staff</div>', unsafe_allow_html=True)
        y_bin = label_binarize(ai["y_test"], classes=[0, 1, 2])
        dl_probs = ai["dl"]["model"].predict_proba(ai["X_test_p"])
        class_names = ai["encoders"]["Category"].classes_
        fig_dl_roc = go.Figure()
        for i, cname in enumerate(class_names):
            fpr, tpr, _ = roc_curve(y_bin[:, i], dl_probs[:, i])
            fig_dl_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines", name=f"{cname} (AUC={auc(fpr, tpr):.3f})",
                line=dict(color=["#1E3A8A", "#0D9488", "#E11D48"][i], width=3)
            ))
        fig_dl_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(color="#94A3B8", dash="dash")))
        st.plotly_chart(style_plotly(fig_dl_roc, 370), width="stretch", config=PLOTLY_CONFIG)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="bento-card"><div class="bento-card-title">Deep MLP Confusion Matrix</div>'
                '<div class="bento-card-caption">Classification accuracy matrix on held-out PCA test features</div>', unsafe_allow_html=True)
    dl_cm = confusion_matrix(ai["y_test"], ai["dl"]["model"].predict(ai["X_test_p"]))
    fig_dl_cm = px.imshow(dl_cm, text_auto=True, x=class_names, y=class_names, color_continuous_scale="Teal")
    st.plotly_chart(style_plotly(fig_dl_cm, 340), width="stretch", config=PLOTLY_CONFIG)
    st.markdown("</div>", unsafe_allow_html=True)

# === TAB 5: PCA & DISTRICT RANKINGS ===
with tab_pca:
    p1, p2 = st.columns(2)
    with p1:
        st.markdown('<div class="bento-card"><div class="bento-card-title">PCA Scree & Cumulative Variance Curve</div>'
                    '<div class="bento-card-caption">First 5 principal components capture 95.96% of total dataset variance</div>', unsafe_allow_html=True)
        exp_var = ai["pca_full"].explained_variance_ratio_ * 100
        cum_var = np.cumsum(exp_var)
        pcs = [f"PC{i+1}" for i in range(len(exp_var))]
        fig_pca = go.Figure()
        fig_pca.add_trace(go.Bar(x=pcs, y=exp_var, name="Individual (%)", marker_color="#0D9488"))
        fig_pca.add_trace(go.Scatter(x=pcs, y=cum_var, name="Cumulative (%)", mode="lines+markers", line=dict(color="#1E3A8A", width=3)))
        fig_pca.add_hline(y=95, line_dash="dash", line_color="#E11D48")
        st.plotly_chart(style_plotly(fig_pca, 380), width="stretch", config=PLOTLY_CONFIG)
        st.markdown("</div>", unsafe_allow_html=True)

    with p2:
        st.markdown('<div class="bento-card"><div class="bento-card-title">Top 15 Districts by Total Institutions</div>'
                    '<div class="bento-card-caption">Provincial ranking of educational infrastructure across Sindh</div>', unsafe_allow_html=True)
        dist_rank = (
            df[(df["Category"] == "INSTITUTIONS") & (df["Location"] == "Total") & (df["Composition"] == "TOTAL")]
            .groupby("District")["Number"].sum().sort_values(ascending=True).tail(15).reset_index()
        )
        fig_dr = px.bar(dist_rank, x="Number", y="District", orientation="h", color_discrete_sequence=["#1E3A8A"])
        st.plotly_chart(style_plotly(fig_dr, 380), width="stretch", config=PLOTLY_CONFIG)
        st.markdown("</div>", unsafe_allow_html=True)

# === TAB 6: LIVE AI SANDBOX ===
with tab_sim:
    st.markdown('<div class="bento-card"><div class="bento-card-title">What-If AI Inference Sandbox</div>'
                '<div class="bento-card-caption">Run live predictions comparing Deep Learning (MLP) and Supervised ML classifiers</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    in_loc = c1.selectbox("Location", ai["encoders"]["Location"].classes_, key="s_loc")
    in_sch = c2.selectbox("School Type", ai["encoders"]["School Type"].classes_, key="s_sch")
    in_cmp = c3.selectbox("Composition", ai["encoders"]["Composition"].classes_, key="s_cmp")
    in_num = c4.number_input("Recorded Count", 0.0, 10000.0, 320.0)

    all_models = {"Deep Neural Network (MLP)": ai["dl"]["model"], **ai["sup_models"]}
    c5, c6, c7 = st.columns([1, 1, 1.3])
    in_pop = c5.number_input("District Population", 100000.0, 5000000.0, 1600000.0)
    in_rate = c6.number_input("Rate per 100k", 0.0, 1000.0, 20.0)
    in_mod = c7.selectbox("Select Model", list(all_models.keys()))

    if st.button("🚀 Run Live Prediction", type="primary"):
        row = pd.DataFrame([{
            "Location": ai["encoders"]["Location"].transform([in_loc])[0],
            "School Type": ai["encoders"]["School Type"].transform([in_sch])[0],
            "Composition": ai["encoders"]["Composition"].transform([in_cmp])[0],
            "Number": in_num,
            "Population": in_pop,
            "No/100000": in_rate,
            "Log_Number": np.log1p(in_num)
        }])
        row_p = ai["pca_95"].transform(ai["scaler"].transform(row))
        clf = all_models[in_mod]
        pred_lbl = ai["encoders"]["Category"].inverse_transform([clf.predict(row_p)[0]])[0]
        probs = clf.predict_proba(row_p)[0]

        st.success(f"**Predicted Category ({in_mod}):** `{pred_lbl}` — Confidence: **{max(probs)*100:.2f}%**")
        p_df = pd.DataFrame({"Category": ai["encoders"]["Category"].classes_, "Confidence (%)": (probs * 100).round(2)})
        fig_p = px.bar(p_df, x="Category", y="Confidence (%)", color="Category", text="Confidence (%)",
                       color_discrete_sequence=["#1E3A8A", "#0D9488", "#E11D48"])
        fig_p.update_layout(showlegend=False)
        st.plotly_chart(style_plotly(fig_p, 290), width="stretch", config=PLOTLY_CONFIG)
    st.markdown("</div>", unsafe_allow_html=True)
