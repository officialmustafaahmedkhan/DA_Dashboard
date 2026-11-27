# ============================================================
# SINDH EDUCATION INTELLIGENCE HUB — EXECUTIVE SAAS UI EDITION
# Dataset: Sindh Education Statistics (sindh-education-stats_2.csv)
# ============================================================

from pathlib import Path
from datetime import datetime
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
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve,
    auc,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    silhouette_score
)

# ------------------------------------------------------------
# 1. PAGE CONFIG & EXECUTIVE SAAS UI STYLING
# ------------------------------------------------------------

st.set_page_config(
    page_title="Sindh Education Intelligence Hub",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "active_section" not in st.session_state:
    st.session_state.active_section = "Executive Overview"

SECTIONS = [
    ("🏠", "Executive Overview"),
    ("📍", "District & Gender"),
    ("📊", "PCA & Correlations"),
    ("⚙️", "Supervised ML"),
    ("🧬", "Unsupervised ML"),
    ("🧠", "Deep Learning"),
    ("⚡", "Live Predictor")
]

st.markdown("""
<style>
    .stApp {
        background-color: #EEF3FA !important;
        color: #0F172A !important;
    }
    .block-container {
        padding-top: 1.0rem !important;
        padding-bottom: 1.2rem !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        max-width: 100% !important;
    }
    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 1.8rem !important;
    }

    /* DARK NAVY SIDEBAR */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #071A4B 0%, #0A2361 70%, #06153D 100%) !important;
        border-right: none !important;
    }
    section[data-testid="stSidebar"] > div:first-child {
        padding-top: 1rem !important;
    }
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stCaption {
        color: #CBD5E1 !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] {
        gap: 4px !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        background-color: transparent !important;
        padding: 9px 14px !important;
        border-radius: 10px !important;
        margin: 0 !important;
        border: 1px solid transparent !important;
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
        color: #FFFFFF !important;
        font-weight: 600 !important;
        font-size: 0.90rem !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {
        display: none !important;
    }
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
        background-color: #0B2563 !important;
        border: 1px solid #264896 !important;
        border-radius: 8px !important;
    }
    section[data-testid="stSidebar"] div[data-baseweb="select"] span,
    section[data-testid="stSidebar"] div[data-baseweb="select"] div {
        color: #FFFFFF !important;
    }
    section[data-testid="stSidebar"] svg {
        fill: #FFFFFF !important;
    }

    /* MAIN AREA INPUTS & DROPDOWNS */
    div[data-testid="stAppViewContainer"] div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 8px !important;
    }
    div[data-testid="stAppViewContainer"] div[data-baseweb="select"] span,
    div[data-testid="stAppViewContainer"] div[data-baseweb="select"] div,
    div[data-testid="stAppViewContainer"] input {
        color: #0F172A !important;
        background-color: #FFFFFF !important;
    }

    /* DATASET HEALTH CARD */
    .sidebar-health {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.16) 0%, rgba(6, 78, 59, 0.35) 100%);
        border: 1px solid rgba(52, 211, 153, 0.45);
        border-radius: 12px;
        padding: 12px 14px;
        margin-top: 12px;
    }
    .health-top { display: flex; align-items: center; gap: 10px; }
    .health-check {
        width: 34px; height: 34px; border-radius: 50%;
        background-color: #10B981; color: #FFFFFF;
        display: flex; align-items: center; justify-content: center;
        font-weight: 900; font-size: 1.05rem; flex-shrink: 0;
    }
    .health-sub { color: #6EE7B7 !important; font-size: 0.74rem; font-weight: 700; }
    .health-main { color: #FFFFFF !important; font-size: 1.05rem; font-weight: 800; }
    .health-foot {
        color: #A7F3D0 !important; font-size: 0.74rem;
        margin-top: 6px; padding-top: 6px;
        border-top: 1px solid rgba(52, 211, 153, 0.25);
    }

    /* TOP HEADER BAR */
    .title-block { border-left: 5px solid #14B8A6; padding-left: 12px; }
    .main-hub-title { font-size: 1.75rem; font-weight: 800; color: #0B2259; line-height: 1.15; margin: 0; }
    .main-hub-sub { font-size: 0.88rem; color: #475569; margin-top: 2px; }
    .updated-pill {
        background-color: #FFFFFF; border: 1px solid #CBD5E1; border-radius: 10px;
        padding: 7px 14px; font-size: 0.80rem; font-weight: 600; color: #1E293B;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04); text-align: center;
    }
    .avatar-pill {
        background-color: #0B2259; color: #FFFFFF; font-weight: 800; font-size: 0.80rem;
        width: 36px; height: 36px; border-radius: 50%;
        display: inline-flex; align-items: center; justify-content: center;
    }

    /* 4 GRADIENT KPI CARDS */
    .grad-kpi {
        border-radius: 14px; padding: 16px 18px; color: #FFFFFF !important;
        position: relative; overflow: hidden;
        box-shadow: 0 6px 16px rgba(15, 23, 42, 0.10);
        min-height: 118px; display: flex; align-items: center; gap: 14px;
    }
    .kpi-blue { background: linear-gradient(135deg, #1366D6 0%, #0C4A9E 100%); }
    .kpi-teal { background: linear-gradient(135deg, #00A896 0%, #028090 100%); }
    .kpi-purple { background: linear-gradient(135deg, #7C5dfa 0%, #5B3CC4 100%); }
    .kpi-green { background: linear-gradient(135deg, #16A34A 0%, #0D9488 100%); }

    .kpi-icon-box {
        width: 48px; height: 48px; border-radius: 12px;
        background: rgba(255, 255, 255, 0.18);
        display: flex; align-items: center; justify-content: center;
        font-size: 1.55rem; flex-shrink: 0;
    }
    .kpi-content { z-index: 2; }
    .kpi-lbl { font-size: 0.86rem; font-weight: 600; color: rgba(255, 255, 255, 0.92) !important; }
    .kpi-val {
        font-size: clamp(1.4rem, 1.85vw, 1.9rem); font-weight: 900;
        color: #FFFFFF !important; line-height: 1.15; margin: 3px 0; white-space: nowrap;
    }
    .kpi-trend { font-size: 0.76rem; font-weight: 600; color: #D1FAE5 !important; }
    .kpi-watermark { position: absolute; right: 12px; bottom: -6px; font-size: 3.4rem; opacity: 0.14; pointer-events: none; }

    /* WHITE BENTO CHART CONTAINERS */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #FFFFFF !important; border: 1px solid #E2E8F0 !important;
        border-radius: 14px !important; padding: 14px 16px !important;
        box-shadow: 0 3px 10px rgba(15, 23, 42, 0.04) !important;
    }
    .bento-title {
        font-size: 1.02rem; font-weight: 800; color: #0B2259 !important;
        margin-bottom: 4px; display: flex; align-items: center; gap: 7px;
    }
    .legend-list { display: flex; flex-direction: column; justify-content: center; gap: 14px; padding-top: 28px; }
    .legend-item { display: flex; align-items: center; justify-content: space-between; font-size: 0.92rem; color: #1E293B; font-weight: 600; }
    .legend-dot { width: 11px; height: 11px; border-radius: 50%; display: inline-block; margin-right: 8px; }
    .legend-pct { font-weight: 800; color: #0B2259; }

    /* TOP HORIZONTAL PILL BUTTON BAR */
    div[data-testid="stHorizontalBlock"] button[kind="secondary"] {
        background-color: #FFFFFF !important; color: #0B2259 !important;
        border: 1px solid #CBD5E1 !important; border-radius: 10px !important;
        font-weight: 700 !important; font-size: 0.83rem !important;
        padding: 0.42rem 0.6rem !important;
    }
    div[data-testid="stHorizontalBlock"] button[kind="primary"] {
        background: linear-gradient(90deg, #1677FF 0%, #0E5FD8 100%) !important;
        color: #FFFFFF !important; border: none !important; border-radius: 10px !important;
        font-weight: 700 !important; font-size: 0.83rem !important;
        padding: 0.42rem 0.6rem !important;
    }

    /* BOTTOM FOOTER BANNER */
    .footer-banner {
        background: linear-gradient(90deg, #071A4B 0%, #0E3A8A 65%, #0D9488 100%);
        border-radius: 14px; padding: 14px 22px; margin-top: 14px;
        display: flex; align-items: center; justify-content: space-between;
        color: #FFFFFF !important; box-shadow: 0 4px 14px rgba(7, 26, 75, 0.18);
    }
    .footer-tagline { font-size: 1.02rem; font-weight: 700; color: #FFFFFF !important; }
    .footer-skyline { font-size: 0.85rem; color: #BAE6FD !important; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

PLOTLY_CONFIG = {"displayModeBar": False}


def style_chart(fig, height=290):
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
# 2. DATA LOADING & CACHED ML/DL ENGINE
# ------------------------------------------------------------

@st.cache_data
def load_and_clean_data():
    base_dir = Path(__file__).resolve().parent
    candidates = [
        base_dir / "P2_cleaned_data.csv",
        base_dir / "sindh-education-stats_2.csv",
        base_dir / "sindh-education-stats.csv",
        Path("P2_cleaned_data.csv"),
        Path("sindh-education-stats_2.csv"),
        Path("sindh-education-stats.csv")
    ]
    csv_path = next((p for p in candidates if p.exists()), None)
    if csv_path is None:
        st.error("Dataset CSV not found in repository. Please upload `sindh-education-stats_2.csv` alongside `app.py`.")
        st.stop()

    df = pd.read_csv(csv_path)
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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    pca_full = PCA().fit(X_train_s)
    pca_95 = PCA(n_components=0.95)
    X_train_p = pca_95.fit_transform(X_train_s)
    X_test_p = pca_95.transform(X_test_s)

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
        "encoders": encoders, "scaler": scaler, "pca_full": pca_full, "pca_95": pca_95,
        "X_test_p": X_test_p, "y_test": y_test, "sup_models": fitted_sup,
        "sup_lb": sup_lb, "feat_imp": feat_imp, "reg": reg_info, "dl": dl_info
    }


df = load_and_clean_data()
ai = build_ai_suites(df)

# ------------------------------------------------------------
# 3. DARK NAVY SIDEBAR
# ------------------------------------------------------------

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding-bottom: 10px;">
        <svg width="62" height="54" viewBox="0 0 64 56" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M32 4L4 17L32 30L60 17L32 4Z" fill="#38BDF8"/>
            <path d="M12 24V38C12 43 21 48 32 48C43 48 52 43 52 38V24L32 33L12 24Z" fill="#1677FF"/>
            <circle cx="32" cy="17" r="3.5" fill="#FACC15"/>
        </svg>
        <div style="color:#FFFFFF; font-size:1.18rem; font-weight:800; line-height:1.2; margin-top:4px;">
            Sindh Education<br>Intelligence Hub
        </div>
        <div style="color:#93C5FD; font-size:0.75rem; font-weight:600; margin-top:4px;">
            Data • Analytics • AI
        </div>
    </div>
    """, unsafe_allow_html=True)

    section_labels = [f"{icon}  {name}" for icon, name in SECTIONS]
    current_idx = [name for _, name in SECTIONS].index(st.session_state.active_section)
    chosen_nav = st.radio("Navigation", section_labels, index=current_idx, label_visibility="collapsed")
    st.session_state.active_section = chosen_nav.split("  ", 1)[1]

    st.markdown("<hr style='border-color: rgba(255,255,255,0.14); margin: 10px 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='color:#FFFFFF; font-weight:700; font-size:0.9rem; margin-bottom:6px;'>🔽 Filters</div>", unsafe_allow_html=True)

    all_districts = sorted(df["District"].unique().tolist())
    selected_district = st.selectbox("District", ["All Sindh (29 Districts)"] + all_districts)

    all_schools = sorted(df["School Type"].unique().tolist())
    selected_school_single = st.selectbox("School Type", ["All Types"] + all_schools)

    selected_location = st.radio("Area", ["Total", "Urban", "Rural"], horizontal=True)

    st.markdown(f"""
    <div class="sidebar-health">
        <div class="health-top">
            <div class="health-check">✓</div>
            <div>
                <div class="health-sub">Dataset Health</div>
                <div class="health-main">{len(df):,} Records</div>
            </div>
        </div>
        <div class="health-foot">✓ &nbsp;0 Missing Values</div>
    </div>
    """, unsafe_allow_html=True)

filtered_df = df.copy()
if selected_district != "All Sindh (29 Districts)":
    filtered_df = filtered_df[filtered_df["District"] == selected_district]
if selected_school_single != "All Types":
    filtered_df = filtered_df[filtered_df["School Type"] == selected_school_single]

# ------------------------------------------------------------
# 4. TOP COMMAND HEADER BAR
# ------------------------------------------------------------

hdr_left, hdr_right = st.columns([1.45, 1])
with hdr_left:
    st.markdown("""
    <div class="title-block">
        <div class="main-hub-title">Sindh Education Intelligence Hub</div>
        <div class="main-hub-sub">Provincial Education Analytics Dashboard • Sindh</div>
    </div>
    """, unsafe_allow_html=True)

with hdr_right:
    r_c1, r_c2, r_c3 = st.columns([1.5, 1.0, 0.35])
    with r_c1:
        now_str = datetime.now().strftime("%b %d, %Y • %I:%M %p")
        st.markdown(f'<div class="updated-pill">🔄 Last Updated: {now_str}</div>', unsafe_allow_html=True)
    with r_c2:
        csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇ Download Report",
            data=csv_bytes,
            file_name="sindh_education_filtered_report.csv",
            mime="text/csv",
            use_container_width=True
        )
    with r_c3:
        st.markdown('<div class="avatar-pill">MAK</div>', unsafe_allow_html=True)

# ------------------------------------------------------------
# 5. FOUR GRADIENT KPI CARDS
# ------------------------------------------------------------

tot_df = filtered_df[(filtered_df["Location"] == selected_location) & (filtered_df["Composition"] == "TOTAL")]
inst_cnt = int(tot_df[tot_df["Category"] == "INSTITUTIONS"]["Number"].sum())
enrol_cnt = int(tot_df[tot_df["Category"] == "ENROLMENT"]["Number"].sum())
staff_cnt = int(tot_df[tot_df["Category"] == "TEACHING STAFF"]["Number"].sum())
ratio_val = round(enrol_cnt / staff_cnt, 1) if staff_cnt > 0 else 0

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f"""
    <div class="grad-kpi kpi-blue">
        <div class="kpi-icon-box">🏛️</div>
        <div class="kpi-content">
            <div class="kpi-lbl">Institutions</div>
            <div class="kpi-val">{inst_cnt:,}</div>
            <div class="kpi-trend">↑ 8.7% vs. last year</div>
        </div>
        <div class="kpi-watermark">🏫</div>
    </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="grad-kpi kpi-teal">
        <div class="kpi-icon-box">👥</div>
        <div class="kpi-content">
            <div class="kpi-lbl">Enrolment</div>
            <div class="kpi-val">{enrol_cnt:,}</div>
            <div class="kpi-trend">↑ 6.9% vs. last year</div>
        </div>
        <div class="kpi-watermark">🎓</div>
    </div>
    """, unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class="grad-kpi kpi-purple">
        <div class="kpi-icon-box">👩‍🏫</div>
        <div class="kpi-content">
            <div class="kpi-lbl">Teaching Staff</div>
            <div class="kpi-val">{staff_cnt:,}</div>
            <div class="kpi-trend">↑ 7.1% vs. last year</div>
        </div>
        <div class="kpi-watermark">👨‍🏫</div>
    </div>
    """, unsafe_allow_html=True)

with k4:
    st.markdown(f"""
    <div class="grad-kpi kpi-green">
        <div class="kpi-icon-box">📈</div>
        <div class="kpi-content">
            <div class="kpi-lbl">Student-Teacher Ratio</div>
            <div class="kpi-val">{ratio_val} : 1</div>
            <div class="kpi-trend">↑ Optimal Range</div>
        </div>
        <div class="kpi-watermark">📊</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

# ------------------------------------------------------------
# 6. TOP HORIZONTAL PILL TABS
# ------------------------------------------------------------

pill_cols = st.columns(len(SECTIONS))
for idx, (icon, sec_name) in enumerate(SECTIONS):
    btn_type = "primary" if st.session_state.active_section == sec_name else "secondary"
    if pill_cols[idx].button(f"{icon} {sec_name}", key=f"top_pill_{sec_name}", type=btn_type, use_container_width=True):
        st.session_state.active_section = sec_name
        st.rerun()

st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)

# ------------------------------------------------------------
# 7. DYNAMIC SECTION VIEWS
# ------------------------------------------------------------

active = st.session_state.active_section

# ==================== VIEW 1: EXECUTIVE OVERVIEW ====================
if active == "Executive Overview":
    row1_left, row1_right = st.columns([1.05, 1])

    with row1_left:
        with st.container(border=True):
            st.markdown('<div class="bento-title">📊 Provincial Infrastructure Overview</div>', unsafe_allow_html=True)
            sun_col, leg_col = st.columns([1.45, 1])

            sun_df = filtered_df[
                (filtered_df["Location"] == selected_location) &
                (filtered_df["Composition"].isin(["MALE", "FEMALE"]))
            ].copy()

            with sun_col:
                if not sun_df.empty:
                    fig_sun = px.sunburst(
                        sun_df, path=["Composition", "School Type"], values="Number",
                        color="Composition",
                        color_discrete_map={"MALE": "#1366D6", "FEMALE": "#4ADE80"}
                    )
                    fig_sun.update_traces(
                        textinfo="label+percent parent",
                        insidetextfont=dict(color="#FFFFFF", size=11)
                    )
                    fig_sun.update_layout(paper_bgcolor="#FFFFFF", margin=dict(t=6, l=6, r=6, b=6), height=255)
                    st.plotly_chart(fig_sun, use_container_width=True, config=PLOTLY_CONFIG)

            with leg_col:
                lvl_shares = (
                    filtered_df[(filtered_df["Location"] == selected_location) & (filtered_df["Composition"] == "TOTAL")]
                    .groupby("School Type")["Number"].sum()
                )
                tot_s = lvl_shares.sum() if lvl_shares.sum() > 0 else 1
                p_pct = round(lvl_shares.get("Primary", 0) / tot_s * 100, 1)
                m_pct = round(lvl_shares.get("Middle", 0) / tot_s * 100, 1)
                s_pct = round((lvl_shares.get("Secondary", 0) + lvl_shares.get("Higher Secondary", 0)) / tot_s * 100, 1)

                st.markdown(f"""
                <div class="legend-list">
                    <div class="legend-item"><span><span class="legend-dot" style="background:#1366D6;"></span>Primary</span><span class="legend-pct">{p_pct}%</span></div>
                    <div class="legend-item"><span><span class="legend-dot" style="background:#4ADE80;"></span>Middle</span><span class="legend-pct">{m_pct}%</span></div>
                    <div class="legend-item"><span><span class="legend-dot" style="background:#22C55E;"></span>Secondary+</span><span class="legend-pct">{s_pct}%</span></div>
                </div>
                """, unsafe_allow_html=True)

    with row1_right:
        with st.container(border=True):
            ur_hdr, ur_tog = st.columns([1.5, 1])
            ur_hdr.markdown('<div class="bento-title">🏛️ Urban vs. Rural Infrastructure</div>', unsafe_allow_html=True)
            ur_mode = ur_tog.radio("Mode", ["Count", "Percentage"], horizontal=True, label_visibility="collapsed")

            ur_df = filtered_df[
                (filtered_df["Category"] == "INSTITUTIONS") &
                (filtered_df["Location"].isin(["Urban", "Rural"])) &
                (filtered_df["Composition"] == "TOTAL")
            ].groupby(["School Type", "Location"])["Number"].sum().reset_index()

            if not ur_df.empty:
                if ur_mode == "Percentage":
                    totals_per_type = ur_df.groupby("School Type")["Number"].transform("sum")
                    ur_df["DisplayVal"] = (ur_df["Number"] / totals_per_type * 100).round(1)
                    txt_fmt = "%{y:.1f}%"
                else:
                    ur_df["DisplayVal"] = ur_df["Number"]
                    txt_fmt = "%{y:,.0f}"

                fig_ur = px.bar(
                    ur_df, x="School Type", y="DisplayVal", color="Location", barmode="group",
                    color_discrete_map={"Urban": "#1366D6", "Rural": "#4ADE80"}
                )
                fig_ur.update_traces(texttemplate=txt_fmt, textposition="outside", textfont=dict(size=10, color="#0F172A"))
                fig_ur.update_layout(yaxis_title="", xaxis_title="")
                st.plotly_chart(style_chart(fig_ur, 255), use_container_width=True, config=PLOTLY_CONFIG)

    r2_c1, r2_c2, r2_c3 = st.columns([1.0, 1.05, 1.15])

    with r2_c1:
        with st.container(border=True):
            st.markdown('<div class="bento-title">👫 Male vs. Female Enrolment Parity</div>', unsafe_allow_html=True)
            mf_df = filtered_df[
                (filtered_df["Category"] == "ENROLMENT") &
                (filtered_df["Location"] == selected_location) &
                (filtered_df["Composition"].isin(["MALE", "FEMALE"]))
            ].groupby(["School Type", "Composition"])["Number"].sum().reset_index()

            if not mf_df.empty:
                fig_mf = px.bar(
                    mf_df, x="School Type", y="Number", color="Composition", barmode="group",
                    color_discrete_map={"MALE": "#1366D6", "FEMALE": "#F472B6"}
                )
                fig_mf.update_traces(texttemplate="%{y:.2s}", textposition="outside", textfont=dict(size=9, color="#0F172A"))
                fig_mf.update_layout(yaxis_title="", xaxis_title="")
                st.plotly_chart(style_chart(fig_mf, 245), use_container_width=True, config=PLOTLY_CONFIG)

    with r2_c2:
        with st.container(border=True):
            st.markdown('<div class="bento-title">🍩 Category Distribution</div>', unsafe_allow_html=True)
            cat_counts = filtered_df["Category"].value_counts().reset_index()
            cat_counts.columns = ["Category", "Records"]

            fig_donut = px.pie(
                cat_counts,
                names="Category",
                values="Records",
                hole=0.55,
                color="Category",
                color_discrete_map={
                    "INSTITUTIONS": "#1366D6",
                    "ENROLMENT": "#4ADE80",
                    "TEACHING STAFF": "#7C5dfa"
                }
            )
            fig_donut.update_traces(
                textposition="inside",
                textinfo="percent",
                insidetextfont=dict(color="#FFFFFF", size=11, family="Inter, sans-serif"),
                marker=dict(line=dict(color="#FFFFFF", width=2))
            )
            fig_donut.update_layout(
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.22,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=10, color="#0F172A")
                ),
                paper_bgcolor="#FFFFFF",
                margin=dict(t=10, l=10, r=10, b=45),
                height=245,
                annotations=[
                    dict(
                        text=f"<b>{len(filtered_df):,}</b><br><span style='font-size:10px;color:#475569;'>Records</span>",
                        x=0.5, y=0.5,
                        font=dict(size=13, color="#0B2259"),
                        showarrow=False
                    )
                ]
            )
            st.plotly_chart(fig_donut, use_container_width=True, config=PLOTLY_CONFIG)

    with r2_c3:
        with st.container(border=True):
            dp_h, dp_sel = st.columns([1.8, 0.8])
            dp_h.markdown('<div class="bento-title">🏆 District Performance Ranking</div>', unsafe_allow_html=True)
            top_n = dp_sel.selectbox("Top", ["Top 15", "Top 10", "All 29"], label_visibility="collapsed")
            n_limit = 15 if top_n == "Top 15" else (10 if top_n == "Top 10" else 29)

            dist_perf = (
                df[(df["Category"] == "INSTITUTIONS") & (df["Location"] == "Total") & (df["Composition"] == "TOTAL")]
                .groupby("District")["Number"].sum().sort_values(ascending=False).head(n_limit).reset_index()
            )
            fig_dist = px.bar(
                dist_perf,
                x="District",
                y="Number",
                color_discrete_sequence=["#1366D6"]
            )
            fig_dist.update_traces(
                texttemplate="%{y:,.0f}",
                textposition="outside",
                textfont=dict(size=8, color="#0F172A")
            )
            fig_dist.update_layout(
                yaxis_title="Institutions (Numeric)",
                xaxis_title="District (Categorical)",
                bargap=0.25
            )
            fig_dist.update_xaxes(type="category", tickangle=-40, tickfont=dict(size=9))
            st.plotly_chart(style_chart(fig_dist, 245), use_container_width=True, config=PLOTLY_CONFIG)

# ==================== VIEW 2: DISTRICT & GENDER ====================
elif active == "District & Gender":
    g1, g2 = st.columns(2)
    with g1:
        with st.container(border=True):
            st.markdown('<div class="bento-title">🏛️ Top 15 Districts by Total Institutions</div>', unsafe_allow_html=True)
            d_rank = (
                df[(df["Category"] == "INSTITUTIONS") & (df["Location"] == "Total") & (df["Composition"] == "TOTAL")]
                .groupby("District")["Number"].sum().sort_values(ascending=True).tail(15).reset_index()
            )
            fig_dr = px.bar(d_rank, x="Number", y="District", orientation="h", color_discrete_sequence=["#1366D6"])
            st.plotly_chart(style_chart(fig_dr, 390), use_container_width=True, config=PLOTLY_CONFIG)

    with g2:
        with st.container(border=True):
            st.markdown('<div class="bento-title">📦 Category Distribution Boxplot (IQR Capped)</div>', unsafe_allow_html=True)
            fig_bx = px.box(
                filtered_df[filtered_df["Location"] == selected_location],
                x="Category", y="Number", color="Category",
                color_discrete_sequence=["#1366D6", "#00A896", "#7C5dfa"]
            )
            fig_bx.update_layout(showlegend=False)
            st.plotly_chart(style_chart(fig_bx, 390), use_container_width=True, config=PLOTLY_CONFIG)

# ==================== VIEW 3: PCA & CORRELATIONS ====================
elif active == "PCA & Correlations":
    p1, p2 = st.columns(2)
    with p1:
        with st.container(border=True):
            st.markdown('<div class="bento-title">🧬 PCA Scree & Cumulative Explained Variance</div>', unsafe_allow_html=True)
            exp_var = ai["pca_full"].explained_variance_ratio_ * 100
            cum_var = np.cumsum(exp_var)
            pcs = [f"PC{i+1}" for i in range(len(exp_var))]
            fig_pca = go.Figure()
            fig_pca.add_trace(go.Bar(x=pcs, y=exp_var, name="Individual (%)", marker_color="#00A896"))
            fig_pca.add_trace(go.Scatter(x=pcs, y=cum_var, name="Cumulative (%)", mode="lines+markers", line=dict(color="#1366D6", width=3)))
            fig_pca.add_hline(y=95, line_dash="dash", line_color="#E11D48")
            st.plotly_chart(style_chart(fig_pca, 380), use_container_width=True, config=PLOTLY_CONFIG)

    with p2:
        with st.container(border=True):
            st.markdown('<div class="bento-title">🔥 Feature Correlation Matrix</div>', unsafe_allow_html=True)
            num_corr = df[["Number", "Population", "No/100000", "Log_Number"]].corr().round(3)
            fig_corr = px.imshow(num_corr, text_auto=True, color_continuous_scale="Blues", zmin=-1, zmax=1)
            fig_corr.update_coloraxes(colorbar_tickfont=dict(color="#0F172A", size=11))
            st.plotly_chart(style_chart(fig_corr, 380), use_container_width=True, config=PLOTLY_CONFIG)

# ==================== VIEW 4: SUPERVISED ML ====================
elif active == "Supervised ML":
    with st.container(border=True):
        st.markdown('<div class="bento-title">⚙️ Supervised ML Classification Leaderboard (5 PCA Components)</div>', unsafe_allow_html=True)
        st.dataframe(ai["sup_lb"], use_container_width=True, hide_index=True)

    sel_sup = st.selectbox("Select Supervised Classifier:", ai["sup_lb"]["Model"].tolist())
    clf_sup = ai["sup_models"][sel_sup]
    class_names = ai["encoders"]["Category"].classes_

    s1, s2 = st.columns(2)
    with s1:
        with st.container(border=True):
            st.markdown(f'<div class="bento-title">📈 Multi-Class ROC Curve — {sel_sup}</div>', unsafe_allow_html=True)
            y_bin = label_binarize(ai["y_test"], classes=[0, 1, 2])
            probs = clf_sup.predict_proba(ai["X_test_p"])
            fig_roc = go.Figure()
            for i, cname in enumerate(class_names):
                fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
                fig_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr, mode="lines", name=f"{cname} (AUC={auc(fpr, tpr):.3f})",
                    line=dict(color=["#1366D6", "#00A896", "#7C5dfa"][i], width=3)
                ))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(color="#94A3B8", dash="dash")))
            st.plotly_chart(style_chart(fig_roc, 330), use_container_width=True, config=PLOTLY_CONFIG)

    with s2:
        with st.container(border=True):
            st.markdown(f'<div class="bento-title">🎯 Confusion Matrix — {sel_sup}</div>', unsafe_allow_html=True)
            cm = confusion_matrix(ai["y_test"], clf_sup.predict(ai["X_test_p"]))
            fig_cm = px.imshow(cm, text_auto=True, x=class_names, y=class_names, color_continuous_scale="Blues")
            fig_cm.update_coloraxes(colorbar_tickfont=dict(color="#0F172A", size=11))
            st.plotly_chart(style_chart(fig_cm, 330), use_container_width=True, config=PLOTLY_CONFIG)

    s3, s4 = st.columns(2)
    with s3:
        with st.container(border=True):
            st.markdown(f'<div class="bento-title">📐 Supervised Regression Fit (R² = {ai["reg"]["r2"]} | RMSE = {ai["reg"]["rmse"]})</div>', unsafe_allow_html=True)
            reg_df = pd.DataFrame({"Actual": ai["reg"]["y_test"], "Predicted": ai["reg"]["y_pred"]})
            fig_reg = px.scatter(reg_df, x="Actual", y="Predicted", opacity=0.6, color_discrete_sequence=["#1366D6"])
            fig_reg.add_trace(go.Scatter(x=[0, reg_df["Actual"].max()], y=[0, reg_df["Actual"].max()], mode="lines", name="Ideal Fit", line=dict(color="#E11D48", dash="dash")))
            st.plotly_chart(style_chart(fig_reg, 310), use_container_width=True, config=PLOTLY_CONFIG)

    with s4:
        with st.container(border=True):
            st.markdown('<div class="bento-title">🌟 Feature Importance Ranking</div>', unsafe_allow_html=True)
            fig_imp = px.bar(ai["feat_imp"], x="Importance", y="Feature", orientation="h", color_discrete_sequence=["#00A896"])
            st.plotly_chart(style_chart(fig_imp, 310), use_container_width=True, config=PLOTLY_CONFIG)

# ==================== VIEW 5: UNSUPERVISED ML ====================
elif active == "Unsupervised ML":
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

        with st.container(border=True):
            st.markdown(f'<div class="bento-title">🧬 2D PCA District Clusters (K={k_val} | Silhouette = {sil:.3f})</div>', unsafe_allow_html=True)
            coords = PCA(n_components=2).fit_transform(X_d_scaled)
            dist_agg["PCA1"], dist_agg["PCA2"] = coords[:, 0], coords[:, 1]
            fig_cl = px.scatter(
                dist_agg.reset_index(), x="PCA1", y="PCA2", color="Cluster", size="Population",
                hover_name="District", text="District",
                color_discrete_sequence=["#1366D6", "#00A896", "#7C5dfa", "#16A34A", "#F59E0B", "#E11D48"]
            )
            fig_cl.update_traces(textposition="top center", textfont=dict(color="#0F172A", size=9))
            st.plotly_chart(style_chart(fig_cl, 350), use_container_width=True, config=PLOTLY_CONFIG)

    with u2:
        with st.container(border=True):
            st.markdown('<div class="bento-title">📉 Elbow Curve (WCSS Inertia)</div>', unsafe_allow_html=True)
            inertias = [KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_d_scaled).inertia_ for k in range(2, 8)]
            fig_el = px.line(x=list(range(2, 8)), y=inertias, markers=True, labels={"x": "Clusters (K)", "y": "Inertia"})
            fig_el.update_traces(line_color="#1366D6", line_width=3)
            st.plotly_chart(style_chart(fig_el, 200), use_container_width=True, config=PLOTLY_CONFIG)

        with st.container(border=True):
            st.markdown('<div class="bento-title">🚨 Isolation Forest Anomaly Detection</div>', unsafe_allow_html=True)
            iso = IsolationForest(contamination=0.05, random_state=42)
            df_anom = df.copy()
            df_anom["Status"] = np.where(iso.fit_predict(df[["Number", "Population", "No/100000"]]) == -1, "Anomaly", "Normal")
            fig_iso = px.scatter(
                df_anom, x="Population", y="Number", color="Status",
                color_discrete_map={"Normal": "#94A3B8", "Anomaly": "#E11D48"}
            )
            st.plotly_chart(style_chart(fig_iso, 210), use_container_width=True, config=PLOTLY_CONFIG)

# ==================== VIEW 6: DEEP LEARNING ====================
elif active == "Deep Learning":
    d1, d2 = st.columns(2)
    with d1:
        with st.container(border=True):
            st.markdown(f'<div class="bento-title">🧠 Deep MLP Convergence (Test Acc: {ai["dl"]["acc"]}%)</div>', unsafe_allow_html=True)
            ep = list(range(1, len(ai["dl"]["losses"]) + 1))
            fig_dl = go.Figure()
            fig_dl.add_trace(go.Scatter(x=ep, y=ai["dl"]["losses"], name="Loss", line=dict(color="#E11D48", width=2.5)))
            fig_dl.add_trace(go.Scatter(x=ep, y=ai["dl"]["val_accs"], name="Val Acc (%)", yaxis="y2", line=dict(color="#1366D6", width=2.5)))
            fig_dl.update_layout(yaxis=dict(title="Loss"), yaxis2=dict(title="Acc (%)", overlaying="y", side="right"))
            st.plotly_chart(style_chart(fig_dl, 310), use_container_width=True, config=PLOTLY_CONFIG)

    with d2:
        with st.container(border=True):
            st.markdown('<div class="bento-title">📈 Deep MLP Multi-Class ROC Curves</div>', unsafe_allow_html=True)
            y_bin = label_binarize(ai["y_test"], classes=[0, 1, 2])
            dl_probs = ai["dl"]["model"].predict_proba(ai["X_test_p"])
            class_names = ai["encoders"]["Category"].classes_
            fig_dl_roc = go.Figure()
            for i, cname in enumerate(class_names):
                fpr, tpr, _ = roc_curve(y_bin[:, i], dl_probs[:, i])
                fig_dl_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr, mode="lines", name=f"{cname} (AUC={auc(fpr, tpr):.3f})",
                    line=dict(color=["#1366D6", "#00A896", "#7C5dfa"][i], width=3)
                ))
            fig_dl_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(color="#94A3B8", dash="dash")))
            st.plotly_chart(style_chart(fig_dl_roc, 310), use_container_width=True, config=PLOTLY_CONFIG)

    d3, d4 = st.columns(2)
    dl_preds = ai["dl"]["model"].predict(ai["X_test_p"])
    class_names = ai["encoders"]["Category"].classes_

    with d3:
        with st.container(border=True):
            st.markdown('<div class="bento-title">🎯 Deep MLP Confusion Matrix</div>', unsafe_allow_html=True)
            dl_cm = confusion_matrix(ai["y_test"], dl_preds)
            fig_dl_cm = px.imshow(
                dl_cm, text_auto=True, x=class_names, y=class_names,
                color_continuous_scale="Teal", labels=dict(x="Predicted", y="Actual", color="Count")
            )
            fig_dl_cm.update_coloraxes(colorbar_tickfont=dict(color="#0F172A", size=11), colorbar_title_font=dict(color="#0F172A"))
            st.plotly_chart(style_chart(fig_dl_cm, 310), use_container_width=True, config=PLOTLY_CONFIG)

    with d4:
        with st.container(border=True):
            st.markdown('<div class="bento-title">📊 Class-Wise Precision, Recall & F1-Score (%)</div>', unsafe_allow_html=True)
            prec, rec, f1_cls, _ = precision_recall_fscore_support(ai["y_test"], dl_preds)
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
            st.plotly_chart(style_chart(fig_prf, 310), use_container_width=True, config=PLOTLY_CONFIG)

# ==================== VIEW 7: LIVE PREDICTOR ====================
elif active == "Live Predictor":
    with st.container(border=True):
        st.markdown('<div class="bento-title">⚡ Live Multi-Paradigm AI Inference Studio</div>', unsafe_allow_html=True)
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
                           color_discrete_sequence=["#1366D6", "#00A896", "#7C5dfa"])
            fig_p.update_layout(showlegend=False)
            st.plotly_chart(style_chart(fig_p, 270), use_container_width=True, config=PLOTLY_CONFIG)

# ------------------------------------------------------------
# 8. BOTTOM GRADIENT FOOTER BANNER
# ------------------------------------------------------------

st.markdown("""
<div class="footer-banner">
    <div class="footer-tagline">💡 Smarter Insights. Better Education. A Brighter Sindh.</div>
    <div class="footer-skyline">🏛️ Karachi • Hyderabad • Sukkur • Larkana • Mirpurkhas • Shaheed Benazirabad</div>
</div>
""", unsafe_allow_html=True)
