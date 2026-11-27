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
    page_title="Sindh Education Intelligence Hub",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
:root { --navy:#0b1f46; --blue:#2563eb; --teal:#0d9488; --purple:#7c3aed; --bg:#f3f7fc; }
.stApp { background: linear-gradient(135deg,#f7faff 0%,#eef4fb 100%); color:#14213d; }
[data-testid="stHeader"] { background:rgba(243,247,252,.8); }
.block-container { padding-top:1.25rem; padding-bottom:2rem; max-width:1600px; }
.main-header { font-size:2.15rem; font-weight:850; letter-spacing:-.8px; color:#102b59; margin:0 0 .2rem 0; }
.sub-header { font-size:1rem; color:#526581; margin-bottom:1rem; }
.hero-mark { border-left:5px solid #0d9488; padding-left:14px; margin-bottom:12px; }
.kpi-card { background:linear-gradient(145deg,#ffffff 0%,#f8fbff 100%); border:1px solid #d8e3f2; border-radius:16px; padding:16px 16px; box-shadow:0 4px 14px rgba(15,38,78,.07); text-align:left; min-height:106px; display:flex; flex-direction:column; justify-content:center; transition:transform .15s ease, box-shadow .15s ease; }
.kpi-card:hover { transform:translateY(-2px); box-shadow:0 8px 22px rgba(15,38,78,.11); }
.kpi-title { color:#526581 !important; font-size:.86rem; font-weight:750; margin-bottom:7px; white-space:nowrap; }
.kpi-value { color:#102b59 !important; font-size:1.55rem; font-weight:900; line-height:1.2; letter-spacing:-.5px; white-space:nowrap; }
[data-testid="stMetric"] { background:#fff !important; border:1px solid #d8e3f2 !important; padding:14px 16px !important; border-radius:15px !important; box-shadow:0 4px 12px rgba(15,38,78,.06); }
[data-testid="stMetricLabel"] p { color:#526581 !important; font-weight:700 !important; }
[data-testid="stMetricValue"] { color:#102b59 !important; font-weight:850 !important; }
section[data-testid="stSidebar"] { background:linear-gradient(180deg,#0b1f46 0%,#102c59 65%,#0b3855 100%); }
section[data-testid="stSidebar"] * { color:#f4f8ff; }
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { color:#dbeafe !important; font-weight:700; }
section[data-testid="stSidebar"] input, section[data-testid="stSidebar"] div[data-baseweb="select"] { color:#102b59 !important; }
.sidebar-brand { text-align:center; padding:10px 4px 16px; margin-bottom:10px; border-bottom:1px solid rgba(255,255,255,.16); }
.sidebar-brand .brand-icon { font-size:2.5rem; }
.sidebar-brand h2 { color:#fff; font-size:1.15rem; line-height:1.25; margin:.25rem 0; }
.sidebar-brand p { color:#c7d8f5; font-size:.78rem; margin:0; }
.stTabs [data-baseweb="tab-list"] { gap:5px; background:#e8eff8; border:1px solid #d6e2f2; border-radius:14px; padding:5px; }
.stTabs [data-baseweb="tab"] { background:transparent; border-radius:10px; padding:8px 12px; color:#425575; font-weight:700; }
.stTabs [aria-selected="true"] { background:#2563eb !important; color:#fff !important; }
.stTabs [data-baseweb="tab-highlight"] { background:transparent; }
[data-testid="stPlotlyChart"] { background:#fff; border:1px solid #e1e9f4; border-radius:14px; padding:5px; box-shadow:0 3px 12px rgba(15,38,78,.04); }
[data-testid="stDataFrame"] { border:1px solid #d8e3f2; border-radius:12px; overflow:hidden; }
.stButton button { border-radius:10px; font-weight:750; }
hr { border-color:#d9e4f2; }
h1,h2,h3 { color:#102b59 !important; letter-spacing:-.2px; }
@media (max-width:900px) { .main-header { font-size:1.65rem; } .kpi-value { font-size:1.25rem; } }
</style>
""", unsafe_allow_html=True)


def render_kpi_card(title, value):
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">{title}</div>
