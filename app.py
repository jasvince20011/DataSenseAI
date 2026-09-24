import streamlit as st
import pandas as pd
from google import genai
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from io import BytesIO
import zipfile
import time
from pathlib import Path
from pypdf import PdfReader
from google.genai import types

# Gemini connection
client = genai.Client(
    api_key=st.secrets["GEMINI_API_KEY"]
)

# ==========================================
# PDF REPORT GENERATOR
# ==========================================

def create_pdf_report(title, analysis_text, df):

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    story = []

    # Title
    story.append(
        Paragraph(
            title,
            styles["Title"]
        )
    )

    story.append(Spacer(1, 15))

    # Dataset summary
    story.append(
        Paragraph(
            f"<b>Dataset:</b> {df.shape[0]} rows × {df.shape[1]} columns",
            styles["Normal"]
        )
    )

    story.append(Spacer(1, 10))

    story.append(
        Paragraph(
            f"<b>Missing Values:</b> {int(df.isnull().sum().sum())}",
            styles["Normal"]
        )
    )

    story.append(Spacer(1, 10))

    story.append(
        Paragraph(
            f"<b>Duplicate Rows:</b> {int(df.duplicated().sum())}",
            styles["Normal"]
        )
    )

    story.append(Spacer(1, 20))

    # AI analysis
    story.append(
        Paragraph(
            "<b>AI-Generated Business Analysis</b>",
            styles["Heading2"]
        )
    )

    story.append(Spacer(1, 10))

    # Convert AI text into PDF paragraphs
    for line in analysis_text.split("\n"):

        line = line.strip()

        if line:

            safe_line = (
                line
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )

            story.append(
                Paragraph(
                    safe_line,
                    styles["Normal"]
                )
            )

            story.append(Spacer(1, 6))

    doc.build(story)

    buffer.seek(0)

    return buffer
# ==========================================
# DASHBOARD GENERATOR
# ==========================================

def build_dashboard(df, file_name):
    """Create an interactive KPI dashboard for the uploaded CSV."""
    import plotly.express as px

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    st.markdown(f"## 🎯 Dashboard — {file_name}")
    st.caption("Interactive KPI and business overview generated from the uploaded dataset.")

    # KPI calculations
    total_rows = len(df)
    total_columns = len(df.columns)
    total_missing = int(df.isnull().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    # Useful business-like numeric metrics
    primary_numeric = numeric_cols[0] if numeric_cols else None
    secondary_numeric = numeric_cols[1] if len(numeric_cols) > 1 else None

    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.metric("Total Records", f"{total_rows:,}")

    with k2:
        st.metric("Total Columns", f"{total_columns:,}")

    with k3:
        st.metric("Missing Values", f"{total_missing:,}")

    with k4:
        st.metric("Duplicate Rows", f"{duplicate_rows:,}")

    if primary_numeric:
        a, b, c, d = st.columns(4)

        with a:
            st.metric(
                f"Total {primary_numeric}",
                f"{df[primary_numeric].sum(skipna=True):,.2f}"
            )

        with b:
            st.metric(
                f"Average {primary_numeric}",
                f"{df[primary_numeric].mean(skipna=True):,.2f}"
            )

        with c:
            st.metric(
                f"Maximum {primary_numeric}",
                f"{df[primary_numeric].max(skipna=True):,.2f}"
            )

        with d:
            st.metric(
                f"Minimum {primary_numeric}",
                f"{df[primary_numeric].min(skipna=True):,.2f}"
            )

    # Main dashboard charts
    dash_col1, dash_col2 = st.columns(2)

    with dash_col1:
        if categorical_cols:
            category = next(
                (
                    c for c in categorical_cols
                    if 2 <= df[c].nunique(dropna=True) <= 20
                ),
                None
            )

            if category:
                counts = (
                    df[category]
                    .dropna()
                    .value_counts()
                    .head(10)
                    .reset_index()
                )
                counts.columns = [category, "Count"]

                fig = px.bar(
                    counts,
                    x=category,
                    y="Count",
                    title=f"Top {category} Categories",
                    text="Count"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No suitable categorical column for the dashboard.")

    with dash_col2:
        if len(numeric_cols) >= 1:
            value_col = primary_numeric
            fig = px.histogram(
                df,
                x=value_col,
                nbins=20,
                title=f"Distribution — {value_col}"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    # Second row
    dash_col3, dash_col4 = st.columns(2)

    with dash_col3:
        if len(numeric_cols) >= 2:
            fig = px.scatter(
                df,
                x=numeric_cols[0],
                y=numeric_cols[1],
                title=f"{numeric_cols[0]} vs {numeric_cols[1]}"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        elif date_columns_for_dashboard(df):
            date_col = date_columns_for_dashboard(df)[0]
            dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
            monthly = (
                dates.dt.to_period("M")
                .value_counts()
                .sort_index()
            )
            trend_df = pd.DataFrame({
                "Month": monthly.index.astype(str),
                "Records": monthly.values
            })
            fig = px.line(
                trend_df,
                x="Month",
                y="Records",
                markers=True,
                title=f"Records Over Time — {date_col}"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    with dash_col4:
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr()
            fig = px.imshow(
                corr,
                text_auto=".2f",
                aspect="auto",
                title="Numeric Correlation Heatmap"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        elif categorical_cols:
            category = next(
                (
                    c for c in categorical_cols
                    if 2 <= df[c].nunique(dropna=True) <= 8
                ),
                None
            )
            if category:
                pie_data = (
                    df[category]
                    .dropna()
                    .value_counts()
                    .head(8)
                    .reset_index()
                )
                pie_data.columns = [category, "Count"]
                fig = px.pie(
                    pie_data,
                    names=category,
                    values="Count",
                    hole=0.42,
                    title=f"{category} Share"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

    # Data quality panel
    st.markdown("### 🔎 Dashboard Data Quality")

    quality_df = pd.DataFrame({
        "Metric": [
            "Rows",
            "Columns",
            "Missing Values",
            "Duplicate Rows",
            "Numeric Columns",
            "Categorical Columns"
        ],
        "Value": [
            total_rows,
            total_columns,
            total_missing,
            duplicate_rows,
            len(numeric_cols),
            len(categorical_cols)
        ]
    })

    st.dataframe(
        quality_df,
        hide_index=True,
        use_container_width=True
    )


def date_columns_for_dashboard(df):
    detected = []

    for column in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[column]):
            detected.append(column)
            continue

        if df[column].dtype == "object":
            converted = pd.to_datetime(
                df[column],
                errors="coerce"
            )
            if converted.notna().mean() >= 0.7:
                detected.append(column)

    return detected


# ==========================================
# PAGE SETTINGS
# ==========================================

st.set_page_config(
    page_title="DataSense AI",
    page_icon="🤖",
    layout="wide"
)

# ==========================================
# COLOR THEME ENGINE
# ==========================================
THEME_OPTIONS = {
    "daylight": "☀️ Daylight",
    "dark": "🌙 Dark",
    "futuristic": "🚀 Futuristic",
    "professional": "💼 Professional",
}
selected_theme = st.query_params.get("theme", "daylight")
if selected_theme not in THEME_OPTIONS:
    selected_theme = "daylight"

# ==========================================
# GLASSMORPHISM AI COMMAND CENTER STYLE
# ==========================================

st.markdown("""
<style>
.stApp {
    background:
        radial-gradient(circle at 8% 8%, rgba(99,102,241,.18), transparent 28%),
        radial-gradient(circle at 92% 18%, rgba(34,211,238,.12), transparent 25%),
        radial-gradient(circle at 50% 95%, rgba(168,85,247,.10), transparent 32%),
        #070a12;
    color: #f5f7fb;
}

.block-container {
    max-width: 1450px;
    padding-top: 1.5rem;
    padding-bottom: 4rem;
}

h1 { font-size: 3rem !important; font-weight: 800 !important; letter-spacing: -1.5px; }
h2, h3 { font-weight: 750 !important; }

[data-testid="stMetric"] {
    background: rgba(255,255,255,.055);
    border: 1px solid rgba(255,255,255,.11);
    padding: 22px;
    border-radius: 18px;
    backdrop-filter: blur(18px);
    box-shadow: 0 10px 35px rgba(0,0,0,.25);
    transition: all .25s ease;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    border-color: rgba(129,140,248,.55);
    box-shadow: 0 14px 40px rgba(79,70,229,.18);
}

[data-testid="stMetricLabel"] { color: #aeb7c7; font-size: 13px; font-weight: 600; }
[data-testid="stMetricValue"] { font-size: 30px; font-weight: 800; }

[data-testid="stFileUploader"] {
    background: rgba(255,255,255,.045);
    border: 1px dashed rgba(129,140,248,.55);
    border-radius: 20px;
    padding: 12px;
    backdrop-filter: blur(16px);
}

.stButton > button {
    border-radius: 12px;
    border: 1px solid rgba(129,140,248,.38);
    background: linear-gradient(135deg, rgba(99,102,241,.22), rgba(34,211,238,.12));
    color: white;
    font-weight: 700;
    padding: .65rem 1.2rem;
    transition: all .25s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    border-color: rgba(129,140,248,.85);
    background: linear-gradient(135deg, rgba(99,102,241,.36), rgba(34,211,238,.20));
}

.stDownloadButton > button {
    border-radius: 12px;
    border: 1px solid rgba(34,211,238,.38);
    background: rgba(34,211,238,.10);
    color: white;
    font-weight: 700;
}

[data-testid="stDataFrame"] {
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,.08);
}

hr { border-color: rgba(255,255,255,.08); }

::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: #070a12; }
::-webkit-scrollbar-thumb { background: rgba(129,140,248,.35); border-radius: 10px; }

.ds-topbar {
    display:flex; justify-content:space-between; align-items:center;
    padding:18px 22px; margin-bottom:24px; border-radius:20px;
    background:rgba(255,255,255,.045);
    border:1px solid rgba(255,255,255,.10);
    backdrop-filter:blur(18px);
    box-shadow:0 12px 40px rgba(0,0,0,.22);
}

.ds-brand { display:flex; align-items:center; gap:14px; }
.ds-logo {
    width:48px; height:48px; display:flex; align-items:center; justify-content:center;
    border-radius:15px; font-size:24px;
    background:linear-gradient(135deg,rgba(99,102,241,.45),rgba(34,211,238,.22));
    border:1px solid rgba(129,140,248,.40);
}
.ds-title { font-size:20px; font-weight:800; letter-spacing:.5px; }
.ds-subtitle { color:#9ca8ba; font-size:12px; margin-top:2px; }

.ds-status {
    display:flex; align-items:center; gap:9px; padding:9px 14px;
    border-radius:999px; background:rgba(16,185,129,.09);
    border:1px solid rgba(52,211,153,.22);
    color:#a7f3d0; font-size:12px; font-weight:700;
}
.ds-dot {
    width:8px; height:8px; border-radius:50%; background:#34d399;
    box-shadow:0 0 12px rgba(52,211,153,.85);
}

.ds-hero {
    padding:30px; margin-bottom:24px; border-radius:24px;
    background:
        radial-gradient(circle at 85% 30%,rgba(34,211,238,.13),transparent 28%),
        radial-gradient(circle at 20% 70%,rgba(99,102,241,.13),transparent 32%),
        rgba(255,255,255,.045);
    border:1px solid rgba(255,255,255,.10);
    backdrop-filter:blur(18px);
}
.ds-kicker { color:#a5b4fc; font-size:12px; font-weight:800; letter-spacing:2px; text-transform:uppercase; }
.ds-hero-title { margin-top:8px; font-size:36px; line-height:1.08; font-weight:850; }
.ds-hero-text { max-width:700px; color:#aeb7c7; margin-top:10px; font-size:15px; }
.ds-section { margin:30px 0 12px; font-size:13px; font-weight:800; letter-spacing:1.4px; text-transform:uppercase; color:#8fa0b8; }

.ds-brain {
    min-height:190px; display:flex; flex-direction:column; align-items:center;
    justify-content:center; text-align:center; border-radius:24px;
    background:radial-gradient(circle,rgba(99,102,241,.18),transparent 48%),rgba(255,255,255,.035);
    border:1px solid rgba(129,140,248,.18);
}
.ds-core {
    width:76px; height:76px; border-radius:50%; display:flex; align-items:center;
    justify-content:center; font-size:34px;
    background:radial-gradient(circle,rgba(129,140,248,.45),rgba(34,211,238,.12));
    border:1px solid rgba(165,180,252,.45);
    box-shadow:0 0 35px rgba(99,102,241,.28);
}
.ds-core-title { margin-top:12px; font-weight:800; }
.ds-core-status { color:#8fa0b8; font-size:12px; margin-top:4px; }



/* ==========================================
   DATASENSE AI COLOR THEMES
   ========================================== */

/* The native Streamlit ⋮ menu remains available separately.
   These four themes control the DataSense AI visual system. */

/* DAYLIGHT — clean bright analytics */
.stApp:has(.ds-theme-daylight) {
    background:
        radial-gradient(circle at 8% 8%, rgba(99,102,241,.10), transparent 28%),
        radial-gradient(circle at 92% 18%, rgba(34,211,238,.08), transparent 25%),
        #f5f7fb !important;
    color:#172033 !important;
}
.ds-theme-daylight [data-testid="stMetric"],
.ds-theme-daylight [data-testid="stFileUploader"] {
    background:rgba(255,255,255,.94) !important;
    border-color:rgba(79,70,229,.22) !important;
    color:#172033 !important;
}
.stApp:has(.ds-theme-daylight) .ds-topbar,
.stApp:has(.ds-theme-daylight) .ds-hero,
.stApp:has(.ds-theme-daylight) .ds-brain {
    background:rgba(255,255,255,.88) !important;
    border-color:rgba(30,41,59,.10) !important;
    color:#172033 !important;
}
.stApp:has(.ds-theme-daylight) .ds-title,
.stApp:has(.ds-theme-daylight) .ds-hero-title,
.ds-theme-daylight h1,
.ds-theme-daylight h2,
.ds-theme-daylight h3 {
    color:#111827 !important;
}
.stApp:has(.ds-theme-daylight) .ds-subtitle,
.stApp:has(.ds-theme-daylight) .ds-hero-text,
.stApp:has(.ds-theme-daylight) .ds-core-status {
    color:#64748b !important;
}
.ds-theme-daylight [data-testid="stMetricLabel"] {color:#475569 !important;}
.ds-theme-daylight [data-testid="stMetricValue"] {color:#111827 !important;}
.ds-theme-daylight hr {border-color:rgba(30,41,59,.10) !important;}
.ds-theme-daylight ::-webkit-scrollbar-track {background:#eef2f7 !important;}
.ds-theme-daylight ::-webkit-scrollbar-thumb {background:rgba(79,70,229,.30) !important;}
.stApp:has(.ds-theme-daylight) .stButton > button {
    background:linear-gradient(135deg,rgba(99,102,241,.12),rgba(34,211,238,.10)) !important;
    color:#172033 !important;
}

/* DARK — existing DataSense glassmorphism */
.stApp:has(.ds-theme-dark) {
    background:
        radial-gradient(circle at 8% 8%, rgba(99,102,241,.18), transparent 28%),
        radial-gradient(circle at 92% 18%, rgba(34,211,238,.12), transparent 25%),
        radial-gradient(circle at 50% 95%, rgba(168,85,247,.10), transparent 32%),
        #070a12 !important;
    color:#f5f7fb !important;
}
.stApp:has(.ds-theme-dark) .ds-topbar,
.stApp:has(.ds-theme-dark) .ds-hero,
.stApp:has(.ds-theme-dark) .ds-brain {
    background:rgba(255,255,255,.045) !important;
    border-color:rgba(255,255,255,.10) !important;
}

/* FUTURISTIC — inspired by the supplied sci-fi circular-tech reference */
.stApp:has(.ds-theme-futuristic) {
    background:
        radial-gradient(circle at 50% 40%, rgba(0,220,255,.16), transparent 24%),
        radial-gradient(circle at 12% 20%, rgba(255,90,35,.14), transparent 28%),
        radial-gradient(circle at 88% 22%, rgba(40,210,190,.12), transparent 25%),
        linear-gradient(135deg,#05080d 0%,#081a21 48%,#02070b 100%) !important;
    color:#e7fbff !important;
}
.stApp:has(.ds-theme-futuristic) .ds-topbar,
.stApp:has(.ds-theme-futuristic) .ds-hero,
.stApp:has(.ds-theme-futuristic) .ds-brain {
    background:
        linear-gradient(135deg,rgba(9,32,40,.88),rgba(3,12,19,.82)) !important;
    border:1px solid rgba(62,229,255,.24) !important;
    box-shadow:0 0 35px rgba(0,210,255,.08), inset 0 0 28px rgba(0,160,190,.05) !important;
}
.stApp:has(.ds-theme-futuristic) .ds-logo,
.stApp:has(.ds-theme-futuristic) .ds-core {
    background:radial-gradient(circle,rgba(0,229,255,.35),rgba(0,71,100,.16)) !important;
    border-color:rgba(75,238,255,.55) !important;
    box-shadow:0 0 28px rgba(0,229,255,.20) !important;
}
.stApp:has(.ds-theme-futuristic) .ds-kicker,
.stApp:has(.ds-theme-futuristic) .ds-section {color:#55eaff !important;}
.stApp:has(.ds-theme-futuristic) .ds-title,
.stApp:has(.ds-theme-futuristic) .ds-hero-title,
.ds-theme-futuristic h1,
.ds-theme-futuristic h2,
.ds-theme-futuristic h3 {color:#f0fdff !important;}
.stApp:has(.ds-theme-futuristic) .ds-subtitle,
.stApp:has(.ds-theme-futuristic) .ds-hero-text,
.stApp:has(.ds-theme-futuristic) .ds-core-status {color:#9bc7cf !important;}
.ds-theme-futuristic [data-testid="stMetric"] {
    background:linear-gradient(145deg,rgba(9,37,47,.72),rgba(4,17,25,.76)) !important;
    border-color:rgba(0,224,255,.22) !important;
}
.ds-theme-futuristic [data-testid="stMetricLabel"] {color:#91cbd4 !important;}
.ds-theme-futuristic [data-testid="stMetricValue"] {color:#e9fdff !important;}
.ds-theme-futuristic [data-testid="stFileUploader"] {
    background:rgba(3,24,31,.72) !important;
    border-color:rgba(52,235,255,.45) !important;
}
.stApp:has(.ds-theme-futuristic) .stButton > button {
    background:linear-gradient(135deg,rgba(0,198,255,.18),rgba(26,232,196,.10)) !important;
    border-color:rgba(45,232,255,.42) !important;
    color:#eaffff !important;
}
.stApp:has(.ds-theme-futuristic) .stButton > button:hover {
    background:linear-gradient(135deg,rgba(0,198,255,.30),rgba(26,232,196,.18)) !important;
    box-shadow:0 0 22px rgba(0,220,255,.16) !important;
}
.ds-theme-futuristic hr {border-color:rgba(60,225,245,.14) !important;}
.ds-theme-futuristic ::-webkit-scrollbar-track {background:#061016 !important;}
.ds-theme-futuristic ::-webkit-scrollbar-thumb {background:rgba(0,225,255,.32) !important;}

/* PROFESSIONAL — inspired by the supplied Tableau-style reference */
.stApp:has(.ds-theme-professional) {
    background:
        radial-gradient(circle at 12% 12%, rgba(222,224,254,.58), transparent 28%),
        radial-gradient(circle at 92% 18%, rgba(207,238,226,.42), transparent 25%),
        #f6f7f4 !important;
    color:#1f2937 !important;
}
.stApp:has(.ds-theme-professional) .ds-topbar,
.stApp:has(.ds-theme-professional) .ds-hero,
.stApp:has(.ds-theme-professional) .ds-brain {
    background:rgba(255,255,255,.92) !important;
    border:1px solid rgba(31,41,55,.09) !important;
    box-shadow:0 12px 35px rgba(31,41,55,.07) !important;
    color:#1f2937 !important;
}
.stApp:has(.ds-theme-professional) .ds-logo {
    background:linear-gradient(135deg,#dee0fe,#cfeee2) !important;
    border-color:#d0d4ef !important;
}
.stApp:has(.ds-theme-professional) .ds-title,
.stApp:has(.ds-theme-professional) .ds-hero-title,
.ds-theme-professional h1,
.ds-theme-professional h2,
.ds-theme-professional h3 {color:#1f2937 !important;}
.stApp:has(.ds-theme-professional) .ds-subtitle,
.stApp:has(.ds-theme-professional) .ds-hero-text,
.stApp:has(.ds-theme-professional) .ds-core-status {color:#697386 !important;}
.stApp:has(.ds-theme-professional) .ds-kicker,
.stApp:has(.ds-theme-professional) .ds-section {color:#5963a5 !important;}
.ds-theme-professional [data-testid="stMetric"] {
    background:#ffffff !important;
    border-color:#e2e5ea !important;
    box-shadow:0 5px 18px rgba(31,41,55,.05) !important;
}
.ds-theme-professional [data-testid="stMetricLabel"] {color:#697386 !important;}
.ds-theme-professional [data-testid="stMetricValue"] {color:#1f2937 !important;}
.ds-theme-professional [data-testid="stFileUploader"] {
    background:#ffffff !important;
    border-color:#bfc8d9 !important;
}
.stApp:has(.ds-theme-professional) .stButton > button {
    background:linear-gradient(135deg,#dee0fe,#e7f0ec) !important;
    border-color:#c9cee6 !important;
    color:#1f2937 !important;
}
.stApp:has(.ds-theme-professional) .stButton > button:hover {
    background:linear-gradient(135deg,#d3d6fc,#dff0e8) !important;
    border-color:#aeb6d8 !important;
}
.ds-theme-professional hr {border-color:#e1e4e8 !important;}
.ds-theme-professional ::-webkit-scrollbar-track {background:#eceeea !important;}
.ds-theme-professional ::-webkit-scrollbar-thumb {background:#aab4c5 !important;}

/* ==========================================
   HOVER COLOR THEME MENU
   ========================================== */
.ds-theme-menu {
    position:relative;
    z-index:99999;
    display:inline-flex;
    align-items:center;
    font-size:12px;
    font-weight:800;
}
.ds-theme-trigger {
    display:flex;
    align-items:center;
    gap:8px;
    padding:9px 14px;
    border-radius:999px;
    background:rgba(255,255,255,.07);
    border:1px solid rgba(255,255,255,.15);
    color:inherit;
    cursor:pointer;
    white-space:nowrap;
    box-shadow:0 8px 24px rgba(0,0,0,.12);
}
.ds-theme-dropdown {
    position:absolute;
    right:0;
    top:calc(100% + 8px);
    min-width:190px;
    padding:7px;
    border-radius:16px;
    background:rgba(12,17,25,.97);
    border:1px solid rgba(255,255,255,.14);
    box-shadow:0 18px 45px rgba(0,0,0,.30);
    backdrop-filter:blur(20px);
    opacity:0;
    visibility:hidden;
    transform:translateY(-6px);
    transition:all .16s ease;
}
.ds-theme-menu:hover .ds-theme-dropdown,
.ds-theme-dropdown:hover {
    opacity:1;
    visibility:visible;
    transform:translateY(0);
}
.ds-theme-dropdown a {
    display:flex;
    align-items:center;
    gap:10px;
    padding:10px 12px;
    margin:2px 0;
    border-radius:10px;
    color:#dbeafe !important;
    text-decoration:none !important;
    font-size:12px;
    font-weight:750;
}
.ds-theme-dropdown a:hover,
.ds-theme-dropdown a.active {
    background:rgba(99,102,241,.20);
    color:#ffffff !important;
}
.stApp:has(.ds-theme-daylight) .ds-theme-trigger,
.stApp:has(.ds-theme-professional) .ds-theme-trigger {
    background:rgba(255,255,255,.78);
    color:#334155;
    border-color:rgba(31,41,55,.10);
}
.stApp:has(.ds-theme-professional) .ds-theme-dropdown,
.stApp:has(.ds-theme-daylight) .ds-theme-dropdown {
    background:rgba(255,255,255,.98);
    border-color:#dfe3e8;
}
.stApp:has(.ds-theme-professional) .ds-theme-dropdown a,
.stApp:has(.ds-theme-daylight) .ds-theme-dropdown a {
    color:#334155 !important;
}
.stApp:has(.ds-theme-professional) .ds-theme-dropdown a:hover,
.stApp:has(.ds-theme-daylight) .ds-theme-dropdown a:hover,
.stApp:has(.ds-theme-professional) .ds-theme-dropdown a.active,
.stApp:has(.ds-theme-daylight) .ds-theme-dropdown a.active {
    background:#eef0ff;
    color:#1f2937 !important;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# AI COMMAND CENTER HEADER
# ==========================================

theme_links = []
for key, label in THEME_OPTIONS.items():
    active = " active" if key == selected_theme else ""
    theme_links.append(
        f'<a class="{active.strip()}" href="?theme={key}">{label}</a>'
    )
theme_menu_html = "".join(theme_links)

st.markdown(f"""
<div class="ds-theme-{selected_theme}">
<div class="ds-topbar">
    <div class="ds-brand">
        <div class="ds-logo">🤖</div>
        <div>
            <div class="ds-title">DATASENSE AI</div>
            <div class="ds-subtitle">Autonomous Business Intelligence Command Center</div>
        </div>
    </div>
    <div style="display:flex;align-items:center;gap:12px;">
        <div class="ds-theme-menu">
            <div class="ds-theme-trigger">🎨 Color Theme <span>⌄</span></div>
            <div class="ds-theme-dropdown">
                {theme_menu_html}
            </div>
        </div>
        <div class="ds-status"><span class="ds-dot"></span> AI SYSTEM ONLINE</div>
    </div>
</div>

<div class="ds-hero">
    <div class="ds-kicker">AI DATA COMMAND CENTER</div>
    <div class="ds-hero-title">Turn raw data into<br>business intelligence.</div>
    <div class="ds-hero-text">
        Upload CSV, Excel, PDF, image, or ZIP files. DataSense AI profiles your data,
        discovers patterns, generates business insights, and creates reports.
    </div>
</div>

<div class="ds-section">DATA INPUT</div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# MULTI-FORMAT FILE UPLOAD
# ==========================================

SUPPORTED_EXTENSIONS = ["csv", "xlsx", "xls", "pdf", "png", "jpg", "jpeg", "zip"]
MAX_EXTRACTED_FILES = 50
MAX_TOTAL_EXTRACTED_BYTES = 200 * 1024 * 1024

def _safe_filename(name):
    return Path(str(name).replace("\\", "/")).name

def _read_pdf_text(data, max_pages=40, max_chars=60000):
    try:
        reader = PdfReader(BytesIO(data))
        parts = []
        for page in reader.pages[:max_pages]:
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            if text.strip():
                parts.append(text.strip())
            if sum(len(x) for x in parts) >= max_chars:
                break
        return "\n\n".join(parts)[:max_chars]
    except Exception:
        return ""

def _add_content_item(items, structured, name, data):
    safe_name = _safe_filename(name)
    ext = Path(safe_name).suffix.lower()
    if ext == ".csv":
        try:
            df = pd.read_csv(BytesIO(data))
            csv_buffer = BytesIO()
            df.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)
            csv_buffer.name = safe_name
            structured.append(csv_buffer)
            items.append({"name": safe_name, "kind": "dataframe", "df": df, "bytes": data, "mime": "text/csv"})
        except Exception as e:
            items.append({"name": safe_name, "kind": "error", "error": f"Could not read CSV: {e}"})
    elif ext in {".xlsx", ".xls"}:
        try:
            excel = pd.ExcelFile(BytesIO(data))
            for sheet_name in excel.sheet_names[:20]:
                try:
                    df = pd.read_excel(BytesIO(data), sheet_name=sheet_name)
                    display_name = f"{safe_name} — {sheet_name}"
                    csv_buffer = BytesIO()
                    df.to_csv(csv_buffer, index=False)
                    csv_buffer.seek(0)
                    csv_buffer.name = display_name
                    structured.append(csv_buffer)
                    items.append({"name": display_name, "kind": "dataframe", "df": df, "bytes": data, "mime": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "source_type": "excel"})
                except Exception as sheet_error:
                    items.append({"name": f"{safe_name} — {sheet_name}", "kind": "error", "error": str(sheet_error)})
        except Exception as e:
            items.append({"name": safe_name, "kind": "error", "error": f"Could not read Excel file: {e}"})
    elif ext == ".pdf":
        items.append({"name": safe_name, "kind": "pdf", "text": _read_pdf_text(data), "bytes": data, "mime": "application/pdf"})
    elif ext in {".png", ".jpg", ".jpeg"}:
        mime = "image/png" if ext == ".png" else "image/jpeg"
        items.append({"name": safe_name, "kind": "image", "bytes": data, "mime": mime})

def _process_uploaded_files(raw_files):
    items, structured, seen = [], [], set()
    total_uncompressed = 0
    def process_bytes(name, data, depth=0):
        nonlocal total_uncompressed
        safe_name = _safe_filename(name)
        ext = Path(safe_name).suffix.lower()
        if ext == ".zip" and depth < 2:
            try:
                with zipfile.ZipFile(BytesIO(data)) as archive:
                    for member in [m for m in archive.infolist() if not m.is_dir()]:
                        if len(items) >= MAX_EXTRACTED_FILES:
                            break
                        member_name = _safe_filename(member.filename)
                        member_ext = Path(member_name).suffix.lower()
                        if member_ext not in {".csv", ".xlsx", ".xls", ".pdf", ".png", ".jpg", ".jpeg", ".zip"}:
                            continue
                        total_uncompressed += int(member.file_size or 0)
                        if total_uncompressed > MAX_TOTAL_EXTRACTED_BYTES:
                            st.warning("⚠️ ZIP extraction stopped at 200 MB for safety.")
                            break
                        try:
                            process_bytes(f"{safe_name} / {member_name}", archive.read(member), depth + 1)
                        except Exception as member_error:
                            items.append({"name": member_name, "kind": "error", "error": str(member_error)})
            except zipfile.BadZipFile:
                items.append({"name": safe_name, "kind": "error", "error": "Invalid ZIP file."})
            return
        key = (safe_name.lower(), len(data))
        if key in seen:
            return
        seen.add(key)
        _add_content_item(items, structured, safe_name, data)
    for raw_file in raw_files or []:
        try:
            raw_file.seek(0)
            process_bytes(raw_file.name, raw_file.read())
        except Exception as e:
            items.append({"name": getattr(raw_file, "name", "Uploaded file"), "kind": "error", "error": str(e)})
    return items, structured

def _gemini_text(contents, max_attempts=2):
    models = ["gemini-3.5-flash-lite", "gemini-2.5-flash-lite"]
    last_error = None
    for model in models:
        for attempt in range(max_attempts):
            try:
                response = client.models.generate_content(model=model, contents=contents)
                text = getattr(response, "text", None)
                if text and text.strip():
                    return text.strip(), None
                last_error = "Gemini returned an empty response."
            except Exception as exc:
                last_error = str(exc)
                if attempt < max_attempts - 1:
                    time.sleep(1.5 * (attempt + 1))
    return None, last_error

raw_uploaded_files = st.file_uploader(
    "📁 Upload CSV, Excel, PDF, images or ZIP files",
    type=SUPPORTED_EXTENSIONS,
    accept_multiple_files=True,
    help="ZIP files can contain CSV, Excel, PDF, PNG or JPG files."
)

analysis_items, uploaded_files = _process_uploaded_files(raw_uploaded_files)

# Native Streamlit ⋮ menu provides System / Light / Dark.

if analysis_items:
    st.markdown(f"""
    <div class="ds-section">MISSION CONTROL</div>
    <div class="ds-topbar">
        <div>
            <div class="ds-title">DATASETS IN SESSION</div>
            <div class="ds-subtitle">{len(analysis_items)} file(s) ready for analysis</div>
        </div>
        <div class="ds-status"><span class="ds-dot"></span> DATA ENGINE READY</div>
    </div>
    """, unsafe_allow_html=True)

    dataset_cols = st.columns(min(len(analysis_items), 3))

    for i, item in enumerate(analysis_items):
        with dataset_cols[i % 3]:
            st.markdown(f"""
            <div style="
                padding:18px; margin-bottom:14px; border-radius:18px;
                background:rgba(255,255,255,.045);
                border:1px solid rgba(255,255,255,.10);
                backdrop-filter:blur(16px); min-height:105px;">
                <div style="font-size:12px;color:#a5b4fc;font-weight:800;">
                    DATASET {i+1:02d}
                </div>
                <div style="font-size:16px;font-weight:800;margin-top:7px;">
                    📁 {item["name"]}
                </div>
                <div style="font-size:12px;color:#8fa0b8;margin-top:5px;">
                    Ready for AI analysis
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="ds-section">AI REASONING CORE</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 1.4, 1])

    with c1:
        st.markdown("""
        <div class="ds-brain">
            <div class="ds-core">📊</div>
            <div class="ds-core-title">DATA</div>
            <div class="ds-core-status">Structure & quality</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="ds-brain">
            <div class="ds-core">🧠</div>
            <div class="ds-core-title">AI REASONING</div>
            <div class="ds-core-status">Patterns → Insights → Recommendations</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="ds-brain">
            <div class="ds-core">📄</div>
            <div class="ds-core-title">REPORTS</div>
            <div class="ds-core-status">Business-ready PDF output</div>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# AFTER CSV UPLOAD
# ==========================================

if "show_dashboard" not in st.session_state:
    st.session_state["show_dashboard"] = False

if uploaded_files:

    st.success(
        f"✅ {len(analysis_items)} file(s) loaded successfully!"
    )

    # ======================================
    # ANALYZE EACH CSV FILE
    # ======================================

    if "ai_results" not in st.session_state:
        st.session_state.ai_results = {}

    for file_number, uploaded_file in enumerate(uploaded_files, start=1):

        # Read current CSV
        df = pd.read_csv(uploaded_file)

        st.markdown("---")

        st.header(
            f"📁 {file_number}. {uploaded_file.name}"
        )

        # ======================================
        # BASIC METRICS
        # ======================================

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Rows",
                df.shape[0]
            )

        with col2:
            st.metric(
                "Columns",
                df.shape[1]
            )

        with col3:
            st.metric(
                "Missing Values",
                int(df.isnull().sum().sum())
            )

        with col4:
            st.metric(
                "Duplicate Rows",
                int(df.duplicated().sum())
            )

        # ======================================
        # DATA PREVIEW
        # ======================================

        st.subheader("📊 Data Preview")

        st.dataframe(
            df,
            use_container_width=True
        )

        # ======================================
        # DATASET INFORMATION
        # ======================================

        st.subheader("📋 Dataset Information")

        st.write("### Column Names")

        st.write(
            df.columns.tolist()
        )

        # ======================================
        # DATA TYPES
        # ======================================

        st.write("### Data Types")

        dtype_df = pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str).values
        })

        st.dataframe(
            dtype_df,
            use_container_width=True
        )

        # ======================================
        # NUMERIC COLUMNS
        # ======================================

        numeric_columns = df.select_dtypes(
            include="number"
        ).columns.tolist()

        st.write("### 🔢 Numeric Columns")

        if numeric_columns:

            st.write(numeric_columns)

        else:

            st.info(
                "No numeric columns found."
            )

        # ======================================
        # CATEGORICAL COLUMNS
        # ======================================

        categorical_columns = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        st.write("### 🏷️ Categorical Columns")

        if categorical_columns:

            st.write(categorical_columns)

        else:

            st.info(
                "No categorical columns found."
            )

        # ======================================
        # MISSING VALUES
        # ======================================

        st.write("### ⚠️ Missing Values by Column")

        missing_df = pd.DataFrame({
            "Column": df.columns,
            "Missing Values": df.isnull().sum().values
        })

        missing_df = missing_df[
            missing_df["Missing Values"] > 0
        ]

        if not missing_df.empty:

            st.dataframe(
                missing_df,
                use_container_width=True
            )

        else:

            st.success(
                "No missing values found!"
            )

        # ======================================
        # NUMERICAL STATISTICS
        # ======================================

        st.write("### 📈 Numerical Statistics")

        if numeric_columns:

            statistics = df[
                numeric_columns
            ].describe().T

            st.dataframe(
                statistics,
                use_container_width=True
            )

        else:

            st.info(
                "No numerical columns available "
                "for statistics."
            )

        # ======================================
        # SMART AUTOMATIC CHARTS
        # ======================================

        st.subheader(
            "📊 Smart Data Visualizations"
        )

        # --------------------------------------
        # DETECT DATE COLUMNS
        # --------------------------------------

        date_columns = []

        for column in df.columns:

            if df[column].dtype == "object":

                converted = pd.to_datetime(
                    df[column],
                    errors="coerce"
                )

                valid_ratio = converted.notna().mean()

                if valid_ratio >= 0.7:

                    date_columns.append(column)

        # --------------------------------------
        # FIND USEFUL CATEGORIES
        # --------------------------------------

        useful_categories = []

        for column in categorical_columns:

            unique_count = df[column].nunique(
                dropna=True
            )

            if unique_count <= 20:

                useful_categories.append(column)

        # --------------------------------------
        # MULTI-TYPE CATEGORY VISUALIZATIONS
        # --------------------------------------

        if useful_categories:

            st.write("### 🏷️ Category Visualizations")

            for column in useful_categories[:4]:

                chart_data = (
                    df[column]
                    .dropna()
                    .value_counts()
                    .head(10)
                )

                if not chart_data.empty:

                    st.write(
                        f"#### {column}"
                    )

                    # Use columns so bar + pie are shown side by side
                    chart_col1, chart_col2 = st.columns(2)

                    with chart_col1:

                        st.caption("📊 Bar Chart")

                        st.bar_chart(
                            chart_data,
                            use_container_width=True
                        )

                    with chart_col2:

                        st.caption("🥧 Pie Chart")

                        st.dataframe(
                            pd.DataFrame({
                                "Category": chart_data.index.astype(str),
                                "Count": chart_data.values
                            }),
                            hide_index=True,
                            use_container_width=True
                        )

                        # Streamlit native pie chart is not available,
                        # so use Plotly when installed.
                        try:
                            import plotly.express as px

                            pie_df = pd.DataFrame({
                                "Category": chart_data.index.astype(str),
                                "Count": chart_data.values
                            })

                            fig = px.pie(
                                pie_df,
                                names="Category",
                                values="Count",
                                hole=0.38,
                                title=f"Share of {column}"
                            )

                            fig.update_layout(
                                height=360,
                                margin=dict(l=10, r=10, t=55, b=10)
                            )

                            st.plotly_chart(
                                fig,
                                use_container_width=True
                            )

                        except Exception:
                            pass

        # --------------------------------------
        # NUMERIC ANALYSIS
        # --------------------------------------

        if numeric_columns:

            st.write("### 🔢 Numeric Visualizations")

            for column in numeric_columns[:4]:

                chart_data = df[column].dropna()

                if len(chart_data) > 0:

                    st.write(
                        f"#### {column}"
                    )

                    chart_col1, chart_col2 = st.columns(2)

                    with chart_col1:

                        st.caption("📊 Histogram")

                        try:

                            import plotly.express as px

                            numeric_fig = px.histogram(
                                df,
                                x=column,
                                nbins=20,
                                title=f"Distribution of {column}"
                            )

                            numeric_fig.update_layout(
                                height=360,
                                margin=dict(l=10, r=10, t=55, b=10)
                            )

                            st.plotly_chart(
                                numeric_fig,
                                use_container_width=True
                            )

                        except Exception:

                            histogram_data = pd.cut(
                                chart_data,
                                bins=10
                            ).value_counts().sort_index()

                            histogram_data.index = (
                                histogram_data.index.astype(str)
                            )

                            st.bar_chart(
                                histogram_data,
                                use_container_width=True
                            )

                    with chart_col2:

                        st.caption("📦 Box Plot")

                        try:

                            import plotly.express as px

                            box_fig = px.box(
                                df,
                                y=column,
                                points="outliers",
                                title=f"Spread & Outliers — {column}"
                            )

                            box_fig.update_layout(
                                height=360,
                                margin=dict(l=10, r=10, t=55, b=10)
                            )

                            st.plotly_chart(
                                box_fig,
                                use_container_width=True
                            )

                        except Exception:

                            st.info(
                                f"Box plot unavailable for {column}."
                            )

        # --------------------------------------
        # NUMERIC RELATIONSHIP / SCATTER PLOT
        # --------------------------------------

        if len(numeric_columns) >= 2:

            st.write("### 🔵 Numeric Relationship")

            scatter_x = numeric_columns[0]
            scatter_y = numeric_columns[1]

            try:

                import plotly.express as px

                scatter_df = df[
                    [scatter_x, scatter_y]
                ].dropna()

                if not scatter_df.empty:

                    scatter_fig = px.scatter(
                        scatter_df,
                        x=scatter_x,
                        y=scatter_y,
                        title=f"{scatter_x} vs {scatter_y}",
                        trendline="ols"
                    )

                    scatter_fig.update_layout(
                        height=430,
                        margin=dict(l=10, r=10, t=55, b=10)
                    )

                    st.plotly_chart(
                        scatter_fig,
                        use_container_width=True
                    )

            except Exception:

                st.info(
                    "Scatter plot could not be generated for these columns."
                )

        # --------------------------------------
        # CORRELATION HEATMAP
        # --------------------------------------

        if len(numeric_columns) >= 2:

            st.write("### 🔥 Correlation Heatmap")

            try:

                import plotly.express as px

                corr = df[numeric_columns].corr()

                heatmap_fig = px.imshow(
                    corr,
                    text_auto=".2f",
                    aspect="auto",
                    title="Correlation Between Numeric Columns"
                )

                heatmap_fig.update_layout(
                    height=500,
                    margin=dict(l=10, r=10, t=55, b=10)
                )

                st.plotly_chart(
                    heatmap_fig,
                    use_container_width=True
                )

            except Exception:

                st.info(
                    "Correlation heatmap could not be generated."
                )

        # --------------------------------------
        # DATE / TIME ANALYSIS
        # --------------------------------------

        if date_columns:

            st.write("### 📅 Time Visualizations")

            for column in date_columns[:2]:

                dates = pd.to_datetime(
                    df[column],
                    errors="coerce"
                ).dropna()

                if not dates.empty:

                    monthly_counts = (
                        dates
                        .dt.to_period("M")
                        .value_counts()
                        .sort_index()
                    )

                    monthly_counts.index = (
                        monthly_counts.index.astype(str)
                    )

                    if not monthly_counts.empty:

                        st.write(
                            f"#### Records Over Time — {column}"
                        )

                        try:

                            import plotly.express as px

                            time_df = pd.DataFrame({
                                "Month": monthly_counts.index,
                                "Records": monthly_counts.values
                            })

                            time_fig = px.line(
                                time_df,
                                x="Month",
                                y="Records",
                                markers=True,
                                title=f"Monthly Trend — {column}"
                            )

                            time_fig.update_layout(
                                height=400,
                                margin=dict(l=10, r=10, t=55, b=10)
                            )

                            st.plotly_chart(
                                time_fig,
                                use_container_width=True
                            )

                        except Exception:

                            st.line_chart(
                                monthly_counts,
                                use_container_width=True
                            )

# ==========================================
# 📄 PDF / 🖼️ IMAGE ANALYSIS
# ==========================================

multimodal_items = [item for item in analysis_items if item.get("kind") in {"pdf", "image"}]
if multimodal_items:
    st.markdown("---")
    st.subheader("📄 Document & Image Intelligence")
    st.write("Analyze uploaded PDFs and images with Gemini AI.")
    if st.button("🧠 Analyze PDFs & Images", key="analyze_multimodal", use_container_width=True):
        st.session_state["multimodal_results"] = {}
        for item in multimodal_items:
            with st.spinner(f"🧠 Analyzing {item['name']}..."):
                if item["kind"] == "pdf":
                    extracted = item.get("text", "").strip()
                    if extracted:
                        prompt = f"""You are DataSense AI, a professional business/data analyst.
Analyze this PDF content.
DOCUMENT: {item['name']}
CONTENT:
{extracted[:60000]}
Return: 1. Summary 2. Key Findings 3. Important Numbers / Facts 4. Insights 5. Recommendations. Use only information supported by the document."""
                        answer, error = _gemini_text(prompt)
                    else:
                        answer, error = None, "No selectable text could be extracted. The PDF may be scanned/image-only."
                else:
                    prompt = f"""You are DataSense AI, a professional business/data analyst. Analyze image {item['name']}. Identify visible tables, charts, numbers, labels, trends and useful observations. Return: 1. What the image contains 2. Key extracted data 3. Trends / patterns 4. Insights 5. Recommendations. Do not invent information."""
                    image_part = types.Part.from_bytes(data=item["bytes"], mime_type=item["mime"])
                    answer, error = _gemini_text([image_part, prompt])
                st.session_state["multimodal_results"][item["name"]] = answer or f"Unable to analyze this file: {error}"
    for name, result in st.session_state.get("multimodal_results", {}).items():
        st.markdown(f"### 📁 {name}")
        st.write(result)


# ==========================================
# DETERMINISTIC Q&A FALLBACK
# ==========================================

def _local_qa_fallback(df, question):
    """Answer common quantitative questions locally if Gemini is unavailable."""
    q = question.lower().strip()
    numeric = df.select_dtypes(include="number").columns.tolist()
    categorical = df.select_dtypes(include=["object", "category"]).columns.tolist()

    def fmt(value):
        try:
            if float(value).is_integer():
                return f"{int(value):,}"
            return f"{float(value):,.2f}"
        except Exception:
            return str(value)

    # Row/count questions
    if any(k in q for k in ["how many rows", "number of rows", "row count", "total records", "records are there"]):
        return f"The dataset contains {len(df):,} records."

    # Missing / duplicate questions
    if "missing" in q and ("value" in q or "data" in q):
        total_missing = int(df.isna().sum().sum())
        return f"The dataset contains {total_missing:,} missing values in total."

    if "duplicate" in q:
        return f"The dataset contains {int(df.duplicated().sum()):,} duplicate rows."

    # Numeric aggregate questions
    agg_map = [
        ("average", "mean"), ("mean", "mean"),
        ("maximum", "max"), ("highest", "max"), ("largest", "max"),
        ("minimum", "min"), ("lowest", "min"), ("smallest", "min"),
        ("total", "sum"), ("sum", "sum"),
    ]
    chosen = next(((word, op) for word, op in agg_map if word in q), None)

    if chosen and numeric:
        # Prefer a numeric column explicitly mentioned in the question.
        target = next((c for c in numeric if str(c).lower() in q), None)
        if target is None:
            target = numeric[0]
        series = pd.to_numeric(df[target], errors="coerce").dropna()
        if not series.empty:
            value = getattr(series, chosen[1])()
            return f"For **{target}**, the {chosen[0]} is **{fmt(value)}**."

    # Top category questions
    if any(k in q for k in ["top", "highest", "most", "best", "leading"]) and categorical:
        target_cat = next((c for c in categorical if str(c).lower() in q), categorical[0])
        counts = df[target_cat].value_counts(dropna=True)
        if not counts.empty:
            top_value = counts.index[0]
            return f"The most frequent value in **{target_cat}** is **{top_value}**, with **{int(counts.iloc[0]):,}** records."

    return None

# ==========================================
# 🔎 ASK DATA — AI BUSINESS Q&A
# ==========================================

if analysis_items:
    st.markdown("---")
    st.subheader("🔎 Ask DataSense AI")
    st.write("Ask questions about your uploaded CSV, Excel, PDF or image files. DataSense AI uses the selected file only.")
    qa_options = [item["name"] for item in analysis_items if item.get("kind") != "error"]
    if qa_options:
        qa_file_name = st.selectbox("📁 Select file", qa_options, key="qa_dataset")
        qa_question = st.text_input("💬 Ask a question about your data", placeholder="Example: Which product category has the highest revenue?", key="qa_question")
        example_cols = st.columns(3)
        with example_cols[0]: st.caption("💡 Try: What are the top categories?")
        with example_cols[1]: st.caption("💡 Try: What trends do you see?")
        with example_cols[2]: st.caption("💡 Try: What business action should I consider?")
        if st.button("🔍 Answer My Question", key="answer_data_question", use_container_width=True):
            if not qa_question.strip():
                st.warning("Please enter a question first.")
            else:
                selected_item = next(item for item in analysis_items if item["name"] == qa_file_name)
                qa_kind = selected_item.get("kind")
                if qa_kind == "dataframe":
                    qa_df = selected_item["df"]
                    qa_numeric = qa_df.select_dtypes(include="number").columns.tolist()
                    qa_categorical = qa_df.select_dtypes(include=["object", "category"]).columns.tolist()
                    qa_context = f"""Dataset: {qa_file_name}
Rows: {qa_df.shape[0]}
Columns: {qa_df.shape[1]}
Column names: {qa_df.columns.tolist()}
Data types: {qa_df.dtypes.astype(str).to_dict()}
Missing values: {qa_df.isnull().sum().to_dict()}
Duplicate rows: {int(qa_df.duplicated().sum())}
Numeric columns: {qa_numeric}
Categorical columns: {qa_categorical}
"""
                    if qa_numeric:
                        qa_context += f"\nNumerical statistics:\n{qa_df[qa_numeric].describe().round(2).to_string()}\n"
                        aggregates = []
                        for col in qa_numeric[:30]:
                            series = qa_df[col].dropna()
                            aggregates.append({"column": col, "sum": float(series.sum()) if not series.empty else None, "mean": float(series.mean()) if not series.empty else None, "min": float(series.min()) if not series.empty else None, "max": float(series.max()) if not series.empty else None})
                        qa_context += f"\nExact numeric aggregates:\n{aggregates}\n"
                    for col in qa_categorical[:12]:
                        if qa_df[col].nunique(dropna=True) <= 50:
                            qa_context += f"\nTop values for {col}:\n{qa_df[col].value_counts(dropna=True).head(20).to_dict()}\n"
                    qa_context += f"\nFirst 50 rows:\n{qa_df.head(50).to_dict(orient='records')}\n"
                    qa_prompt = f"""You are DataSense AI, a reliable business data analyst. Answer the user's question using ONLY this dataset context.
DATASET CONTEXT:
{qa_context}
USER QUESTION:
{qa_question}
Rules: Answer directly; prefer exact aggregates for totals/averages/min/max; use supplied category counts; do not invent values; if insufficient data, say what is missing; keep it concise and clear."""
                    with st.spinner("🧠 DataSense AI is analyzing your question..."):
                        qa_answer, qa_error = _gemini_text(qa_prompt, max_attempts=2)

                    # If Gemini is temporarily unavailable, answer common
                    # quantitative questions directly from the dataframe.
                    if not qa_answer:
                        qa_answer = _local_qa_fallback(qa_df, qa_question)
                        if qa_answer:
                            qa_error = None
                elif qa_kind == "pdf":
                    text = selected_item.get("text", "").strip()
                    if not text:
                        qa_answer, qa_error = None, "This PDF has no selectable text. It may be scanned/image-only."
                    else:
                        qa_prompt = f"""Answer the user's question using ONLY this PDF content. PDF: {qa_file_name} CONTENT: {text[:60000]} QUESTION: {qa_question} Give a direct answer and say if the document does not contain enough information."""
                        with st.spinner("🧠 DataSense AI is reading the PDF..."):
                            qa_answer, qa_error = _gemini_text(qa_prompt, max_attempts=2)
                elif qa_kind == "image":
                    image_part = types.Part.from_bytes(data=selected_item["bytes"], mime_type=selected_item["mime"])
                    qa_prompt = f"""Answer the user's question using ONLY information visible in this image. IMAGE: {qa_file_name} QUESTION: {qa_question} Read visible text, tables, chart labels and numbers carefully. Do not invent values."""
                    with st.spinner("🧠 DataSense AI is analyzing the image..."):
                        qa_answer, qa_error = _gemini_text([image_part, qa_prompt], max_attempts=2)
                else:
                    qa_answer, qa_error = None, selected_item.get("error", "Unsupported file")
                if qa_answer:
                    st.markdown("### 💡 AI Answer")
                    safe_answer = qa_answer.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
                    st.markdown(f"<div style='padding:22px;border-radius:18px;background:rgba(99,102,241,.10);border:1px solid rgba(129,140,248,.25);margin-top:10px;'>{safe_answer}</div>", unsafe_allow_html=True)
                else:
                    st.error(f"Unable to answer the question right now. {qa_error or 'Please try again.'}")


# ==========================================
# INTERACTIVE DASHBOARD
# ==========================================

if uploaded_files:

    st.markdown("---")
    st.subheader("🎯 Dashboard Center")
    st.write(
        "Create an interactive KPI dashboard from the uploaded CSV datasets."
    )

    if st.button(
        "🚀 Create Dashboard",
        key="create_dashboard",
        use_container_width=True
    ):
        st.session_state["show_dashboard"] = True

    if st.session_state.get("show_dashboard", False):

        st.markdown("### 📊 Business Intelligence Dashboard")

        for dashboard_number, dashboard_file in enumerate(
            uploaded_files,
            start=1
        ):

            dashboard_file.seek(0)
            dashboard_df = pd.read_csv(dashboard_file)

            with st.expander(
                f"📁 Dashboard {dashboard_number} — {dashboard_file.name}",
                expanded=True
            ):
                build_dashboard(
                    dashboard_df,
                    dashboard_file.name
                )


# ======================================
# AI BUSINESS INSIGHTS - ALL FILES
# ======================================

if uploaded_files:

    st.subheader("🤖 AI Business Insights")

    if "all_ai_reports" not in st.session_state:
        st.session_state.all_ai_reports = {}

    if st.button(
        "✨ Generate AI Insights for All Files",
        key="generate_all_ai"
    ):

        st.session_state.all_ai_reports = {}

        for file_number, uploaded_file in enumerate(
            uploaded_files,
            start=1
        ):

            with st.spinner(
                f"🤖 Analyzing {uploaded_file.name}..."
            ):

                # Read this uploaded file from the beginning
                uploaded_file.seek(0)
                df_ai = pd.read_csv(uploaded_file)

                # ----------------------------------
                # DETECT COLUMNS
                # ----------------------------------

                numeric_columns_ai = df_ai.select_dtypes(
                    include="number"
                ).columns.tolist()

                categorical_columns_ai = df_ai.select_dtypes(
                    include=["object", "category"]
                ).columns.tolist()

                # ----------------------------------
                # USEFUL CATEGORIES
                # ----------------------------------

                useful_categories_ai = []

                for column in categorical_columns_ai:

                    unique_count = df_ai[column].nunique(
                        dropna=True
                    )

                    if unique_count <= 20:
                        useful_categories_ai.append(column)

                # ----------------------------------
                # DATASET SUMMARY
                # ----------------------------------

                dataset_summary = f"""
Dataset name:
{uploaded_file.name}

Dataset has {df_ai.shape[0]} rows and {df_ai.shape[1]} columns.

Columns:
{df_ai.columns.tolist()}

Data types:
{df_ai.dtypes.astype(str).to_dict()}

Missing values:
{df_ai.isnull().sum().to_dict()}

Total missing values:
{int(df_ai.isnull().sum().sum())}

Duplicate rows:
{int(df_ai.duplicated().sum())}
"""

                # ----------------------------------
                # NUMERICAL STATISTICS
                # ----------------------------------

                if numeric_columns_ai:

                    dataset_summary += f"""

Numerical statistics:
{df_ai[numeric_columns_ai].describe().round(2).to_string()}
"""

                # ----------------------------------
                # CATEGORY INFORMATION
                # ----------------------------------

                for column in useful_categories_ai[:5]:

                    top_values = (
                        df_ai[column]
                        .value_counts()
                        .head(5)
                        .to_dict()
                    )

                    dataset_summary += f"""

Top values for {column}:
{top_values}
"""

                # ----------------------------------
                # GEMINI PROMPT
                # ----------------------------------

                prompt = f"""
You are a professional business data analyst.

Analyze this CSV dataset carefully.

{dataset_summary}

Provide a clear business analysis.

Use exactly these sections:

1. Key Findings
- Give 3 to 5 important findings.

2. Data Quality
- Mention missing values, duplicate rows, and obvious data-quality issues.

3. Business Insights
- Explain useful patterns, distributions, or trends found in the data.

4. Recommendations
- Give 3 practical recommendations based only on the available data.

Use simple English.

Do not invent information.

Do not make unsupported assumptions.

Only use information supported by the dataset.
"""

                # ----------------------------------
                # GEMINI
                # ----------------------------------

                analysis_text, analysis_error = _gemini_text(prompt, max_attempts=2)
                if not analysis_text:
                    st.error(f"Unable to analyze {uploaded_file.name}: {analysis_error or 'Gemini returned no answer.'}")
                    continue

                # ----------------------------------
                # CREATE PDF
                # ----------------------------------

                pdf_file = create_pdf_report(
                    f"DataSense AI - {uploaded_file.name}",
                    analysis_text,
                    df_ai
                )

                pdf_bytes = pdf_file.getvalue()

                # ----------------------------------
                # SAVE RESULT
                # ----------------------------------

                report_key = f"{file_number}_{uploaded_file.name}"

                st.session_state.all_ai_reports[report_key] = {
                    "file_name": uploaded_file.name,
                    "analysis": analysis_text,
                    "pdf": pdf_bytes
                }

    # ======================================
    # DISPLAY ALL GENERATED REPORTS
    # ======================================

    if st.session_state.all_ai_reports:

        st.markdown("---")

        st.subheader("📊 Generated Business Reports")

        for report_key, report in st.session_state.all_ai_reports.items():

            st.markdown("---")

            st.markdown(
                f"## 📁 {report['file_name']}"
            )

            st.markdown(
                "### 💡 AI-Generated Analysis"
            )

            st.write(
                report["analysis"]
            )

            st.markdown(
                "### 📄 Business Report"
            )

            st.download_button(
                label="📥 Download Business Report PDF",
                data=report["pdf"],
                file_name=f"DataSense_AI_{report['file_name']}.pdf",
                mime="application/pdf",
                key=f"download_{report_key}"
            )
