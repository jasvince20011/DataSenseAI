import streamlit as st
import pandas as pd
from google import genai
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from io import BytesIO

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
   THEME SUPPORT
   ========================================== */

html[data-theme="light"] .stApp,
body[data-theme="light"] .stApp {
    background:
        radial-gradient(circle at 8% 8%, rgba(99,102,241,.10), transparent 28%),
        radial-gradient(circle at 92% 18%, rgba(34,211,238,.09), transparent 25%),
        #f5f7fb !important;
    color: #172033 !important;
}

html[data-theme="light"] [data-testid="stMetric"],
body[data-theme="light"] [data-testid="stMetric"] {
    background: rgba(255,255,255,.92) !important;
    border-color: rgba(30,41,59,.12) !important;
    color: #172033 !important;
}

html[data-theme="light"] .ds-topbar,
body[data-theme="light"] .ds-topbar,
html[data-theme="light"] .ds-hero,
body[data-theme="light"] .ds-hero,
html[data-theme="light"] .ds-brain,
body[data-theme="light"] .ds-brain {
    background: rgba(255,255,255,.88) !important;
    border-color: rgba(30,41,59,.10) !important;
    color: #172033 !important;
}

html[data-theme="light"] .ds-title,
body[data-theme="light"] .ds-title,
html[data-theme="light"] .ds-hero-title,
body[data-theme="light"] .ds-hero-title,
html[data-theme="light"] h1,
body[data-theme="light"] h1,
html[data-theme="light"] h2,
body[data-theme="light"] h2,
html[data-theme="light"] h3,
body[data-theme="light"] h3 {
    color: #111827 !important;
}

html[data-theme="light"] .ds-subtitle,
body[data-theme="light"] .ds-subtitle,
html[data-theme="light"] .ds-hero-text,
body[data-theme="light"] .ds-hero-text,
html[data-theme="light"] .ds-core-status,
body[data-theme="light"] .ds-core-status {
    color: #64748b !important;
}

html[data-theme="light"] [data-testid="stFileUploader"],
body[data-theme="light"] [data-testid="stFileUploader"] {
    background: rgba(255,255,255,.90) !important;
    border-color: rgba(79,70,229,.30) !important;
}

html[data-theme="light"] hr,
body[data-theme="light"] hr {
    border-color: rgba(30,41,59,.10) !important;
}

html[data-theme="light"] ::-webkit-scrollbar-track,
body[data-theme="light"] ::-webkit-scrollbar-track {
    background: #eef2f7;
}

/* User theme selector button */
.ds-theme-selector {
    display:flex;
    align-items:center;
    gap:8px;
    padding:7px 12px;
    border-radius:999px;
    background:rgba(255,255,255,.05);
    border:1px solid rgba(255,255,255,.10);
    color:#cbd5e1;
    font-size:12px;
    font-weight:700;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# AI COMMAND CENTER HEADER
# ==========================================

st.markdown("""
<div class="ds-topbar">
    <div class="ds-brand">
        <div class="ds-logo">🤖</div>
        <div>
            <div class="ds-title">DATASENSE AI</div>
            <div class="ds-subtitle">Autonomous Business Intelligence Command Center</div>
        </div>
    </div>
    <div class="ds-status"><span class="ds-dot"></span> AI SYSTEM ONLINE</div>
</div>

<div class="ds-hero">
    <div class="ds-kicker">AI DATA COMMAND CENTER</div>
    <div class="ds-hero-title">Turn raw data into<br>business intelligence.</div>
    <div class="ds-hero-text">
        Upload one or multiple CSV datasets. DataSense AI profiles your data,
        discovers patterns, generates business insights, and creates reports.
    </div>
</div>

<div class="ds-section">DATA INPUT</div>
""", unsafe_allow_html=True)

# ==========================================
# CSV UPLOAD
# ==========================================


uploaded_files = st.file_uploader(
    "📁 Upload your CSV files",
    type=["csv"],
    accept_multiple_files=True
)

# ==========================================
# THEME + DASHBOARD CONTROLS
# ==========================================

theme_col1, theme_col2 = st.columns([5, 1])

with theme_col2:
    selected_theme = st.selectbox(
        "🎨 Theme",
        ["🌙 Dark", "☀️ Light"],
        key="datasense_theme"
    )

# Apply a CSS class to the current app based on the selected theme.
# Streamlit's own menu remains available from the top-right three dots.
if selected_theme == "☀️ Light":
    st.markdown("""
    <style>
    .stApp {
        background:
            radial-gradient(circle at 8% 8%, rgba(99,102,241,.10), transparent 28%),
            radial-gradient(circle at 92% 18%, rgba(34,211,238,.09), transparent 25%),
            #f5f7fb !important;
        color:#172033 !important;
    }
    .ds-topbar, .ds-hero, .ds-brain {
        background:rgba(255,255,255,.92) !important;
        border-color:rgba(30,41,59,.10) !important;
    }
    .ds-title, .ds-hero-title, h1, h2, h3 {
        color:#111827 !important;
    }
    .ds-subtitle, .ds-hero-text, .ds-core-status {
        color:#64748b !important;
    }
    [data-testid="stMetric"] {
        background:rgba(255,255,255,.92) !important;
        border-color:rgba(30,41,59,.12) !important;
    }
    [data-testid="stFileUploader"] {
        background:rgba(255,255,255,.90) !important;
    }
    </style>
    """, unsafe_allow_html=True)

if uploaded_files:
    st.markdown(f"""
    <div class="ds-section">MISSION CONTROL</div>
    <div class="ds-topbar">
        <div>
            <div class="ds-title">DATASETS IN SESSION</div>
            <div class="ds-subtitle">{len(uploaded_files)} dataset(s) ready for analysis</div>
        </div>
        <div class="ds-status"><span class="ds-dot"></span> DATA ENGINE READY</div>
    </div>
    """, unsafe_allow_html=True)

    dataset_cols = st.columns(min(len(uploaded_files), 3))

    for i, file in enumerate(uploaded_files):
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
                    📁 {file.name}
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
        f"✅ {len(uploaded_files)} CSV file(s) uploaded successfully!"
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
# 🔎 ASK DATA — AI BUSINESS Q&A
# ==========================================

if uploaded_files:

    st.markdown("---")
    st.subheader("🔎 Ask DataSense AI")
    st.write(
        "Ask questions about your uploaded CSV data and get answers "
        "based only on the selected dataset."
    )

    qa_files = {}
    for qa_file in uploaded_files:
        qa_file.seek(0)
        qa_files[qa_file.name] = pd.read_csv(qa_file)

    qa_file_name = st.selectbox(
        "📁 Select dataset",
        list(qa_files.keys()),
        key="qa_dataset"
    )

    qa_question = st.text_input(
        "💬 Ask a question about your data",
        placeholder=(
            "Example: Which product category has the highest revenue?"
        ),
        key="qa_question"
    )

    example_cols = st.columns(3)

    with example_cols[0]:
        st.caption("💡 Try: What are the top categories?")

    with example_cols[1]:
        st.caption("💡 Try: What trends do you see?")

    with example_cols[2]:
        st.caption("💡 Try: What business action should I consider?")

    if st.button(
        "🔍 Answer My Question",
        key="answer_data_question",
        use_container_width=True
    ):

        if not qa_question.strip():
            st.warning("Please enter a question first.")

        else:

            qa_df = qa_files[qa_file_name]

            # Build a compact but useful dataset context.
            qa_numeric = qa_df.select_dtypes(
                include="number"
            ).columns.tolist()

            qa_categorical = qa_df.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()

            qa_context = f"""
Dataset: {qa_file_name}

Rows: {qa_df.shape[0]}
Columns: {qa_df.shape[1]}

Column names:
{qa_df.columns.tolist()}

Data types:
{qa_df.dtypes.astype(str).to_dict()}

Missing values:
{qa_df.isnull().sum().to_dict()}

Duplicate rows:
{int(qa_df.duplicated().sum())}

Numeric columns:
{qa_numeric}

Categorical columns:
{qa_categorical}
"""

            if qa_numeric:
                qa_context += f"""

Numerical statistics:
{qa_df[qa_numeric].describe().round(2).to_string()}
"""

            for qa_column in qa_categorical[:8]:
                unique_count = qa_df[qa_column].nunique(dropna=True)

                if unique_count <= 30:
                    qa_top = (
                        qa_df[qa_column]
                        .value_counts(dropna=True)
                        .head(15)
                        .to_dict()
                    )

                    qa_context += f"""

Top values for {qa_column}:
{qa_top}
"""

            # Include a sample so the AI can answer questions
            # about actual rows without sending an unnecessarily
            # large CSV to the model.
            sample_rows = qa_df.head(30).to_dict(orient="records")

            qa_context += f"""

First 30 rows:
{sample_rows}
"""

            qa_prompt = f"""
You are DataSense AI, a professional business data analyst.

Answer the user's question using ONLY the uploaded dataset information
provided below.

DATASET CONTEXT:
{qa_context}

USER QUESTION:
{qa_question}

Rules:
1. Give a direct answer first.
2. Use numbers from the dataset whenever they support the answer.
3. Explain the calculation or reasoning briefly when useful.
4. If the question asks for a ranking, identify the relevant categories
   or values from the available data.
5. If the question asks for a business insight, explain the observed
   pattern and why it matters.
6. Do not invent values, columns, trends, or business facts.
7. If the dataset does not contain enough information to answer,
   clearly say that the uploaded data does not contain enough information.
8. Keep the answer easy to understand.
9. Use bullet points when there are multiple findings.

Return only the answer to the user's question.
"""

            with st.spinner("🧠 DataSense AI is analyzing your question..."):

                try:

                    qa_response = client.models.generate_content(
                        model="gemini-3.5-flash-lite",
                        contents=qa_prompt
                    )

                    qa_answer = qa_response.text

                    st.markdown("### 💡 AI Answer")

                    st.markdown(
                        f"""
                        <div style="
                            padding:22px;
                            border-radius:18px;
                            background:rgba(99,102,241,.10);
                            border:1px solid rgba(129,140,248,.25);
                            margin-top:10px;
                        ">
                            {qa_answer.replace(chr(10), '<br>')}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                except Exception as e:

                    st.error(
                        f"Unable to answer the question: {str(e)}"
                    )


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

                response = client.models.generate_content(
                    model="gemini-3.5-flash-lite",
                    contents=prompt
                )

                analysis_text = response.text

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
