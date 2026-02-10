import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import google.generativeai as genai
from streamlit_gsheets import GSheetsConnection
import plotly.express as px
import plotly.graph_objects as go
import time
import re
import unicodedata
import io
from datetime import datetime

# ==============================================================================
# 1. CONFIGURACIÓN VISUAL — DARK GLASSMORPHISM THEME
# ==============================================================================
st.set_page_config(
    page_title="ONErpm · DSP Analytics",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* ── Base ── */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, sans-serif !important;
    }
    .stApp {
        background: #0F1117 !important;
    }
    
    /* ── Headers ── */
    h1, h2, h3, h4, h5 { 
        color: #F9FAFB !important; 
        font-weight: 700 !important;
        letter-spacing: -0.3px;
    }
    p, span, li, label, div {
        color: #D1D5DB !important;
    }
    
    /* ── Glass Card (Metrics) ── */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.04) !important;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px 24px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    div[data-testid="stMetric"]:hover {
        background: rgba(255, 255, 255, 0.07) !important;
        border-color: rgba(249, 115, 22, 0.4);
        box-shadow: 0 0 30px rgba(249, 115, 22, 0.1);
        transform: translateY(-3px);
    }
    div[data-testid="stMetricLabel"] { 
        color: #9CA3AF !important; 
        font-size: 0.78rem !important; 
        font-weight: 500 !important; 
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    div[data-testid="stMetricValue"] { 
        color: #F9FAFB !important; 
        font-size: 1.8rem !important; 
        font-weight: 800 !important;
    }
    div[data-testid="stMetricDelta"] {
        font-weight: 600 !important;
    }
    
    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        padding: 4px;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        color: #9CA3AF !important;
        font-weight: 500;
        padding: 10px 20px;
        transition: all 0.2s;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #F97316 0%, #EA580C 100%) !important;
        color: #FFFFFF !important;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab-border"] { display: none; }
    .stTabs [data-baseweb="tab-highlight"] { display: none; }
    
    /* ── Buttons ── */
    .stButton button, .stDownloadButton button {
        background: linear-gradient(135deg, #F97316 0%, #EA580C 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px;
        font-weight: 600;
        padding: 0.6rem 1.2rem;
        transition: all 0.3s;
    }
    .stButton button:hover, .stDownloadButton button:hover {
        box-shadow: 0 8px 25px rgba(249, 115, 22, 0.35);
        transform: translateY(-2px);
    }
    
    /* ── Chat Messages ── */
    .stChatMessage {
        background: rgba(255, 255, 255, 0.04) !important;
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 16px;
    }
    
    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #13151F 0%, #0F1117 100%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSelectbox div {
        color: #D1D5DB !important;
    }
    
    /* ── Inputs & Selects ── */
    .stSelectbox > div > div,
    .stMultiSelect > div > div {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px;
        color: #F9FAFB !important;
    }
    
    /* ── Chat Input ── */
    .stChatInput textarea {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px;
        color: #F9FAFB !important;
    }
    .stChatInput textarea:focus {
        border-color: rgba(249, 115, 22, 0.5) !important;
        box-shadow: 0 0 15px rgba(249, 115, 22, 0.15);
    }
    
    /* ── Dataframe ── */
    .stDataFrame {
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        overflow: hidden;
    }
    
    /* ── Dividers ── */
    hr { border-color: rgba(255, 255, 255, 0.06) !important; }
    
    /* ── Expander ── */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.04) !important;
        border-radius: 10px;
    }
    
    /* ── Alerts ── */
    .stAlert { border-radius: 12px; }
    
    /* ── Suggested Prompt Pills ── */
    .prompt-pill {
        display: inline-block;
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 20px;
        padding: 8px 16px;
        margin: 4px;
        color: #D1D5DB !important;
        font-size: 0.82rem;
        cursor: pointer;
        transition: all 0.2s;
    }
    .prompt-pill:hover {
        background: rgba(249,115,22,0.15);
        border-color: rgba(249,115,22,0.4);
        color: #F97316 !important;
    }
    
    /* ── Glass Section Cards ── */
    .glass-section {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 16px;
    }
    
    /* ── Growth Table ── */
    .growth-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 16px;
        border-bottom: 1px solid rgba(255,255,255,0.04);
        transition: background 0.2s;
    }
    .growth-row:hover {
        background: rgba(255,255,255,0.03);
    }
    .growth-row:last-child {
        border-bottom: none;
    }
    .dsp-name {
        color: #F9FAFB !important;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .dsp-count {
        color: #9CA3AF !important;
        font-size: 0.82rem;
    }
    .growth-badge-up {
        background: rgba(16,185,129,0.15);
        color: #10B981 !important;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .growth-badge-down {
        background: rgba(239,68,68,0.15);
        color: #EF4444 !important;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    /* ── Hide Streamlit branding ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

if "GOOGLE_API_KEY" not in st.secrets:
    st.error("🚨 CRÍTICO: Falta API Key en Secrets.")
    st.stop()
else:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# ==============================================================================
# 2. MOTOR ETL
# ==============================================================================
def normalize_text(text):
    if not isinstance(text, str): return str(text)
    text = "".join(c for c in unicodedata.normalize('NFKD', text) if not unicodedata.combining(c))
    return text.upper().strip()

URL_SHEET = "https://docs.google.com/spreadsheets/d/10y2YowTEgQYdWxs6c8D0fgJDDwGIT8_wyH0rQbERgG0/edit?gid=1919114384#gid=1919114384"

@st.cache_data(ttl=3600, show_spinner="📡 Conectando con Google Sheets...")
def load_data():
    conn = st.connection("gsheets", type=GSheetsConnection)
    return conn.read(spreadsheet=URL_SHEET, worksheet="DSP COPY")

@st.cache_data(ttl=3600, show_spinner="⚙️ Procesando datos...")
def clean_dataframe(df):
    try:
        df.columns = [str(c).upper().replace('\n', ' ').replace('/', '_').replace('.', '').strip().replace(' ', '_') for c in df.columns]
        cleaned_cols_log = []
        ignore_cols = ['YEAR', 'MONTH', 'WEEK', 'Q', 'INCLUSION_DATE', 'RELEASE_DATE']
        
        for col in df.columns:
            if col not in ignore_cols:
                clean_name = f"{col}_CLEAN"
                df[clean_name] = df[col].apply(lambda x: normalize_text(str(x)) if pd.notnull(x) else "UNKNOWN")
                cleaned_cols_log.append(clean_name)
        
        col_inc = next((c for c in df.columns if 'INCLUSION' in c), None)
        col_year = next((c for c in df.columns if c == 'YEAR'), None)
        col_month = next((c for c in df.columns if c == 'MONTH'), None)

        df['Year_Final'] = 0
        df['Month_Final'] = 0
        
        if col_inc:
            dt_inc = pd.to_datetime(df[col_inc], errors='coerce')
            df['Year_Final'] = dt_inc.dt.year.fillna(0).astype(int)
            df['Month_Final'] = dt_inc.dt.month.fillna(0).astype(int)
            
        if col_year:
            y_man = pd.to_numeric(df[col_year], errors='coerce').fillna(0).astype(int)
            df['Year_Final'] = df.apply(lambda x: y_man[x.name] if x['Year_Final'] == 0 else x['Year_Final'], axis=1)

        if col_month:
            mapa_mes = {'ENERO':1, 'ENE':1, 'JAN':1, 'FEBRERO':2, 'FEB':2, 'MARZO':3, 'MAR':3, 'ABRIL':4, 'ABR':4, 'MAYO':5, 'MAY':5, 'JUNIO':6, 'JUN':6, 'JULIO':7, 'JUL':7, 'AGOSTO':8, 'AGO':8, 'SEPTIEMBRE':9, 'SEP':9, 'OCTUBRE':10, 'OCT':10, 'NOVIEMBRE':11, 'NOV':11, 'DICIEMBRE':12, 'DIC':12}
            def get_month(x):
                s = normalize_text(str(x))
                if s.isdigit(): return int(s)
                return mapa_mes.get(s, 0)
            m_man = df[col_month].apply(get_month)
            df['Month_Final'] = df.apply(lambda x: m_man[x.name] if x['Month_Final'] == 0 else x['Month_Final'], axis=1)

        col_dsp = next((c for c in cleaned_cols_log if 'DSP' in c), None)
        if col_dsp: df = df[df[col_dsp] != 'UNKNOWN']

        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str)

        return df, cleaned_cols_log

    except Exception as e:
        st.error(f"Error ETL: {e}")
        return pd.DataFrame(), []

@st.cache_resource(ttl=86400, show_spinner="🤖 Cargando modelos IA...")
def get_valid_models():
    try:
        models = genai.list_models()
        valid = [m.name for m in models if 'generateContent' in m.supported_generation_methods]
        return valid
    except: return []

# ==============================================================================
# 3. HELPERS
# ==============================================================================
CHART_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#9CA3AF', family='Inter'),
    margin=dict(t=20, b=40, l=40, r=20),
)
GRID_STYLE = dict(gridcolor='rgba(255,255,255,0.04)', zerolinecolor='rgba(255,255,255,0.04)')
ACCENT_SEQUENCE = ['#F97316', '#EA580C', '#FB923C', '#10B981', '#3B82F6', '#EF4444', '#FBBF24', '#8B5CF6']

# DSP Brand Colors
DSP_COLORS = {
    'SPOTIFY': '#84CC16',
    'AMAZON': '#38BDF8',
    'CLARO': '#F87171',
    'YOUTUBE': '#EF4444',
    'MOVISTAR': '#1E40AF',
    'TIDAL': '#9CA3AF',
    'APPLE': '#F9FAFB',
    'DEEZER': '#A855F7',
}
DEFAULT_DSP_COLOR = '#FBBF24'  # Yellow for others

def get_dsp_color(dsp_name):
    name = str(dsp_name).upper()
    for key, color in DSP_COLORS.items():
        if key in name:
            return color
    return DEFAULT_DSP_COLOR

def get_greeting():
    hour = datetime.now().hour
    if hour < 12: return "Buenos días"
    elif hour < 18: return "Buenas tardes"
    else: return "Buenas noches"

MONTH_NAMES = {1:'Ene', 2:'Feb', 3:'Mar', 4:'Abr', 5:'May', 6:'Jun',
               7:'Jul', 8:'Ago', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dic'}

# ==============================================================================
# 4. SIDEBAR
# ==============================================================================
ONERPM_LOGO = "https://onerpm.com/wp-content/themes/onerpm-v2/assets/images/header-logo-dark.png"

with st.sidebar:
    st.markdown(f"""
    <div style='text-align: center; padding: 8px 0 16px 0;'>
        <img src='{ONERPM_LOGO}' width='140' style='filter: brightness(1.3); object-fit: contain; object-position: bottom;'>
        <p style='color: #6B7280 !important; font-size: 0.7rem; margin-top: 6px; letter-spacing: 0.15em; text-transform: uppercase;'>DSP Analytics Suite</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    valid_models_list = get_valid_models()
    pro_model = next((m for m in valid_models_list if "pro" in m), valid_models_list[0] if valid_models_list else None)
    flash_model = next((m for m in valid_models_list if "flash" in m), valid_models_list[0] if valid_models_list else None)
    
    sel_model = st.selectbox("🧠 Modelo IA", valid_models_list, index=valid_models_list.index(pro_model) if pro_model else 0)
    
    st.divider()
    raw_df = load_data()
    df, cols_clean = clean_dataframe(raw_df)
    
    if not df.empty:
        col_dsp = next((c for c in cols_clean if 'DSP' in c), None)
        pivot = df.groupby(['Year_Final', 'Month_Final', col_dsp]).size().reset_index(name='Count') if col_dsp else pd.DataFrame()
        pivot = pivot[pivot['Count'] > 0]
        truth_table = pivot.to_string(index=False)
        
        # Glassmorphism stat cards
        year_max = df['Year_Final'].max()
        count_ytd = len(df[df['Year_Final'] == year_max])
        
        st.markdown(f"""
        <div style='
            background: linear-gradient(135deg, rgba(249,115,22,0.15) 0%, rgba(234,88,12,0.15) 100%);
            border: 1px solid rgba(249,115,22,0.3);
            padding: 20px; border-radius: 16px; text-align: center; backdrop-filter: blur(10px);
        '>
            <p style='color: #9CA3AF !important; font-size: 0.7rem; margin: 0; text-transform: uppercase; letter-spacing: 0.1em;'>Total Registros</p>
            <p style='color: #F9FAFB !important; font-size: 2rem; font-weight: 800; margin: 6px 0 0 0; 
               background: linear-gradient(135deg, #F97316, #FB923C); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>{len(df):,}</p>
        </div>
        <div style='height: 8px'></div>
        <div style='display: flex; gap: 8px;'>
            <div style='flex: 1; background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); padding: 12px; border-radius: 12px; text-align: center;'>
                <p style='color: #6B7280 !important; font-size: 0.6rem; margin: 0; text-transform: uppercase;'>Año {year_max}</p>
                <p style='color: #F9FAFB !important; font-size: 1.1rem; font-weight: 700; margin: 4px 0 0 0;'>{count_ytd:,}</p>
            </div>
            <div style='flex: 1; background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); padding: 12px; border-radius: 12px; text-align: center;'>
                <p style='color: #6B7280 !important; font-size: 0.6rem; margin: 0; text-transform: uppercase;'>DSPs</p>
                <p style='color: #F9FAFB !important; font-size: 1.1rem; font-weight: 700; margin: 4px 0 0 0;'>{df[col_dsp].nunique() if col_dsp else 0}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<div style='height: 12px'></div>", unsafe_allow_html=True)
        
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='CleanData', index=False)
        st.download_button("📥 Descargar Excel", buffer, "onerpm_clean.xlsx")
        
    st.divider()
    st.markdown("""
    <p style='color: #4B5563 !important; font-size: 0.65rem; text-align: center; letter-spacing: 0.05em;'>
        v33.0 · ONErpm DSP Analytics<br>Powered by Gemini AI
    </p>
    """, unsafe_allow_html=True)

# ==============================================================================
# 5. MAIN CONTENT
# ==============================================================================
if not df.empty:
    
    # ── Welcome Header ──
    greeting = get_greeting()
    st.markdown(f"""
    <div style='margin-bottom: 8px;'>
        <h1 style='margin: 0; font-size: 1.8rem; font-weight: 800;'>
            {greeting} 👋
        </h1>
        <p style='color: #6B7280 !important; font-size: 0.9rem; margin-top: 4px;'>
            Aquí tienes el resumen de tus DSPs · {datetime.now().strftime('%d %b %Y')}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    tab_dash, tab_ai, tab_raw = st.tabs(["📊 Dashboard", "🤖 Analista IA", "🔎 Datos"])

    # ── TAB 1: DASHBOARD ──
    with tab_dash:
        # ── Filters ──
        all_years = sorted(df['Year_Final'].unique())
        all_months_nums = sorted(df['Month_Final'].unique())
        all_months_labels = {m: MONTH_NAMES.get(m, str(m)) for m in all_months_nums if m > 0}
        
        f1, f2, f3 = st.columns([1, 1, 2])
        with f1:
            sel_year = st.selectbox("📅 Año", ["Todos"] + [str(y) for y in all_years if y > 0], index=0, key="dash_year")
        with f2:
            month_options = ["Todos"] + [all_months_labels[m] for m in all_months_labels]
            sel_month_mode = st.selectbox("📆 Mes", month_options, index=0, key="dash_month_mode")
        
        # Resolve selected months
        if sel_month_mode == "Todos":
            sel_months = list(all_months_labels.keys())
        else:
            sel_months = [m for m, name in all_months_labels.items() if name == sel_month_mode]
        
        # Apply filters
        dff = df.copy()
        if sel_year != "Todos":
            dff = dff[dff['Year_Final'] == int(sel_year)]
        if sel_months:
            dff = dff[dff['Month_Final'].isin(sel_months)]
        
        st.markdown("<div style='height: 8px'></div>", unsafe_allow_html=True)
        
        # ── KPIs ──
        total = len(dff)
        year_max = int(sel_year) if sel_year != "Todos" else df['Year_Final'].max()
        year_prev = year_max - 1
        count_ytd = len(dff[dff['Year_Final'] == year_max]) if sel_year == "Todos" else total
        count_prev_df = df[df['Year_Final'] == year_prev]
        if sel_months:
            count_prev_df = count_prev_df[count_prev_df['Month_Final'].isin(sel_months)]
        count_prev = len(count_prev_df)
        growth = ((count_ytd - count_prev) / count_prev * 100) if count_prev > 0 else 0
        n_dsps = dff[col_dsp].nunique() if col_dsp else 0
        top_dsp = dff[col_dsp].mode()[0] if col_dsp and not dff.empty else "N/A"
        
        months_current = dff.groupby('Month_Final').size()
        avg_monthly = months_current.mean() if len(months_current) > 0 else 0
        
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Registros", f"{total:,}")
        c2.metric(f"vs {year_prev}", f"{count_ytd:,}", delta=f"{growth:+.1f}%")
        c3.metric("DSP Líder", top_dsp)
        c4.metric("DSPs Activos", f"{n_dsps}")
        c5.metric("Prom. Mensual", f"{avg_monthly:,.0f}")
        
        st.markdown("<div style='height: 16px'></div>", unsafe_allow_html=True)
        
        # ── Row 1: Bar + Ranking ──
        g1, g2 = st.columns([3, 2])
        with g1:
            st.markdown("##### 📈 Registros por Año")
            yd = dff.groupby('Year_Final').size().reset_index(name='Total')
            fig_bar = px.bar(
                yd, x='Year_Final', y='Total', text_auto=True,
                color_discrete_sequence=['#F97316']
            )
            fig_bar.update_traces(
                marker=dict(line=dict(width=0), cornerradius=6),
                textfont=dict(color='#F9FAFB', size=12, family='Inter'),
                textposition='outside'
            )
            fig_bar.update_layout(
                **CHART_LAYOUT,
                xaxis={**GRID_STYLE, 'title': ''}, 
                yaxis={**GRID_STYLE, 'title': ''},
                showlegend=False, bargap=0.3,
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with g2:
            st.markdown("##### 🏆 Ranking DSPs")
            if col_dsp and not dff.empty:
                dsp_totals = dff[col_dsp].value_counts().nlargest(8)
                top_dsps_list = dsp_totals.index
                
                # Compute YoY growth per DSP
                dsp_by_year = dff.groupby([col_dsp, 'Year_Final']).size().unstack(fill_value=0)
                
                rows_html = ""
                for dsp in top_dsps_list:
                    dsp_color = get_dsp_color(dsp)
                    total_dsp = int(dsp_totals[dsp])
                    
                    # YoY comparison
                    if year_max in dsp_by_year.columns and year_prev in dsp_by_year.columns and dsp in dsp_by_year.index:
                        curr = int(dsp_by_year.loc[dsp, year_max])
                        prev = int(dsp_by_year.loc[dsp, year_prev])
                        if prev > 0:
                            g_pct = ((curr - prev) / prev) * 100
                            is_up = g_pct >= 0
                            badge_bg = 'rgba(16,185,129,0.15)' if is_up else 'rgba(239,68,68,0.15)'
                            badge_color = '#10B981' if is_up else '#EF4444'
                            badge_text = f"{'↑' if is_up else '↓'} {abs(g_pct):.0f}%"
                        elif curr > 0:
                            badge_bg = 'rgba(16,185,129,0.15)'
                            badge_color = '#10B981'
                            badge_text = '✦ Nuevo'
                        else:
                            badge_bg = 'rgba(255,255,255,0.05)'
                            badge_color = '#9CA3AF'
                            badge_text = '—'
                    else:
                        badge_bg = 'rgba(255,255,255,0.05)'
                        badge_color = '#9CA3AF'
                        badge_text = '—'

                    rows_html += f"""<div style='display:flex; justify-content:space-between; align-items:center; padding:10px 16px; border-bottom:1px solid rgba(255,255,255,0.04);'><div style='display:flex; align-items:center; gap:10px;'><div style='width:8px; height:8px; border-radius:50%; background:{dsp_color};'></div><span style='color:#F9FAFB; font-weight:600; font-size:0.88rem;'>{dsp}</span></div><div style='display:flex; align-items:center; gap:12px;'><span style='color:#9CA3AF; font-size:0.82rem;'>{total_dsp:,}</span><span style='background:{badge_bg}; color:{badge_color}; padding:3px 10px; border-radius:20px; font-size:0.73rem; font-weight:600;'>{badge_text}</span></div></div>"""
                
                full_html = f"<div style='background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06); border-radius:16px; padding:4px 0; overflow:hidden;'>{rows_html}</div>"
                st.html(full_html)
        
        # ── Row 2: Area chart ──
        st.markdown("##### 📉 Tendencia Mensual")
        monthly = dff.groupby(['Year_Final', 'Month_Final']).size().reset_index(name='Total')
        monthly['SortKey'] = monthly['Year_Final'] * 100 + monthly['Month_Final']
        monthly = monthly.sort_values('SortKey')
        monthly['Label'] = monthly['Month_Final'].map(MONTH_NAMES).fillna('') + ' ' + monthly['Year_Final'].astype(str)
        
        fig_area = go.Figure()
        fig_area.add_trace(go.Scatter(
            x=monthly['Label'], y=monthly['Total'],
            mode='lines+markers',
            fill='tozeroy',
            line=dict(color='#F97316', width=3, shape='spline'),
            fillcolor='rgba(249, 115, 22, 0.08)',
            marker=dict(size=6, color='#F97316', line=dict(width=2, color='#0F1117')),
            hovertemplate='<b>%{x}</b><br>Registros: %{y:,}<extra></extra>'
        ))
        fig_area.update_layout(
            **CHART_LAYOUT,
            xaxis=dict(**GRID_STYLE, tickangle=-45, tickfont=dict(size=10), type='category', categoryorder='array', categoryarray=monthly['Label'].tolist()),
            yaxis=GRID_STYLE,
            hovermode='x unified',
        )
        st.plotly_chart(fig_area, use_container_width=True)
        
        # ── Row 3: Donut + Heatmap ──
        r1, r2 = st.columns(2)
        with r1:
            st.markdown("##### 🎯 Distribución de DSPs")
            if col_dsp:
                dist = dff[col_dsp].value_counts().reset_index()
                dist.columns = ['DSP', 'Total']
                dist_top = dist.head(8)
                dsp_chart_colors = [get_dsp_color(d) for d in dist_top['DSP']]
                fig_donut = px.pie(
                    dist_top, names='DSP', values='Total', hole=0.65,
                    color_discrete_sequence=dsp_chart_colors
                )
                fig_donut.update_traces(
                    textinfo='percent+label',
                    textfont=dict(size=11, color='#111111'),
                    marker=dict(line=dict(color='#0F1117', width=2))
                )
                fig_donut.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#9CA3AF', family='Inter'),
                    margin=dict(t=10, b=10, l=10, r=10),
                    showlegend=False,
                )
                st.plotly_chart(fig_donut, use_container_width=True)
        
        with r2:
            st.markdown("##### 🔥 Actividad por Mes y Año")
            heat_data = dff.groupby(['Year_Final', 'Month_Final']).size().reset_index(name='Count')
            heat_pivot = heat_data.pivot_table(index='Year_Final', columns='Month_Final', values='Count', fill_value=0)
            heat_pivot.columns = [MONTH_NAMES.get(c, str(c)) for c in heat_pivot.columns]
            
            fig_heat = px.imshow(
                heat_pivot.values,
                x=heat_pivot.columns.tolist(),
                y=[str(y) for y in heat_pivot.index],
                color_continuous_scale=[[0, '#1A1D29'], [0.3, '#7C2D12'], [0.6, '#EA580C'], [1, '#FB923C']],
                aspect='auto'
            )
            fig_heat.update_traces(
                hovertemplate='<b>%{y} · %{x}</b><br>Registros: %{z:,}<extra></extra>'
            )
            fig_heat.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#9CA3AF', family='Inter'),
                margin=dict(t=10, b=30, l=50, r=10),
                coloraxis_showscale=False,
                xaxis=dict(side='bottom'),
            )
            st.plotly_chart(fig_heat, use_container_width=True)

    # ── TAB 2: CHAT IA ──
    with tab_ai:
        st.markdown("""
        <div style='margin-bottom: 16px;'>
            <h2 style='margin: 0; font-size: 1.5rem;'>🤖 Consultor Inteligente</h2>
            <p style='color: #6B7280 !important; font-size: 0.85rem; margin-top: 4px;'>Pregunta cualquier cosa sobre tus datos — impulsado por Gemini AI</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Suggested prompts
        st.markdown("""
        <div style='margin-bottom: 16px;'>
            <p style='color: #6B7280 !important; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 8px;'>Sugerencias rápidas</p>
            <div>
                <span class='prompt-pill'>📈 Proyección Spotify Q1 2026</span>
                <span class='prompt-pill'>🔄 Comparar DSPs último año</span>
                <span class='prompt-pill'>📊 Top 10 DSPs por crecimiento</span>
                <span class='prompt-pill'>🎯 Análisis de tendencias mensuales</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "👋 ¡Hola! Soy tu analista de datos. Puedo hacer proyecciones, comparativas, y análisis profundos. Escribe tu consulta o usa una de las sugerencias de arriba."}]

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Escribe tu consulta aquí..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                caja = st.empty()
                
                def call_ai(prompt_text, model_name, fallback_name):
                    config = genai.GenerationConfig(temperature=0, max_output_tokens=8192)
                    try:
                        caja.info(f"⚡ Generando con {model_name.split('/')[-1]}...")
                        return genai.GenerativeModel(model_name).generate_content(prompt_text, generation_config=config)
                    except Exception as e:
                        err = str(e)
                        if "429" in err or "Quota" in err or "503" in err:
                            caja.warning("⏳ API ocupada, reintentando en 15s...")
                            bar = st.progress(0)
                            for t in range(15):
                                time.sleep(1)
                                bar.progress((t+1)/15)
                            bar.empty()
                            try:
                                return genai.GenerativeModel(model_name).generate_content(prompt_text, generation_config=config)
                            except:
                                if fallback_name and fallback_name != model_name:
                                    caja.info(f"🔄 Usando {fallback_name.split('/')[-1]}...")
                                    return genai.GenerativeModel(fallback_name).generate_content(prompt_text, generation_config=config)
                                raise
                        raise e

                code = None
                try:
                    # Limit context to avoid huge prompts
                    # Find key columns dynamically
                    col_artist = next((c for c in cols_clean if 'ARTIST' in c), None)
                    col_genre = next((c for c in cols_clean if 'GENRE' in c or 'ESTILO' in c), None)
                    col_bu = next((c for c in cols_clean if 'BU' in c or 'UNIT' in c), None)
                    
                    # Get top values for context
                    top_artists = df[col_artist].value_counts().head(5).index.tolist() if col_artist else []
                    top_genres = df[col_genre].value_counts().head(5).index.tolist() if col_genre else []
                    
                    # Columns summary
                    col_list = ", ".join(cols_clean)
                    
                    # Sample raw data (5 rows) to show structure
                    sample_rows = df.head(5)[[c for c in df.columns if 'CLEAN' in c or 'Year' in c or 'Month' in c]].to_string(index=False)
                    
                    prompt_sys = f"""Genera SOLO código Python ejecutable. MÁXIMO 50 LÍNEAS. SIN imports, SIN crear DataFrames.

VARIABLES YA CARGADAS:
- df: DataFrame REAL ({len(df)} filas).
- Columnas ÚTILES: {col_list}
- Year_Final (int), Month_Final (int)

EJEMPLOS DE VALORES REALES:
- Artistas ({col_artist}): {top_artists}
- Géneros ({col_genre}): {top_genres}
- BU ({col_bu}): (Usa {col_genre} si es None)

MUESTRA DE DATOS (5 filas):
{sample_rows}

SOLICITUD: {prompt}

REGLAS OBLIGATORIAS:
1. **VISUALIZACIÓN PREMIUM**: 
   - SIEMPRE genera un gráfico (`st.plotly_chart`).
   - Usa tarjetas Glassmorphism para KPIs (`st.markdown`).
   - NO uses `print()`.
2. **LÓGICA DE ARTISTAS (FEATURING)**:
   - "Chimbala" cuenta si está en "Ken-Y, Chimbala".
   - USA: `df[df['{col_artist}'].str.contains('NOMBRE', case=False, na=False)]`.
   - NO uses `==` para artistas, usa `str.contains`.
3. **BILLETES (PLACEMENTS)**: 
   - **1 Fila = 1 Placement**. Cuenta con `len()`.
   - Si un artista tiene 50 filas, tiene 50 placements.
4. **FILTROS GENERALES**: Usa `normalize_text` para DSPs o BUs, pero `str.contains` para artistas.
5. **COLUMNAS CALCULADAS**: Crea columnas ANTES de agrupar.

EJEMPLO DE OUTPUT (Tarjeta + Gráfico):
current_year = df['Year_Final'].max()
data = df[df['Year_Final'].isin([current_year, current_year-1])].copy()
# Filtrar artista con contains para capturar feats
data = data[data['{col_artist}'].str.contains('CHIMBALA', case=False, na=False)]

# Gráfico obligatorio
by_year = data.groupby('Year_Final').size().reset_index(name='Total')
fig = px.bar(by_year, x='Year_Final', y='Total', color_discrete_sequence=['#F97316'])
fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#D1D5DB'))
st.plotly_chart(fig, use_container_width=True)

# Tarjetas de resumen
total = len(data)
st.markdown(f'''
<div style='background: rgba(249,115,22,0.1); border: 1px solid rgba(249,115,22,0.2); border-radius: 12px; padding: 16px; text-align: center;'>
    <h3 style='color: #F9FAFB; margin: 0;'>{{total}}</h3>
    <p style='color: #F97316; margin: 0; font-size: 0.8rem; text-transform: uppercase;'>Placements Totales</p>
</div>
''', unsafe_allow_html=True)

for _, row in top.iterrows():
    st.markdown(f'''
    <div style='background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); border-radius: 12px; padding: 12px; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center;'>
        <div>
            <span style='color: #9CA3AF; font-size: 0.8rem; text-transform: uppercase;'>{{row['BU_Effective']}}</span>
            <h4 style='color: #F9FAFB; margin: 2px 0 0 0;'>{{row['{col_artist}'].title()}}</h4>
        </div>
        <div style='background: rgba(249,115,22,0.2); color: #F97316; padding: 4px 12px; border-radius: 20px; font-weight: bold;'>
            {{row['Total']}} Destaques
        </div>
    </div>
    ''', unsafe_allow_html=True)

EJEMPLO CORRECTO COMPLETO:
data = df[df['DSP_CLEAN'] == normalize_text('Spotify')]
by_year = data.groupby('Year_Final').size().reset_index(name='Total')
fig = px.bar(by_year, x='Year_Final', y='Total', color_discrete_sequence=['#F97316'])
fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#D1D5DB'))
st.plotly_chart(fig, use_container_width=True)
st.markdown(f"**Resultado:** Spotify tiene {{len(data):,}} registros totales.")

IMPORTANTE: Sé CONCISO. Máximo 40 líneas. No escribas comentarios innecesarios. Código directo y funcional."""

                    # Use flash model first (fast), fallback to pro
                    primary = flash_model if flash_model else sel_model
                    fallback = sel_model if sel_model != primary else None
                    response = call_ai(prompt_sys, primary, fallback)
                    code = response.text.replace("```python", "").replace("```", "").strip()
                    
                    # Sanitize: remove any import lines or DataFrame creation the AI might hallucinate
                    clean_lines = []
                    for line in code.split('\n'):
                        stripped = line.strip()
                        if stripped.startswith('import ') or stripped.startswith('from '):
                            continue
                        if 'pd.DataFrame(' in stripped and 'np.random' in code:
                            continue
                        if 'np.random.seed' in stripped:
                            continue
                        if 'np.repeat(' in stripped or 'np.random.choice(' in stripped:
                            continue
                        clean_lines.append(line)
                    code = '\n'.join(clean_lines)
                    
                    # Check for truncated code (unclosed brackets)
                    opens = code.count('(') - code.count(')') + code.count('[') - code.count(']')
                    if opens > 0:
                        caja.warning("⚠️ Código truncado. Reintentando con respuesta más corta...")
                        prompt_retry = f"El código anterior fue muy largo y se cortó. Genera una versión MÁS CORTA (máximo 25 líneas) que haga lo mismo. SIN imports. SIN comentarios.\n\nSOLICITUD: {prompt}\n\ndf tiene columnas: {col_list}, Year_Final, Month_Final. Usa .groupby().size() para contar."
                        response2 = call_ai(prompt_retry, primary, fallback)
                        code = response2.text.replace('```python', '').replace('```', '').strip()
                        # Re-sanitize
                        clean_lines = [l for l in code.split('\n') if not l.strip().startswith('import ') and not l.strip().startswith('from ')]
                        code = '\n'.join(clean_lines)
                    
                    if not code.strip():
                        caja.warning("⚠️ El modelo no generó código válido para ejecutar.")
                    else:
                        caja.info("🛠️ Ejecutando código...")
                        
                        # Capture stdout just in case the AI uses print() instead of st.write()
                        import contextlib
                        f = io.StringIO()
                        with contextlib.redirect_stdout(f):
                            exec_globals = {
                                "df": df, "pd": pd, "np": np, "st": st, "px": px, "go": go,
                                "LinearRegression": LinearRegression,
                                "normalize_text": normalize_text, "unicodedata": unicodedata, "io": io
                            }
                            exec(code, exec_globals)
                        
                        caja.empty()
                        
                        # If meaningful stdout, show it and save to history
                        out = f.getvalue()
                        if out.strip():
                            st.text(out)
                            st.session_state.messages.append({"role": "assistant", "content": f"```\n{out}\n```"})
                        else:
                            # If no stdout, assume st.markdown/plotly rendered directly. 
                            # We can't easily capture st calls for history without rerunning, 
                            # so we add a generic success message to history.
                            st.session_state.messages.append({"role": "assistant", "content": "✅ Análisis completado."})
                            st.success("✅ Análisis completado.")

                except Exception as e:
                    caja.error(f"Error: {e}")
                    if code:
                        with st.expander("🔍 Ver código generado"): st.code(code, language='python')

    # ── TAB 3: DATA EXPLORER ──
    with tab_raw:
        st.markdown("""
        <div style='margin-bottom: 16px;'>
            <h2 style='margin: 0; font-size: 1.5rem;'>🔎 Explorador de Datos</h2>
            <p style='color: #6B7280 !important; font-size: 0.85rem; margin-top: 4px;'>Filtra y explora los datos en detalle</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Filters in a cleaner layout
        fr1, fr2, fr3 = st.columns([2, 2, 1])
        with fr1:
            years_available = sorted(df['Year_Final'].unique())
            sel_years = st.multiselect("📅 Año", years_available, default=years_available)
        with fr2:
            if col_dsp:
                dsps_available = sorted(df[col_dsp].unique())
                sel_dsps = st.multiselect("🎧 DSP", dsps_available, default=dsps_available)
        with fr3:
            search_text = st.text_input("🔍 Buscar", placeholder="Texto...")
        
        filtered = df[df['Year_Final'].isin(sel_years)]
        if col_dsp and sel_dsps:
            filtered = filtered[filtered[col_dsp].isin(sel_dsps)]
        
        # Apply text search across all string columns
        if search_text:
            mask = filtered.apply(lambda row: row.astype(str).str.contains(search_text, case=False).any(), axis=1)
            filtered = filtered[mask]
        
        # Stats row
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Registros", f"{len(filtered):,}")
        s2.metric("% del Total", f"{len(filtered)/len(df)*100:.1f}%")
        s3.metric("DSPs", f"{filtered[col_dsp].nunique() if col_dsp else 0}")
        s4.metric("Años", f"{filtered['Year_Final'].nunique()}")
        
        st.dataframe(filtered, use_container_width=True, height=500)
        
        # Export filtered data
        if len(filtered) < len(df):
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine='xlsxwriter') as w:
                filtered.to_excel(w, sheet_name='Filtered', index=False)
            st.download_button("📥 Exportar Filtrado", buf, "onerpm_filtered.xlsx")
