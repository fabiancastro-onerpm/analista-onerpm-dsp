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

# ==============================================================================
# 1. ARQUITECTURA VISUAL (ESTILO DASHBOARD)
# ==============================================================================
st.set_page_config(
    page_title="ONErpm Enterprise Dashboard",
    page_icon="🎹",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Fondo limpio */
    .stApp { background-color: #FFFFFF !important; }
    
    /* Tipografía */
    h1, h2, h3, p, div, span, li, label { 
        color: #1F2937 !important; 
        font-family: 'Segoe UI', sans-serif; 
    }
    
    /* Métricas (KPI Cards) */
    div[data-testid="stMetric"] { 
        background-color: #F8F9FA !important; 
        border: 1px solid #E9ECEF; 
        border-left: 5px solid #005fcc; /* ONErpm Blue */
        border-radius: 8px; 
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    div[data-testid="stMetricLabel"] { color: #6C757D !important; font-size: 0.9rem; font-weight: 700; }
    div[data-testid="stMetricValue"] { color: #212529 !important; font-size: 1.8rem; font-weight: 800; }
    
    /* Pestañas */
    .stTabs [data-baseweb="tab-list"] { gap: 20px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; border-radius: 5px; color: #495057; }
    .stTabs [aria-selected="true"] { background-color: #E7F1FF; color: #005fcc; font-weight: bold; }
    
    /* Chat */
    .stChatMessage { background-color: #F8F9FA !important; border: 1px solid #E9ECEF; }
</style>
""", unsafe_allow_html=True)

# Validación API
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("🚨 CRÍTICO: Falta API Key en Secrets.")
    st.stop()
else:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# ==============================================================================
# 2. MOTOR DE DATOS (ETL ROBUSTO)
# ==============================================================================
def normalize_text(text):
    if not isinstance(text, str): return str(text)
    text = "".join(c for c in unicodedata.normalize('NFKD', text) if not unicodedata.combining(c))
    return text.upper().strip()

URL_SHEET = "https://docs.google.com/spreadsheets/d/10y2YowTEgQYdWxs6c8D0fgJDDwGIT8_wyH0rQbERgG0/edit?gid=1919114384#gid=1919114384"

@st.cache_data(ttl=3600, show_spinner="📡 Conectando...")
def load_data():
    conn = st.connection("gsheets", type=GSheetsConnection)
    return conn.read(spreadsheet=URL_SHEET, worksheet="DSP COPY")

@st.cache_data(ttl=3600, show_spinner="⚙️ Procesando Dashboard...")
def clean_dataframe(df):
    try:
        df.columns = [str(c).upper().replace('\n', ' ').replace('/', '_').replace('.', '').strip().replace(' ', '_') for c in df.columns]
        cleaned_cols_log = []
        ignore_cols = ['YEAR', 'MONTH', 'WEEK', 'Q', 'INCLUSION_DATE', 'RELEASE_DATE']
        
        # Limpieza de texto
        for col in df.columns:
            if col not in ignore_cols:
                clean_name = f"{col}_CLEAN"
                df[clean_name] = df[col].apply(lambda x: normalize_text(str(x)) if pd.notnull(x) else "UNKNOWN")
                cleaned_cols_log.append(clean_name)
        
        # Fechas
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

        return df, cleaned_cols_log

    except Exception as e:
        st.error(f"Error ETL: {e}")
        return pd.DataFrame(), []

def get_valid_models():
    try:
        models = genai.list_models()
        valid = [m.name for m in models if 'generateContent' in m.supported_generation_methods]
        return valid
    except: return []

# ==============================================================================
# 3. INTERFAZ LATERAL
# ==============================================================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/19/Spotify_logo_without_text.svg/168px-Spotify_logo_without_text.svg.png", width=50)
    st.title("ONErpm Suite")
    st.caption("v30.0 Enterprise Tabular")
    
    valid_models_list = get_valid_models()
    if valid_models_list:
        default_idx = 0
        for i, m in enumerate(valid_models_list):
            if "pro" in m: default_idx = i; break
        sel_model = st.selectbox("Modelo IA:", valid_models_list, index=default_idx)
    
    st.divider()
    
    raw_df = load_data()
    df, cols_clean = clean_dataframe(raw_df)
    
    if not df.empty:
        # Chivato para la IA
        col_dsp = next((c for c in cols_clean if 'DSP' in c), None)
        pivot = df.groupby(['Year_Final', 'Month_Final', col_dsp]).size().reset_index(name='Count') if col_dsp else pd.DataFrame()
        pivot = pivot[pivot['Count'] > 0]
        truth_table = pivot.to_string(index=False)
        
        st.success(f"Base de Datos: {len(df):,} filas")
        
        # Exportador
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='CleanData', index=False)
        st.download_button("📥 Descargar Excel Limpio", buffer, "onerpm_clean.xlsx")

# ==============================================================================
# 4. ÁREA DE TRABAJO (TABS)
# ==============================================================================
if not df.empty:
    
    # ESTRUCTURA DE PESTAÑAS
    tab_dash, tab_ai, tab_raw = st.tabs(["📊 Dashboard General (Review)", "🤖 Analista IA (Consultor)", "🔎 Datos Crudos"])

    # --------------------------------------------------------------------------
    # PESTAÑA 1: DASHBOARD (Python Puro - Sin cuotas)
    # --------------------------------------------------------------------------
    with tab_dash:
        st.header("Resumen de Rendimiento Global")
        st.caption("Vista general automática. No consume créditos de IA.")
        
        # 1. KPIs Superiores
        total_rows = len(df)
        year_max = df['Year_Final'].max()
        year_prev = year_max - 1
        
        count_max = len(df[df['Year_Final'] == year_max])
        count_prev = len(df[df['Year_Final'] == year_prev])
        delta = count_max - count_prev
        delta_pct = (delta / count_prev * 100) if count_prev > 0 else 0
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Histórico", f"{total_rows:,}")
        c2.metric(f"Destaques {year_prev}", f"{count_prev:,}")
        c3.metric(f"Destaques {year_max} (YTD)", f"{count_max:,}", f"{delta_pct:.1f}% vs {year_prev}")
        
        top_dsp_val = df[col_dsp].mode()[0] if col_dsp else "N/A"
        c4.metric("DSP Dominante", top_dsp_val)
        
        st.divider()
        
        # 2. Gráficas Principales
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            st.subheader("Evolución Anual")
            # Agrupación simple
            year_data = df.groupby('Year_Final').size().reset_index(name='Total')
            fig_year = px.bar(year_data, x='Year_Final', y='Total', template='plotly_white', text_auto=True, color='Total')
            fig_year.update_layout(font=dict(color="black"))
            st.plotly_chart(fig_year, use_container_width=True)
            
        with col_g2:
            st.subheader("Market Share (Top 10 DSPs)")
            if col_dsp:
                dsp_data = df[col_dsp].value_counts().nlargest(10).reset_index()
                dsp_data.columns = ['DSP', 'Total']
                fig_pie = px.pie(dsp_data, names='DSP', values='Total', hole=0.4, template='plotly_white')
                st.plotly_chart(fig_pie, use_container_width=True)

    # --------------------------------------------------------------------------
    # PESTAÑA 2: ANALISTA IA (Con espera automática)
    # --------------------------------------------------------------------------
    with tab_ai:
        st.header("Consultor Estratégico & Predictivo")
        st.info("💡 Este módulo usa IA para cálculos complejos, proyecciones y redacción de informes.")
        
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "Analista Senior listo. Puedo proyectar el 2026 o comparar periodos específicos. ¿Qué necesitas?"}]

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Ej: Diferencia Spotify Ene 2025 vs 2026 y proyecciones Q1"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                caja = st.empty()
                caja.info(f"🧠 Analizando con {sel_model}...")
                
                # --- SISTEMA DE ESPERA AUTOMÁTICA (Anti-Error 429) ---
                def call_ai_patient(prompt_text, model_name):
                    max_retries = 2
                    for i in range(max_retries):
                        try:
                            model = genai.GenerativeModel(model_name)
                            return model.generate_content(prompt_text)
                        except Exception as e:
                            error_str = str(e)
                            if "429" in error_str or "Quota" in error_str:
                                wait_time = 65 
                                caja.warning(f"⏳ Límite de tráfico alcanzado. Esperando {wait_time}s para reanudar...")
                                bar = st.progress(0)
                                for t in range(wait_time):
                                    time.sleep(1)
                                    bar.progress((t+1)/wait_time)
                                bar.empty()
                                caja.info("✅ Reanudando análisis...")
                                continue
                            else:
                                raise e
                    raise Exception("Error de conexión persistente.")

                code = None
                try:
                    # PROMPT DE ALTO NIVEL (Corregido)
                    prompt_sys = f"""
                    Eres un Data Scientist Senior experto en Python.
                    
                    OBJETIVO:
                    Crear un reporte visual y narrativo en Streamlit.
                    
                    DATOS DISPONIBLES (Ya en memoria):
                    - DataFrame: `df`
                    - Columnas Texto: {cols_clean}
                    - Fechas: Year_Final, Month_Final
                    - RESUMEN: {truth_table}
                    
                    USUARIO: "{prompt}"
                    
                    REGLAS DE PROGRAMACIÓN (CRÍTICAS):
                    1. **FILTRADO**: 
                       - INCORRECTO: `normalize_text(df['COL'])` -> Error de series.
                       - CORRECTO: `df['COL_CLEAN'] == normalize_text('Valor')`.
                    
                    2. **MATEMÁTICAS**:
                       - Si piden proyecciones: Usa `LinearRegression` con datos históricos.
                       - Agrupa por fecha, entrena y predice.
                    
                    3. **OUTPUT**:
                       - Usa `st.markdown` para escribir el análisis ejecutivo al final.
                       - Usa gráficas interactivas (`plotly`).
                    
                    Genera SOLO el código Python.
                    """

                    response = call_ai_patient(prompt_sys, sel_model)
                    code = response.text.replace("```python", "").replace("```", "").strip()
                    
                    caja.empty()
                    
                    exec_globals = {
                        "df": df, "pd": pd, "np": np, "st": st, "px": px, "go": go,
                        "LinearRegression": LinearRegression,
                        "normalize_text": normalize_text, "unicodedata": unicodedata, "io": io
                    }
                    exec(code, exec_globals)
                    st.session_state.messages.append({"role": "assistant", "content": "✅ Reporte generado."})

                except Exception as e:
                    caja.error(f"Error: {e}")
                    if code:
                        with st.expander("Ver código"): st.code(code)

    # --------------------------------------------------------------------------
    # PESTAÑA 3: DATOS CRUDOS
    # --------------------------------------------------------------------------
    with tab_raw:
        st.header("Inspector de Datos")
        st.dataframe(df, use_container_width=True)
