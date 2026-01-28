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
# 1. UX/UI PREMIUM
# ==============================================================================
st.set_page_config(
    page_title="ONErpm Strategic Analyst",
    page_icon="🎹",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #FFFFFF !important; }
    h1, h2, h3, p, div, span, li, label { 
        color: #1F2937 !important; 
        font-family: 'Helvetica Neue', sans-serif; 
    }
    div[data-testid="stMetric"] { 
        background-color: #F9FAFB !important; 
        border: 1px solid #E5E7EB; 
        border-radius: 10px; 
        padding: 15px;
        border-left: 5px solid #005fcc;
    }
    div[data-testid="stMetricLabel"] { color: #6B7280 !important; font-size: 0.9rem; font-weight: 600; }
    div[data-testid="stMetricValue"] { color: #111827 !important; font-size: 1.8rem; font-weight: 800; }
    .stChatMessage { background-color: #F3F4F6 !important; border: 1px solid #E5E7EB; }
    div[data-testid="stDataFrame"] { border: 1px solid #E5E7EB; }
    [data-testid="stSidebar"] { background-color: #F9FAFB !important; border-right: 1px solid #E5E7EB; }
</style>
""", unsafe_allow_html=True)

if "GOOGLE_API_KEY" not in st.secrets:
    st.error("🚨 CRÍTICO: Falta API Key en Secrets.")
    st.stop()
else:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# ==============================================================================
# 2. MOTOR ETL (LIMPIEZA)
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

@st.cache_data(ttl=3600, show_spinner="🧹 Normalizando...")
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
# 3. INTERFAZ & EXPORTACIÓN
# ==============================================================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/19/Spotify_logo_without_text.svg/168px-Spotify_logo_without_text.svg.png", width=50)
    st.title("Enterprise Suite")
    st.caption("v28.0 Syntax Fix")
    
    valid_models_list = get_valid_models()
    if valid_models_list:
        default_idx = 0
        for i, m in enumerate(valid_models_list):
            if "pro" in m: default_idx = i; break
        sel_model = st.selectbox("Modelo:", valid_models_list, index=default_idx)
    else:
        st.error("No API Access")
        st.stop()
    
    st.divider()
    raw_df = load_data()
    df, cols_clean = clean_dataframe(raw_df)
    
    if not df.empty:
        col_dsp = next((c for c in cols_clean if 'DSP' in c), None)
        pivot = df.groupby(['Year_Final', 'Month_Final', col_dsp]).size().reset_index(name='Count') if col_dsp else pd.DataFrame()
        pivot = pivot[pivot['Count'] > 0]
        truth_table = pivot.to_string(index=False)
        st.success(f"DB: {len(df)} regs")
        
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='CleanData', index=False)
        st.download_button("📥 Excel Limpio", buffer, "onerpm_clean.xlsx")

# ==============================================================================
# 4. CHAT INTELLIGENCE
# ==============================================================================
if not df.empty:
    st.title("🎹 ONErpm Strategic Analyst")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Soy tu Consultor de Datos. Puedo generar reportes ejecutivos, proyecciones y análisis comparativos. ¿Qué necesitas?"}]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ej: Diferencia Spotify Ene 2025 vs 2026"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            caja = st.empty()
            caja.info(f"🧠 Generando Reporte Ejecutivo...")
            
            # --- FUNCIÓN HÍBRIDA ROBUSTA ---
            def call_ai_smart(prompt_text, main_model, available_list):
                try:
                    model = genai.GenerativeModel(main_model)
                    return model.generate_content(prompt_text)
                except Exception as e:
                    if "429" in str(e) or "503" in str(e):
                        fallback = next((m for m in available_list if 'flash' in m), None)
                        if fallback:
                            caja.warning(f"⚠️ Tráfico alto. Cambiando a {fallback}...")
                            time.sleep(4) # Pausa de seguridad
                            return genai.GenerativeModel(fallback).generate_content(prompt_text)
                    raise e

            code = None
            try:
                # PROMPT CORREGIDO (ANTI-ERROR SERIES)
                prompt_sys = f"""
                Eres un Data Scientist Senior.
                
                OBJETIVO:
                Generar un reporte completo en Streamlit (Gráficas + Texto Ejecutivo).
                
                DATOS (Ya cargados en variable `df`):
                - Columnas: {cols_clean}
                - Fechas: Year_Final, Month_Final.
                - RESUMEN: {truth_table}
                
                SOLICITUD: "{prompt}"
                
                REGLAS DE CÓDIGO CRÍTICAS (PARA EVITAR ERRORES):
                1. **FILTRADO SEGURO**:
                   - INCORRECTO: `normalize_text(df['COL']) == ...` (Esto causa error).
                   - CORRECTO: `df['COL_CLEAN'] == normalize_text('Valor')`.
                   - SIEMPRE usa las columnas _CLEAN ya existentes.
                
                2. **PROYECCIONES (LinearRegression)**:
                   - Entrena con histórico 2023-2025. Predice 2026.
                   - Usa `len()` para contar registros. NO uses 'Count'.
                
                3. **REPORTE EJECUTIVO**:
                   - Usa `st.markdown` al final.
                   - Explica los hallazgos: "El crecimiento fue de X%...".
                   - Interpreta la proyección: "Se espera una tendencia...".
                
                Genera SOLO código Python.
                """

                response = call_ai_smart(prompt_sys, sel_model, valid_models_list)
                code = response.text.replace("```python", "").replace("```", "").strip()
                
                caja.empty()
                
                exec_globals = {
                    "df": df, "pd": pd, "np": np, "st": st, "px": px, "go": go,
                    "LinearRegression": LinearRegression,
                    "normalize_text": normalize_text, "unicodedata": unicodedata,
                    "io": io
                }
                
                exec(code, exec_globals)
                
                st.session_state.messages.append({"role": "assistant", "content": "✅ Reporte Finalizado."})

            except Exception as e:
                caja.error(f"Error generando reporte: {e}")
                if code:
                    with st.expander("Ver código (Debug)"):
                        st.code(code)
