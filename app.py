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
# 1. CONFIGURACIÓN VISUAL (ALTO CONTRASTE)
# ==============================================================================
st.set_page_config(
    page_title="ONErpm Enterprise Analyst",
    page_icon="🎹",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #FFFFFF !important; }
    p, h1, h2, h3, h4, li, span, label, div, th, td { 
        color: #000000 !important; 
        font-family: 'Segoe UI', sans-serif; 
    }
    div[data-testid="stMetric"] { 
        background-color: #F8F9FA !important; 
        border: 1px solid #DEE2E6; 
        border-radius: 8px; 
        border-left: 5px solid #007BFF;
    }
    div[data-testid="stMetricLabel"] { color: #495057 !important; font-weight: bold; }
    div[data-testid="stMetricValue"] { color: #000000 !important; font-weight: 800; font-size: 1.6rem !important; }
    .stChatMessage { background-color: #F1F3F5 !important; border: 1px solid #E9ECEF; }
    div[data-testid="stDataFrame"] { border: 1px solid #343A40; }
    [data-testid="stSidebar"] { background-color: #F8F9FA !important; border-right: 1px solid #DEE2E6; }
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

@st.cache_data(ttl=3600, show_spinner="📡 Conectando...")
def load_data():
    conn = st.connection("gsheets", type=GSheetsConnection)
    return conn.read(spreadsheet=URL_SHEET, worksheet="DSP COPY")

@st.cache_data(ttl=3600, show_spinner="🧹 Limpiando...")
def clean_dataframe(df):
    try:
        df.columns = [
            str(c).upper().replace('\n', ' ').replace('/', '_').replace('.', '').strip().replace(' ', '_') 
            for c in df.columns
        ]
        
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
            mapa_mes = {'ENERO':1, 'ENE':1, 'JAN':1, 'FEBRERO':2, 'FEB':2, 'MARZO':3, 'MAR':3,
                        'ABRIL':4, 'ABR':4, 'MAYO':5, 'MAY':5, 'JUNIO':6, 'JUN':6,
                        'JULIO':7, 'JUL':7, 'AGOSTO':8, 'AGO':8, 'SEPTIEMBRE':9, 'SEP':9,
                        'OCTUBRE':10, 'OCT':10, 'NOVIEMBRE':11, 'NOV':11, 'DICIEMBRE':12, 'DIC':12}
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

# ==============================================================================
# 3. INTERFAZ
# ==============================================================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/19/Spotify_logo_without_text.svg/168px-Spotify_logo_without_text.svg.png", width=50)
    st.title("ONErpm Suite")
    st.caption("v25.0 Hybrid Core")
    
    try:
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        opts = sorted(models, key=lambda x: 'pro' in x, reverse=True)
        sel_model = st.selectbox("Modelo Preferido:", opts, index=0)
    except:
        sel_model = "models/gemini-1.5-pro"
    
    st.divider()
    
    raw_df = load_data()
    df, cols_clean = clean_dataframe(raw_df)
    
    if not df.empty:
        col_dsp = next((c for c in cols_clean if 'DSP' in c), None)
        if col_dsp:
            pivot = df.groupby(['Year_Final', 'Month_Final', col_dsp]).size().reset_index(name='Count')
            pivot = pivot[pivot['Count'] > 0]
            truth_table = pivot.to_string(index=False)
        else:
            truth_table = "No DSP data."
            
        st.success(f"DB Online: {len(df)} registros")
        
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Data', index=False)
        st.download_button("📥 Descargar Excel", buffer, "clean_data.xlsx")

# ==============================================================================
# 4. CHAT (HYBRID CORE: PRO -> FLASH FALLBACK)
# ==============================================================================
if not df.empty:
    tab_dash, tab_chat = st.tabs(["📊 Dashboard", "🤖 Analista Predictivo"])

    with tab_dash:
        st.header("Resumen General")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Destaques", f"{len(df):,}")
        c2.metric("Año Actual (Registros)", len(df[df['Year_Final'] == 2026]))
        top_dsp = df[col_dsp].mode()[0] if col_dsp else "N/A"
        c3.metric("DSP #1", top_dsp)
        
        # Gráfica
        hist_data = df.groupby('Year_Final').size().reset_index(name='Total')
        fig = px.bar(hist_data, x='Year_Final', y='Total', title="Tendencia Histórica", template='plotly_white', text_auto=True)
        st.plotly_chart(fig, use_container_width=True)

    with tab_chat:
        st.header("Laboratorio de IA")
        st.markdown("**Capacidades:** Regresión Lineal, Comparativas, Filtros complejos.")
        
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "Hola. Estoy listo para calcular proyecciones y diferencias. ¿Qué necesitas?"}]

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Ej: Diferencia Spotify Enero 2025 vs 2026 y proyecciones Q1"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                caja = st.empty()
                caja.info(f"🧠 Calculando con {sel_model}...")
                
                # --- SISTEMA DE LLAMADA HÍBRIDA (EL SECRETO) ---
                def call_ai_hybrid(prompt_text, primary_model):
                    # Intento 1: Modelo Principal (Pro)
                    try:
                        model = genai.GenerativeModel(primary_model)
                        return model.generate_content(prompt_text)
                    except Exception as e:
                        error_str = str(e)
                        # Si falla por sobrecarga (429/503), cambiamos a Flash
                        if "429" in error_str or "503" in error_str:
                            caja.warning("⚠️ Modelo Pro saturado. Cambiando a 'Flash' (Alta Velocidad) para completar la tarea...")
                            time.sleep(2)
                            backup_model = "models/gemini-1.5-flash"
                            model_bk = genai.GenerativeModel(backup_model)
                            return model_bk.generate_content(prompt_text)
                        else:
                            raise e # Si es otro error, lo lanzamos

                code = None
                try:
                    # PROMPT MATEMÁTICO OPTIMIZADO
                    prompt_sys = f"""
                    Eres un Senior Data Scientist experto en Python + Scikit-Learn.
                    
                    ENTORNO:
                    - `df` CARGADO ({len(df)} filas).
                    - `normalize_text` CARGADA.
                    - Librerías: pandas (pd), numpy (np), LinearRegression (sklearn), plotly.express (px).
                    
                    METADATA:
                    - Texto: {cols_clean}
                    - Fechas: Year_Final, Month_Final.
                    - RESUMEN REAL: {truth_table}
                    
                    USUARIO: "{prompt}"
                    
                    INSTRUCCIONES TÉCNICAS:
                    1. **PROYECCIONES (LinearRegression)**:
                       - Si piden predicción Q1 2026:
                       - Filtra el DSP solicitado.
                       - Agrupa histórico (2023, 2024, 2025) por Año/Mes.
                       - Crea variables numéricas para el tiempo (ej: indice mensual consecutivo).
                       - Entrena el modelo. Predice Ene, Feb, Mar 2026.
                       - Muestra resultado con `st.metric` y gráfica de líneas.
                    
                    2. **COMPARATIVAS**:
                       - Filtra: `df[df['DSP_CLEAN'] == normalize_text('Spotify')]`.
                       - Compara conteos (`len()`) de periodos solicitados.
                       - Gráfica de torta (`px.pie`) si piden distribución.
                    
                    3. **REGLAS**:
                       - ¡NO INVENTES DATOS! Usa `df`.
                       - ¡NO USES COLUMNA 'Count'! No existe en `df`.
                    
                    Genera SOLO código Python.
                    """

                    response = call_ai_hybrid(prompt_sys, sel_model)
                    code = response.text.replace("```python", "").replace("```", "").strip()
                    
                    caja.empty()
                    
                    exec_globals = {
                        "df": df, "pd": pd, "np": np, "st": st, "px": px, "go": go,
                        "LinearRegression": LinearRegression,
                        "normalize_text": normalize_text, "unicodedata": unicodedata
                    }
                    exec(code, exec_globals)
                    
                    st.session_state.messages.append({"role": "assistant", "content": "✅ Cálculo completado."})

                except Exception as e:
                    caja.error(f"Error: {e}")
                    if code:
                        with st.expander("Ver código"):
                            st.code(code)
