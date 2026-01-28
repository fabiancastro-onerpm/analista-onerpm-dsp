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
# 1. CONFIGURACIÓN VISUAL
# ==============================================================================
st.set_page_config(
    page_title="ONErpm Enterprise Dashboard",
    page_icon="🎹",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #FFFFFF !important; }
    h1, h2, h3, p, div, span, li, label { color: #1F2937 !important; font-family: 'Segoe UI', sans-serif; }
    div[data-testid="stMetric"] { 
        background-color: #F8F9FA !important; 
        border: 1px solid #E9ECEF; 
        border-left: 5px solid #005fcc;
        border-radius: 8px; 
        padding: 15px;
    }
    div[data-testid="stMetricLabel"] { color: #6C757D !important; font-size: 0.9rem; font-weight: 700; }
    div[data-testid="stMetricValue"] { color: #212529 !important; font-size: 1.8rem; font-weight: 800; }
    .stChatMessage { background-color: #F8F9FA !important; border: 1px solid #E9ECEF; }
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

@st.cache_data(ttl=3600, show_spinner="⚙️ Procesando...")
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
# 3. BARRA LATERAL
# ==============================================================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/19/Spotify_logo_without_text.svg/168px-Spotify_logo_without_text.svg.png", width=50)
    st.title("ONErpm Suite")
    st.caption("v31.0 Unstoppable")
    
    valid_models_list = get_valid_models()
    # Identificar modelos PRO y FLASH
    pro_model = next((m for m in valid_models_list if "pro" in m), valid_models_list[0] if valid_models_list else None)
    flash_model = next((m for m in valid_models_list if "flash" in m), valid_models_list[0] if valid_models_list else None)
    
    sel_model = st.selectbox("Modelo Principal:", valid_models_list, index=valid_models_list.index(pro_model) if pro_model else 0)
    
    st.divider()
    raw_df = load_data()
    df, cols_clean = clean_dataframe(raw_df)
    
    if not df.empty:
        col_dsp = next((c for c in cols_clean if 'DSP' in c), None)
        pivot = df.groupby(['Year_Final', 'Month_Final', col_dsp]).size().reset_index(name='Count') if col_dsp else pd.DataFrame()
        pivot = pivot[pivot['Count'] > 0]
        truth_table = pivot.to_string(index=False)
        st.success(f"Datos: {len(df):,}")
        
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='CleanData', index=False)
        st.download_button("📥 Excel", buffer, "onerpm_clean.xlsx")

# ==============================================================================
# 4. DASHBOARD & CHAT (PROTOCOLOS DE EMERGENCIA)
# ==============================================================================
if not df.empty:
    
    tab_dash, tab_ai, tab_raw = st.tabs(["📊 Dashboard", "🤖 Analista IA", "🔎 Datos"])

    # --- TAB 1: DASHBOARD ---
    with tab_dash:
        st.header("Resumen de Datos")
        total = len(df)
        year_max = df['Year_Final'].max()
        count_ytd = len(df[df['Year_Final'] == year_max])
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Registros", f"{total:,}")
        c2.metric(f"Año {year_max}", f"{count_ytd:,}")
        c3.metric("DSP #1", df[col_dsp].mode()[0] if col_dsp else "N/A")
        
        g1, g2 = st.columns(2)
        with g1:
            st.caption("Histórico Anual")
            yd = df.groupby('Year_Final').size().reset_index(name='Total')
            st.plotly_chart(px.bar(yd, x='Year_Final', y='Total', template='plotly_white', text_auto=True), use_container_width=True)
        with g2:
            st.caption("Top 5 DSPs")
            if col_dsp:
                dd = df[col_dsp].value_counts().nlargest(5).reset_index()
                dd.columns = ['DSP', 'Total']
                st.plotly_chart(px.pie(dd, names='DSP', values='Total', hole=0.5, template='plotly_white'), use_container_width=True)

    # --- TAB 2: CHAT CON SUPERVIVENCIA ---
    with tab_ai:
        st.header("Consultor Inteligente")
        st.info("💡 Este chat cambia de modelo automáticamente si detecta sobrecarga en Google.")
        
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "Analista listo. Si un modelo falla, usaré otro. ¿Qué necesitas?"}]

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Ej: Proyección Spotify Q1 2026"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                caja = st.empty()
                
                # --- FUNCIÓN DE LLAMADA ULTRARRESILIENTE ---
                def call_ai_ultimate(prompt_text, primary_model_name, fallback_model_name):
                    # INTENTO 1: Modelo Principal (Pro)
                    try:
                        caja.info(f"🧠 Intentando con {primary_model_name}...")
                        return genai.GenerativeModel(primary_model_name).generate_content(prompt_text)
                    except Exception as e:
                        error_str = str(e)
                        
                        # INTENTO 2: Espera larga + Reintento Principal
                        if "429" in error_str or "Quota" in error_str or "503" in error_str:
                            wait_time = 70 # 70 segundos para asegurar
                            caja.warning(f"⏳ Google saturado. Esperando {wait_time}s para reintentar...")
                            
                            bar = st.progress(0)
                            for t in range(wait_time):
                                time.sleep(1)
                                bar.progress((t+1)/wait_time)
                            bar.empty()
                            
                            try:
                                caja.info(f"🔄 Reintentando con {primary_model_name}...")
                                return genai.GenerativeModel(primary_model_name).generate_content(prompt_text)
                            except Exception as e2:
                                # INTENTO 3: FALLBACK A FLASH (MODELO LIGERO)
                                caja.error(f"⚠️ El modelo principal falló de nuevo. Cambiando a EMERGENCIA ({fallback_model_name})...")
                                time.sleep(5)
                                return genai.GenerativeModel(fallback_model_name).generate_content(prompt_text)
                        
                        raise e # Si no es error de cuota, lanzar normal

                code = None
                try:
                    # PROMPT TÉCNICO
                    prompt_sys = f"""
                    Eres un Data Scientist Senior.
                    
                    DATOS DISPONIBLES (En memoria):
                    - `df`: DataFrame completo.
                    - Columnas Texto: {cols_clean}
                    - Fechas: Year_Final, Month_Final
                    - RESUMEN: {truth_table}
                    
                    SOLICITUD: "{prompt}"
                    
                    REGLAS OBLIGATORIAS (Evita errores de sintaxis):
                    1. **Filtrar Texto**: Usa `df['COL_CLEAN'] == normalize_text('Valor')`.
                       - NUNCA uses `normalize_text(df['col'])` -> Eso rompe el código.
                    
                    2. **Proyecciones**: 
                       - Usa `LinearRegression`.
                       - Entrena con histórico, predice futuro.
                       - No uses columna 'Count' (usa len).
                    
                    3. **Reporte**:
                       - Usa `st.markdown` para explicar resultados.
                       - Sé analítico ("Crecimiento del X%...").
                    
                    Genera SOLO código Python.
                    """

                    # Usamos el modelo Flash detectado como respaldo
                    fallback = flash_model if flash_model else sel_model 
                    
                    response = call_ai_ultimate(prompt_sys, sel_model, fallback)
                    code = response.text.replace("```python", "").replace("```", "").strip()
                    
                    caja.empty()
                    
                    exec_globals = {
                        "df": df, "pd": pd, "np": np, "st": st, "px": px, "go": go,
                        "LinearRegression": LinearRegression,
                        "normalize_text": normalize_text, "unicodedata": unicodedata, "io": io
                    }
                    exec(code, exec_globals)
                    st.session_state.messages.append({"role": "assistant", "content": "✅ Hecho."})

                except Exception as e:
                    caja.error(f"Error Final: {e}")
                    if code:
                        with st.expander("Código Fallido"): st.code(code)

    with tab_raw:
        st.header("Datos Crudos")
        st.dataframe(df)
