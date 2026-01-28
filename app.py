import streamlit as st
import pandas as pd
import google.generativeai as genai
from streamlit_gsheets import GSheetsConnection
import plotly.express as px
import plotly.graph_objects as go
import time
import re
import unicodedata

# ==============================================================================
# 1. CONFIGURACIÓN VISUAL (ALTO CONTRASTE)
# ==============================================================================
st.set_page_config(
    page_title="ONErpm Data Analyst",
    page_icon="🎹",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #FFFFFF !important; }
    p, h1, h2, h3, h4, li, span, label, div, th, td { 
        color: #000000 !important; 
        font-family: 'Helvetica Neue', sans-serif; 
    }
    div[data-testid="stMetric"] { 
        background-color: #F3F4F6 !important; 
        border: 1px solid #9CA3AF; 
        border-radius: 8px; 
    }
    div[data-testid="stMetricLabel"] { color: #374151 !important; font-weight: bold; }
    div[data-testid="stMetricValue"] { color: #000000 !important; font-weight: 800; }
    .stChatMessage { background-color: #F9FAFB !important; border: 1px solid #E5E7EB; }
    div[data-testid="stDataFrame"] { border: 1px solid #000; }
    [data-testid="stSidebar"] { background-color: #F8F9FA !important; border-right: 1px solid #E5E7EB; }
</style>
""", unsafe_allow_html=True)

if "GOOGLE_API_KEY" not in st.secrets:
    st.error("🚨 CRÍTICO: Falta API Key en Secrets.")
    st.stop()
else:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# ==============================================================================
# 2. MOTOR ETL (LIMPIEZA UNIVERSAL)
# ==============================================================================
def normalize_text(text):
    """LIMPIEZA PROFUNDA: Sin tildes, Mayúsculas, Sin espacios extra."""
    if not isinstance(text, str): return str(text)
    text = "".join(c for c in unicodedata.normalize('NFKD', text) if not unicodedata.combining(c))
    return text.upper().strip()

URL_SHEET = "https://docs.google.com/spreadsheets/d/10y2YowTEgQYdWxs6c8D0fgJDDwGIT8_wyH0rQbERgG0/edit?gid=1919114384#gid=1919114384"

@st.cache_data(ttl=3600, show_spinner="📡 Descargando datos...")
def load_data():
    conn = st.connection("gsheets", type=GSheetsConnection)
    return conn.read(spreadsheet=URL_SHEET, worksheet="DSP COPY")

@st.cache_data(ttl=3600, show_spinner="🧹 Ejecutando Limpieza Universal...")
def clean_dataframe(df):
    try:
        # 1. Limpieza de Encabezados (Quitar saltos de línea y espacios)
        df.columns = [str(c).upper().replace('\n', ' ').strip() for c in df.columns]
        
        # 2. LISTA MAESTRA DE COLUMNAS DE TEXTO A LIMPIAR
        # Estas son las que el usuario querrá analizar
        target_text_cols = [
            'DSP', 'ARTIST', 'TITLE', 'GENRE', 'SUB-GENRE', 
            'TERRITORY', 'ORIGIN', 'PLAYLIST', 'BUSINESS UNIT', 'FORMAT'
        ]
        
        cleaned_cols_log = []

        # BUCLE DE LIMPIEZA MASIVA
        for target in target_text_cols:
            # Buscamos la columna en el Excel (parecida al nombre target)
            match = next((c for c in df.columns if target in c), None)
            
            if match:
                # CREAMOS LA VERSIÓN _CLEAN AUTOMÁTICAMENTE
                clean_col_name = f"{target.replace('-', '_').replace(' ', '_')}_CLEAN" # Ej: SUB-GENRE -> SUB_GENRE_CLEAN
                df[clean_col_name] = df[match].apply(lambda x: normalize_text(str(x)) if pd.notnull(x) else "UNKNOWN")
                cleaned_cols_log.append(clean_col_name)
        
        # 3. TRATAMIENTO ESPECIAL DE FECHAS (AÑO Y MES)
        col_inclusion = next((c for c in df.columns if 'INCLUSION' in c), None)
        col_release = next((c for c in df.columns if 'RELEASE' in c), None)
        col_year_man = next((c for c in df.columns if 'YEAR' == c), None)
        col_month_man = next((c for c in df.columns if 'MONTH' == c), None)

        df['Year_Final'] = 0
        df['Month_Final'] = 0
        
        # Prioridad 1: Inclusion Date
        if col_inclusion:
            dt_inc = pd.to_datetime(df[col_inclusion], errors='coerce')
            df['Year_Final'] = dt_inc.dt.year.fillna(0).astype(int)
            df['Month_Final'] = dt_inc.dt.month.fillna(0).astype(int)
            
        # Prioridad 2: Rellenar con Manuales
        if col_year_man:
            y_man = pd.to_numeric(df[col_year_man], errors='coerce').fillna(0).astype(int)
            df['Year_Final'] = df.apply(lambda x: y_man[x.name] if x['Year_Final'] == 0 else x['Year_Final'], axis=1)

        if col_month_man:
            mapa_mes = {'ENERO':1, 'ENE':1, 'JAN':1, 'FEBRERO':2, 'FEB':2, 'MARZO':3, 'MAR':3,
                        'ABRIL':4, 'ABR':4, 'MAYO':5, 'MAY':5, 'JUNIO':6, 'JUN':6,
                        'JULIO':7, 'JUL':7, 'AGOSTO':8, 'AGO':8, 'SEPTIEMBRE':9, 'SEP':9,
                        'OCTUBRE':10, 'OCT':10, 'NOVIEMBRE':11, 'NOV':11, 'DICIEMBRE':12, 'DIC':12}
            
            def get_month(x):
                s = normalize_text(str(x))
                if s.isdigit(): return int(s)
                return mapa_mes.get(s, 0)
                
            m_man = df[col_month_man].apply(get_month)
            df['Month_Final'] = df.apply(lambda x: m_man[x.name] if x['Month_Final'] == 0 else x['Month_Final'], axis=1)

        # Filtro de seguridad (Filas vacías)
        if 'DSP_CLEAN' in df.columns:
            df = df[df['DSP_CLEAN'] != 'UNKNOWN']

        return df, cleaned_cols_log

    except Exception as e:
        st.error(f"Error ETL: {e}")
        return pd.DataFrame(), []

# ==============================================================================
# 3. INTERFAZ
# ==============================================================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/19/Spotify_logo_without_text.svg/168px-Spotify_logo_without_text.svg.png", width=50)
    st.title("Control Panel")
    
    try:
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        opts = sorted(models, key=lambda x: 'flash' in x, reverse=True)
        sel_model = st.selectbox("Modelo:", opts)
    except:
        sel_model = "models/gemini-1.5-flash"
    
    st.divider()
    
    raw_df = load_data()
    df, cols_clean = clean_dataframe(raw_df)
    
    if not df.empty:
        # VISOR DE COLUMNAS DISPONIBLES (Para que sepas qué puedes preguntar)
        with st.expander("✅ Columnas Listas para Analizar"):
            st.write(cols_clean)
        
        # TABLA DE VERDAD (AÑO | MES | DSP)
        pivot = df.groupby(['Year_Final', 'Month_Final', 'DSP_CLEAN']).size().reset_index(name='Count')
        pivot = pivot[pivot['Count'] > 0]
        truth_table = pivot.to_string(index=False)
            
    if st.button("🧹 Reiniciar"):
        st.session_state.messages = []
        st.rerun()

# ==============================================================================
# 4. CHAT
# ==============================================================================
if not df.empty:
    st.title("🎹 ONErpm Data Analyst")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": f"He limpiado y estandarizado {len(cols_clean)} columnas (Artistas, Géneros, Territorios, etc). ¿Qué quieres descubrir?"}]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ej: Top 5 Artistas en México en 2025"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            caja = st.empty()
            caja.info(f"🧠 Analizando...")
            
            try:
                # PROMPT UNIVERSAL
                prompt_sys = f"""
                Actúa como Data Analyst Experto.
                
                TABLA DE VERDAD (Referencia de Fechas/DSP):
                {truth_table}
                
                DATOS DISPONIBLES (df):
                - Columnas LIMPIAS (Úsalas para filtrar): {cols_clean}
                - Fechas: `Year_Final`, `Month_Final`.
                
                USUARIO: "{prompt}"
                
                INSTRUCCIONES:
                1. **Mapeo de Columnas**:
                   - Si preguntan por ARTISTAS -> Usa `ARTIST_CLEAN`.
                   - Si preguntan por GÉNERO -> Usa `GENRE_CLEAN`.
                   - Si preguntan por TERRITORIO/PAÍS -> Usa `TERRITORY_CLEAN`.
                   - Si preguntan por DSP -> Usa `DSP_CLEAN`.
                
                2. **Filtrado**:
                   - Usa SIEMPRE `normalize_text('Valor')` para comparar.
                   - Ejemplo: `df[df['TERRITORY_CLEAN'] == normalize_text('Mexico')]`.
                
                3. **Visualización**:
                   - `plotly.express` (px) con `template='plotly_white'`.
                   - Texto negro.
                
                Genera SOLO Python.
                """

                model = genai.GenerativeModel(sel_model)
                response = model.generate_content(prompt_sys)
                code = response.text.replace("```python", "").replace("```", "").strip()
                
                caja.empty()
                
                exec_globals = {
                    "df": df, "pd": pd, "st": st, "px": px, "go": go,
                    "normalize_text": normalize_text, "unicodedata": unicodedata
                }
                exec(code, exec_globals)
                
                st.session_state.messages.append({"role": "assistant", "content": "✅ Análisis completado."})

            except Exception as e:
                caja.error(f"Error: {e}")
                with st.expander("Ver código"):
                    st.code(code)
