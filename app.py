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
# 2. MOTOR ETL (LIMPIEZA TOTAL DE 21 COLUMNAS)
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

@st.cache_data(ttl=3600, show_spinner="🧹 Ejecutando Limpieza en TODAS las columnas...")
def clean_dataframe(df):
    try:
        # 1. Estandarizar Encabezados (UPPERCASE, sin saltos de linea)
        # Esto convierte "Inclusion Date \nMM/DD..." en "INCLUSION_DATE_MM_DD_YYYY"
        df.columns = [
            str(c).upper().replace('\n', ' ').replace('/', '_').replace('.', '').strip().replace(' ', '_') 
            for c in df.columns
        ]
        
        cleaned_cols_log = []

        # 2. BUCLE "FULL SPECTRUM": Limpiar TODAS las columnas de texto
        # Identificamos columnas que son objeto (texto) o que queremos forzar como texto
        for col in df.columns:
            # Ignoramos columnas que ya parecen ser conteos o índices internos
            if col not in ['YEAR', 'MONTH', 'WEEK', 'Q']: 
                # Creamos versión _CLEAN
                clean_name = f"{col}_CLEAN"
                df[clean_name] = df[col].apply(lambda x: normalize_text(str(x)) if pd.notnull(x) else "UNKNOWN")
                cleaned_cols_log.append(clean_name)
        
        # 3. LÓGICA DE TIEMPO (CRÍTICA PARA EL CHIVATO)
        # Buscamos columnas clave dentro de los nombres estandarizados
        col_inc = next((c for c in df.columns if 'INCLUSION' in c), None)
        col_year = next((c for c in df.columns if c == 'YEAR'), None)
        col_month = next((c for c in df.columns if c == 'MONTH'), None)

        df['Year_Final'] = 0
        df['Month_Final'] = 0
        
        # A. Intentar fecha de inclusión
        if col_inc:
            dt_inc = pd.to_datetime(df[col_inc], errors='coerce')
            df['Year_Final'] = dt_inc.dt.year.fillna(0).astype(int)
            df['Month_Final'] = dt_inc.dt.month.fillna(0).astype(int)
            
        # B. Rellenar con Manuales
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

        # Filtro de Seguridad: Buscamos alguna columna que se parezca a DSP para filtrar basura
        col_dsp_clean = next((c for c in cleaned_cols_log if 'DSP' in c), None)
        if col_dsp_clean:
            df = df[df[col_dsp_clean] != 'UNKNOWN']

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
        # VISOR DE COLUMNAS DISPONIBLES (AHORA DEBERÍAN SER TODAS)
        with st.expander(f"✅ {len(cols_clean)} Columnas Limpias"):
            st.write(cols_clean)
        
        # TABLA DE VERDAD (AÑO | MES | DSP)
        # Necesitamos saber cuál es la columna DSP limpia para agrupar
        col_dsp_clean = next((c for c in cols_clean if 'DSP' in c), None)
        
        if col_dsp_clean:
            pivot = df.groupby(['Year_Final', 'Month_Final', col_dsp_clean]).size().reset_index(name='Count')
            pivot = pivot[pivot['Count'] > 0]
            truth_table = pivot.to_string(index=False)
        else:
            truth_table = "No se detectó columna DSP."
            
    if st.button("🧹 Reiniciar"):
        st.session_state.messages = []
        st.rerun()

# ==============================================================================
# 4. CHAT
# ==============================================================================
if not df.empty:
    st.title("🎹 ONErpm Data Analyst")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": f"He procesado el 100% de las columnas. Puedo cruzar Artistas, Formatos, Origen, Business Unit... ¡lo que sea!"}]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ej: Distribución por Format en 2025"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            caja = st.empty()
            caja.info(f"🧠 Analizando...")
            
            try:
                # PROMPT UNIVERSAL V17
                prompt_sys = f"""
                Actúa como Data Analyst Experto.
                
                TABLA DE VERDAD (Referencia de Fechas/DSP):
                {truth_table}
                
                COLUMNAS LIMPIAS DISPONIBLES (df):
                {cols_clean}
                (Nota: Todas estas columnas han sido normalizadas: MAYÚSCULAS, SIN TILDES).
                
                FECHAS CALCULADAS: `Year_Final`, `Month_Final`.
                
                USUARIO: "{prompt}"
                
                INSTRUCCIONES:
                1. **Identificar Columna**: Busca la columna _CLEAN que mejor coincida con lo que pide el usuario.
                   - Ej: "Formato" -> `FORMAT_CLEAN`
                   - Ej: "Origen" -> `ORIGIN_CLEAN`
                   - Ej: "Business Unit" -> `BUSINESS_UNIT_CLEAN`
                
                2. **Filtrar**:
                   - Usa SIEMPRE `normalize_text('Valor')` para comparar valores de texto.
                
                3. **Visualizar**:
                   - `plotly.express` (px) con `template='plotly_white'`.
                   - Si piden conteos/totales, usa `st.metric`.
                
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
