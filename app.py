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
# 1. CONFIGURACIÓN VISUAL (ALTO CONTRASTE & CLEAN UI)
# ==============================================================================
st.set_page_config(
    page_title="ONErpm Data Analyst",
    page_icon="🎹",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Estilos Globales de Alto Contraste */
    .stApp { background-color: #FFFFFF !important; }
    
    /* Tipografía Negra */
    p, h1, h2, h3, h4, li, span, label, div, th, td { 
        color: #000000 !important; 
        font-family: 'Helvetica Neue', sans-serif; 
    }
    
    /* Métricas */
    div[data-testid="stMetric"] { 
        background-color: #F8F9FA !important; 
        border: 1px solid #DEE2E6; 
        border-radius: 8px; 
    }
    div[data-testid="stMetricLabel"] { color: #495057 !important; font-weight: bold; }
    div[data-testid="stMetricValue"] { color: #000000 !important; font-weight: 800; }
    
    /* Chat */
    .stChatMessage { background-color: #F8F9FA !important; border: 1px solid #E9ECEF; }
    
    /* Tablas */
    div[data-testid="stDataFrame"] { border: 1px solid #343A40; }
    
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #F8F9FA !important; border-right: 1px solid #DEE2E6; }
</style>
""", unsafe_allow_html=True)

if "GOOGLE_API_KEY" not in st.secrets:
    st.error("🚨 ERROR: No se encontró la API Key en .streamlit/secrets.toml")
    st.stop()
else:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# ==============================================================================
# 2. MOTOR ETL (LIMPIEZA ROBUSTA)
# ==============================================================================
def normalize_text(text):
    """Limpia texto: Sin tildes, Mayúsculas, Trim."""
    if not isinstance(text, str): return str(text)
    text = "".join(c for c in unicodedata.normalize('NFKD', text) if not unicodedata.combining(c))
    return text.upper().strip()

URL_SHEET = "https://docs.google.com/spreadsheets/d/10y2YowTEgQYdWxs6c8D0fgJDDwGIT8_wyH0rQbERgG0/edit?gid=1919114384#gid=1919114384"

@st.cache_data(ttl=3600, show_spinner="📡 Conectando con Google Sheets...")
def load_data():
    conn = st.connection("gsheets", type=GSheetsConnection)
    return conn.read(spreadsheet=URL_SHEET, worksheet="DSP COPY")

@st.cache_data(ttl=3600, show_spinner="🧹 Procesando Base de Datos...")
def clean_dataframe(df):
    try:
        # 1. Limpieza de Encabezados
        df.columns = [
            str(c).upper().replace('\n', ' ').replace('/', '_').replace('.', '').strip().replace(' ', '_') 
            for c in df.columns
        ]
        
        cleaned_cols_log = []

        # 2. Limpieza Universal de Texto (Todas las columnas excepto numéricas clave)
        ignore_cols = ['YEAR', 'MONTH', 'WEEK', 'Q', 'INCLUSION_DATE', 'RELEASE_DATE']
        for col in df.columns:
            if col not in ignore_cols:
                clean_name = f"{col}_CLEAN"
                df[clean_name] = df[col].apply(lambda x: normalize_text(str(x)) if pd.notnull(x) else "UNKNOWN")
                cleaned_cols_log.append(clean_name)
        
        # 3. Ingeniería de Fechas
        col_inc = next((c for c in df.columns if 'INCLUSION' in c), None)
        col_year = next((c for c in df.columns if c == 'YEAR'), None)
        col_month = next((c for c in df.columns if c == 'MONTH'), None)

        df['Year_Final'] = 0
        df['Month_Final'] = 0
        
        # A. Prioridad: Fecha Completa
        if col_inc:
            dt_inc = pd.to_datetime(df[col_inc], errors='coerce')
            df['Year_Final'] = dt_inc.dt.year.fillna(0).astype(int)
            df['Month_Final'] = dt_inc.dt.month.fillna(0).astype(int)
            
        # B. Respaldo: Manuales
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

        # Filtro de Seguridad (Eliminar filas sin datos)
        # Buscamos la columna DSP limpia para filtrar vacíos
        col_dsp = next((c for c in cleaned_cols_log if 'DSP' in c), None)
        if col_dsp: 
            df = df[df[col_dsp] != 'UNKNOWN']

        return df, cleaned_cols_log

    except Exception as e:
        st.error(f"Error ETL: {e}")
        return pd.DataFrame(), []

# ==============================================================================
# 3. PANEL DE CONTROL
# ==============================================================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/19/Spotify_logo_without_text.svg/168px-Spotify_logo_without_text.svg.png", width=50)
    st.title("Control Panel")
    
    # Selector IA
    try:
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        opts = sorted(models, key=lambda x: 'flash' in x, reverse=True)
        sel_model = st.selectbox("Modelo IA:", opts)
    except:
        sel_model = "models/gemini-1.5-flash"
    
    st.divider()
    
    # Carga
    raw_df = load_data()
    df, cols_clean = clean_dataframe(raw_df)
    
    if not df.empty:
        # Generamos "Chivato" (Resumen) para la IA
        # Buscamos DSP clean
        col_dsp = next((c for c in cols_clean if 'DSP' in c), None)
        if col_dsp:
            pivot = df.groupby(['Year_Final', 'Month_Final', col_dsp]).size().reset_index(name='Count')
            # Solo mostramos filas con datos para ahorrar tokens
            pivot = pivot[pivot['Count'] > 0]
            truth_table = pivot.to_string(index=False)
        else:
            truth_table = "No se detectó columna DSP."
            
        st.success(f"Sistema Online: {len(df)} filas")
        with st.expander("Ver Datos Reales (Chivato)"):
            st.text(truth_table)
            
    if st.button("🧹 Reiniciar Conversación"):
        st.session_state.messages = []
        st.rerun()

# ==============================================================================
# 4. CHATBOT CON PROTECCIÓN ANTI-HARDCODING
# ==============================================================================
if not df.empty:
    st.title("🎹 ONErpm Data Analyst")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hola. Base de datos cargada y protegida. Hazme cualquier pregunta sobre tus datos."}]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ej: Diferencia porcentual Spotify Enero 2025 vs 2026"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            caja = st.empty()
            caja.info(f"🧠 Procesando...")
            
            try:
                # --- PROMPT V20: ANTI-HARDCODING EXTREMO ---
                prompt_sys = f"""
                Eres un Experto Data Analyst programando en Python dentro de Streamlit.
                
                CONTEXTO DE EJECUCIÓN (MUY IMPORTANTE):
                1. El código se ejecuta mediante `exec()`.
                2. La variable `df` YA ESTÁ CARGADA en el entorno global.
                3. La función `normalize_text` YA EXISTE.
                
                DATOS DISPONIBLES:
                - `df`: DataFrame principal con {len(df)} filas.
                - Columnas de Texto Limpias: {cols_clean}
                - Columnas de Fecha: `Year_Final`, `Month_Final`.
                
                TABLA DE VERDAD (Resumen pre-calculado para guiarte):
                {truth_table}
                
                SOLICITUD DEL USUARIO: "{prompt}"
                
                REGLAS DE ORO (VIOLARLAS CAUSA ERROR):
                ❌ 1. PROHIBIDO RECREAR DATOS:
                   - JAMÁS escribas `data = {{...}}`.
                   - JAMÁS escribas `df = pd.DataFrame(...)`.
                   - JAMÁS inventes listas de datos [2022, 2022...].
                   - USA LA VARIABLE `df` QUE YA EXISTE.
                
                ❌ 2. PROHIBIDO USAR COLUMNAS FANTASMA:
                   - El DataFrame `df` NO tiene columna 'Count'.
                   - Para contar, usa `len(df_filtrado)`.
                
                ✅ 3. INSTRUCCIONES DE FILTRADO:
                   - Para texto, usa: `df[df['COLUMNA_CLEAN'] == normalize_text('Valor')]`.
                   - Para fechas, usa: `Year_Final` y `Month_Final`.
                
                ✅ 4. VISUALIZACIÓN:
                   - Usa `plotly.express` con `template='plotly_white'`.
                   - Muestra resultados con `st.metric`.
                
                Genera EXCLUSIVAMENTE el código Python.
                """

                model = genai.GenerativeModel(sel_model)
                response = model.generate_content(prompt_sys)
                code = response.text.replace("```python", "").replace("```", "").strip()
                
                caja.empty()
                
                # Pasamos el entorno global completo para evitar errores de scope
                exec_globals = {
                    "df": df, "pd": pd, "st": st, "px": px, "go": go,
                    "normalize_text": normalize_text, "unicodedata": unicodedata
                }
                
                # Ejecución
                exec(code, exec_globals)
                
                st.session_state.messages.append({"role": "assistant", "content": "✅ Análisis completado."})

            except Exception as e:
                caja.error(f"Error en el análisis: {e}")
                with st.expander("Ver código generado (Debug)"):
                    st.code(code)
