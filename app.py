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
    p, h1, h2, h3, h4, li, span, label, div { color: #000000 !important; font-family: 'Helvetica Neue', sans-serif; }
    div[data-testid="stMetric"] { background-color: #F3F4F6 !important; border: 1px solid #9CA3AF; border-radius: 8px; }
    div[data-testid="stMetricLabel"] { color: #374151 !important; font-weight: bold; }
    div[data-testid="stMetricValue"] { color: #000000 !important; font-weight: 800; }
    .stChatMessage { background-color: #F9FAFB !important; border: 1px solid #E5E7EB; }
    div[data-testid="stDataFrame"] { border: 1px solid #000; }
    .stAlert { background-color: #EFF6FF !important; border: 1px solid #1D4ED8; }
</style>
""", unsafe_allow_html=True)

if "GOOGLE_API_KEY" not in st.secrets:
    st.error("🚨 CRÍTICO: Falta API Key en Secrets.")
    st.stop()
else:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# ==============================================================================
# 2. MOTOR ETL (LIMPIEZA EXACTA DE COLUMNAS)
# ==============================================================================
def normalize_text(text):
    """Limpia texto de celdas (Sin tildes, Mayúsculas)."""
    if not isinstance(text, str): return str(text)
    text = "".join(c for c in unicodedata.normalize('NFKD', text) if not unicodedata.combining(c))
    return text.upper().strip()

URL_SHEET = "https://docs.google.com/spreadsheets/d/10y2YowTEgQYdWxs6c8D0fgJDDwGIT8_wyH0rQbERgG0/edit?gid=1919114384#gid=1919114384"

@st.cache_data(ttl=3600, show_spinner="📡 Descargando datos...")
def load_data():
    conn = st.connection("gsheets", type=GSheetsConnection)
    return conn.read(spreadsheet=URL_SHEET, worksheet="DSP COPY")

@st.cache_data(ttl=3600, show_spinner="🧹 Estandarizando Columnas...")
def clean_dataframe(df):
    try:
        # --- PASO 1: LIMPIEZA AGRESIVA DE ENCABEZADOS ---
        # Convertimos todo a String -> Mayúsculas -> Quitamos saltos de linea -> Quitamos espacios extra
        # Esto convierte "DSP " en "DSP" y "Inclusion Date \nMM/DD" en "INCLUSION DATE MM/DD"
        df.columns = [str(c).upper().replace('\n', ' ').strip() for c in df.columns]
        
        # --- PASO 2: MAPEO DE COLUMNAS (BUSQUEDA INTELIGENTE) ---
        # Buscamos tus columnas clave dentro de los nombres sucios del Excel
        
        # Mapa: Nombre Interno -> Qué buscar en el Excel
        key_columns = {
            'DSP_REAL': 'DSP',
            'ARTIST_REAL': 'ARTIST',
            'INCLUSION_REAL': 'INCLUSION DATE',
            'YEAR_REAL': 'YEAR',
            'MONTH_REAL': 'MONTH'
        }
        
        found_cols = {}
        
        for internal_name, search_term in key_columns.items():
            # Buscamos si alguna columna del excel CONTIENE la palabra clave
            match = next((c for c in df.columns if search_term in c), None)
            if match:
                found_cols[internal_name] = match
        
        # --- PASO 3: PROCESAMIENTO ---
        
        # 3.1 DSP (Crítico)
        if 'DSP_REAL' in found_cols:
            col_name = found_cols['DSP_REAL']
            df['DSP_CLEAN'] = df[col_name].apply(lambda x: normalize_text(str(x)) if pd.notnull(x) else "UNKNOWN")
        else:
            st.error("❌ NO SE ENCONTRÓ LA COLUMNA 'DSP'. Revisa la lista de columnas en el sidebar.")
            df['DSP_CLEAN'] = "UNKNOWN"

        # 3.2 Artista
        if 'ARTIST_REAL' in found_cols:
            df['Artist_CLEAN'] = df[found_cols['ARTIST_REAL']].apply(lambda x: normalize_text(str(x)))

        # 3.3 Fechas (Inclusion Date)
        if 'INCLUSION_REAL' in found_cols:
            col_inc = found_cols['INCLUSION_REAL']
            df[col_inc] = pd.to_datetime(df[col_inc], errors='coerce')
            df['Year_Final'] = df[col_inc].dt.year.fillna(0).astype(int)
            df['Month_Final'] = df[col_inc].dt.month.fillna(0).astype(int)
        else:
            # Si falla inclusion, usamos Year/Month manuales
            year_col = found_cols.get('YEAR_REAL', 'YEAR')
            month_col = found_cols.get('MONTH_REAL', 'MONTH')
            
            df['Year_Final'] = pd.to_numeric(df.get(year_col, 0), errors='coerce').fillna(0).astype(int)
            df['Month_Final'] = pd.to_numeric(df.get(month_col, 0), errors='coerce').fillna(0).astype(int)

        # 3.4 Parche de Respaldo para Year/Month Manuales
        # Si la fecha de inclusión dio 0, intentamos leer las columnas manuales
        if 'YEAR_REAL' in found_cols:
            y_col = found_cols['YEAR_REAL']
            df['Year_Manual'] = pd.to_numeric(df[y_col], errors='coerce').fillna(0).astype(int)
            df['Year_Final'] = df.apply(lambda x: x['Year_Manual'] if x['Year_Final'] == 0 else x['Year_Final'], axis=1)

        if 'MONTH_REAL' in found_cols:
            m_col = found_cols['MONTH_REAL']
            meses_map = {'ENERO':1, 'ENE':1, 'JAN':1, 'FEBRERO':2, 'FEB':2, 'MARZO':3, 'MAR':3,
                         'ABRIL':4, 'ABR':4, 'MAYO':5, 'MAY':5, 'JUNIO':6, 'JUN':6,
                         'JULIO':7, 'JUL':7, 'AGOSTO':8, 'AGO':8, 'SEPTIEMBRE':9, 'SEP':9,
                         'OCTUBRE':10, 'OCT':10, 'NOVIEMBRE':11, 'NOV':11, 'DICIEMBRE':12, 'DIC':12}
            
            def get_month_num(x):
                s = normalize_text(str(x))
                if s.isdigit(): return int(s)
                return meses_map.get(s, 0)
            
            df['Month_Manual'] = df[m_col].apply(get_month_num)
            df['Month_Final'] = df.apply(lambda x: x['Month_Manual'] if x['Month_Final'] == 0 else x['Month_Final'], axis=1)

        # Filtro final
        df = df[df['DSP_CLEAN'] != 'UNKNOWN']
        
        return df, df.columns.tolist()

    except Exception as e:
        st.error(f"Error Crítico ETL: {e}")
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
    df, cols_detected = clean_dataframe(raw_df)
    
    # --- DIAGNÓSTICO DE COLUMNAS (PARA QUE VEAS QUE SÍ LAS LEYÓ) ---
    with st.expander("✅ Columnas Leídas (Check)"):
        st.write(cols_detected)

    if not df.empty:
        st.success(f"Cargado: {len(df)} filas")
        summary = df.groupby(['Year_Final', 'DSP_CLEAN']).size().reset_index(name='Conteo')
        truth_context = summary.to_string(index=False)
        with st.expander("Ver Tabla de Verdad"):
            st.text(truth_context)
            
    if st.button("🧹 Reiniciar"):
        st.session_state.messages = []
        st.rerun()

# ==============================================================================
# 4. CHAT
# ==============================================================================
if not df.empty:
    st.title("🎹 ONErpm Data Analyst")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hola. Columnas identificadas y mapeadas. ¿Qué analizamos?"}]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ej: Comparar Spotify Enero 2025 vs 2026"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            caja = st.empty()
            caja.info(f"🧠 Analizando...")
            
            try:
                valid_dsps = sorted(df['DSP_CLEAN'].unique())
                
                prompt_sys = f"""
                Actúa como Experto en Python.
                
                TABLA DE VERDAD (DATOS REALES YA CALCULADOS):
                {truth_context}
                
                DATOS (df):
                - Columnas: `DSP_CLEAN`, `Year_Final`, `Month_Final`.
                - DSPs: {valid_dsps}
                
                USUARIO: "{prompt}"
                
                INSTRUCCIONES:
                1. Filtra `DSP_CLEAN` con `normalize_text('Nombre')`.
                2. Usa `plotly.express` (template='plotly_white').
                3. KPIs con `st.metric`.
                
                Genera SOLO Python.
                """

                model = genai.GenerativeModel(sel_model)
                response = model.generate_content(prompt_sys)
                code = response.text.replace("```python", "").replace("```", "").strip()
                
                caja.empty()
                
                # --- FIX SCOPE COMPLETO ---
                exec_globals = {
                    "df": df, "pd": pd, "st": st, "px": px, "go": go,
                    "normalize_text": normalize_text, "unicodedata": unicodedata
                }
                exec(code, exec_globals)
                
                st.session_state.messages.append({"role": "assistant", "content": "✅ Hecho."})

            except Exception as e:
                caja.error(f"Error: {e}")
                with st.expander("Ver código"):
                    st.code(code)
