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
# 1. CONFIGURACIÓN VISUAL (ALTO CONTRASTE Y LEGIBILIDAD)
# ==============================================================================
st.set_page_config(
    page_title="ONErpm Data Analyst",
    page_icon="🎹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS PARA FORZAR TEXTO NEGRO Y FONDOS CLAROS
st.markdown("""
<style>
    /* Fondo Global */
    .stApp { background-color: #FFFFFF !important; }
    
    /* Textos Universales */
    p, h1, h2, h3, h4, li, span, label, div {
        color: #000000 !important;
        font-family: 'Arial', sans-serif;
    }
    
    /* Métricas (KPIs) */
    div[data-testid="stMetric"] {
        background-color: #F3F4F6 !important;
        border: 1px solid #9CA3AF;
        border-radius: 8px;
        padding: 10px;
    }
    div[data-testid="stMetricLabel"] { color: #374151 !important; font-weight: bold; }
    div[data-testid="stMetricValue"] { color: #000000 !important; font-weight: 800; }
    
    /* Chat */
    .stChatMessage {
        background-color: #F9FAFB !important;
        border: 1px solid #E5E7EB;
    }
    
    /* Tablas de Datos */
    div[data-testid="stDataFrame"] { border: 1px solid #000; }
    
    /* Alertas */
    .stAlert { background-color: #EFF6FF !important; border: 1px solid #1D4ED8; }
</style>
""", unsafe_allow_html=True)

# API KEY
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("🚨 CRÍTICO: Falta API Key en Secrets.")
    st.stop()
else:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# ==============================================================================
# 2. FUNCIONES DE LIMPIEZA (ETL) - GLOBAL SCOPE
# ==============================================================================
def normalize_text(text):
    """Limpia texto: Mayúsculas, Sin Tildes, Sin Espacios."""
    if not isinstance(text, str): return str(text)
    # Quitar tildes (Á -> A)
    text = "".join(c for c in unicodedata.normalize('NFKD', text) if not unicodedata.combining(c))
    return text.upper().strip()

URL_SHEET = "https://docs.google.com/spreadsheets/d/10y2YowTEgQYdWxs6c8D0fgJDDwGIT8_wyH0rQbERgG0/edit?gid=1919114384#gid=1919114384"

@st.cache_data(ttl=3600, show_spinner="📡 Descargando datos...")
def load_data():
    conn = st.connection("gsheets", type=GSheetsConnection)
    return conn.read(spreadsheet=URL_SHEET, worksheet="DSP COPY")

@st.cache_data(ttl=3600, show_spinner="🧹 Normalizando base de datos...")
def clean_dataframe(df):
    try:
        # 1. Limpieza de nombres de columna
        df.columns = df.columns.astype(str).str.replace(r'\n', ' ', regex=True).str.strip()
        
        # 2. Normalización de Columnas de Texto (DSP, Artista, etc)
        cols_text = ['DSP', 'Artist', 'Title', 'Playlist', 'Genre', 'Territory']
        for col in cols_text:
            if col in df.columns:
                # Crea columna _CLEAN (ej: DSP -> DSP_CLEAN)
                df[f"{col}_CLEAN"] = df[col].apply(lambda x: normalize_text(str(x)) if pd.notnull(x) else "UNKNOWN")

        # 3. Lógica de Fechas (Extracción Real)
        # Priorizamos Inclusion Date, si no, Release Date
        col_fecha = next((c for c in df.columns if 'Inclusion' in c), None)
        if not col_fecha:
            col_fecha = 'Release Date' if 'Release Date' in df.columns else None

        if col_fecha:
            df[col_fecha] = pd.to_datetime(df[col_fecha], errors='coerce')
            df['Year_Final'] = df[col_fecha].dt.year.fillna(0).astype(int)
            df['Month_Final'] = df[col_fecha].dt.month.fillna(0).astype(int)
        else:
            # Si no hay fecha, intentamos usar columnas manuales Year/Month si existen
            df['Year_Final'] = pd.to_numeric(df.get('Year', 0), errors='coerce').fillna(0).astype(int)
            df['Month_Final'] = pd.to_numeric(df.get('Month', 0), errors='coerce').fillna(0).astype(int)

        # 4. Parche Manual: Si Year_Final es 0 pero hay columna Year, úsala
        if 'Year' in df.columns:
            df['Year_Manual'] = pd.to_numeric(df['Year'], errors='coerce').fillna(0).astype(int)
            # Solo sobrescribe si Year_Final era 0
            df['Year_Final'] = df.apply(lambda x: x['Year_Manual'] if x['Year_Final'] == 0 else x['Year_Final'], axis=1)

        # 5. Parche Manual: Meses en texto
        if 'Month' in df.columns:
            meses_map = {'ENERO':1, 'ENE':1, 'JAN':1, 'FEBRERO':2, 'FEB':2, 'MARZO':3, 'MAR':3,
                         'ABRIL':4, 'ABR':4, 'MAYO':5, 'MAY':5, 'JUNIO':6, 'JUN':6,
                         'JULIO':7, 'JUL':7, 'AGOSTO':8, 'AGO':8, 'SEPTIEMBRE':9, 'SEP':9,
                         'OCTUBRE':10, 'OCT':10, 'NOVIEMBRE':11, 'NOV':11, 'DICIEMBRE':12, 'DIC':12}
            
            def get_month_num(x):
                s = normalize_text(str(x))
                if s.isdigit(): return int(s)
                return meses_map.get(s, 0)
            
            df['Month_Manual'] = df['Month'].apply(get_month_num)
            df['Month_Final'] = df.apply(lambda x: x['Month_Manual'] if x['Month_Final'] == 0 else x['Month_Final'], axis=1)

        # 6. Filtro Final: Eliminar filas basura (sin DSP)
        df = df[df['DSP_CLEAN'] != 'UNKNOWN']
        
        return df

    except Exception as e:
        st.error(f"Error limpiando datos: {e}")
        return pd.DataFrame()

# ==============================================================================
# 3. INTERFAZ PRINCIPAL
# ==============================================================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/19/Spotify_logo_without_text.svg/168px-Spotify_logo_without_text.svg.png", width=50)
    st.title("Panel de Control")
    
    # Selector de Modelo
    try:
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        opts = sorted(models, key=lambda x: 'flash' in x, reverse=True)
        sel_model = st.selectbox("Modelo IA:", opts)
    except:
        sel_model = "models/gemini-1.5-flash"
    
    st.divider()
    
    # Carga de Datos
    raw_df = load_data()
    df = clean_dataframe(raw_df)
    
    if not df.empty:
        st.success(f"Datos Cargados: {len(df)} filas")
        
        # --- GENERADOR DE "VERDAD" (Summary Context) ---
        # Creamos una tabla resumen para inyectar en el prompt
        summary = df.groupby(['Year_Final', 'DSP_CLEAN']).size().reset_index(name='Conteo')
        truth_context = summary.to_string(index=False)
        
        with st.expander("Ver Resumen de Verdad (Técnico)"):
            st.text(truth_context)
            
    if st.button("🗑️ Reiniciar Chat"):
        st.session_state.messages = []
        st.rerun()

# ==============================================================================
# 4. CHAT INTELIGENTE
# ==============================================================================
if not df.empty:
    st.title("🎹 ONErpm Data Analyst (V12)")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hola. Tengo todos los datos normalizados. Puedo analizar cualquier año (2024, 2025, 2026...). ¿Qué necesitas?"}]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ej: Comparar Spotify Enero 2025 vs 2026"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            caja = st.empty()
            caja.info(f"🧠 Analizando con {sel_model}...")
            
            try:
                # LISTA DE DSPS VÁLIDOS (Para que la IA sepa cómo se escriben)
                valid_dsps = sorted(df['DSP_CLEAN'].unique())
                
                # PROMPT MAESTRO (INYECCIÓN DE CONTEXTO)
                prompt_sys = f"""
                Actúa como Experto en Python y Análisis de Datos.
                
                CONTEXTO DE DATOS (DataFrame `df`):
                - Columnas: `DSP_CLEAN`, `Year_Final`, `Month_Final`.
                - DSPs Válidos (YA NORMALIZADOS): {valid_dsps}
                
                TABLA DE RESUMEN REAL (Use esto para saber qué años tienen datos):
                {truth_context}
                
                USUARIO PREGUNTA: "{prompt}"
                
                INSTRUCCIONES DE CÓDIGO:
                1. **FILTRADO**:
                   - Si el usuario dice "Claro Música", filtra: `df[df['DSP_CLEAN'] == 'CLARO MUSICA']`.
                   - Si el usuario dice "Spotify", filtra: `df[df['DSP_CLEAN'] == 'SPOTIFY']`.
                   - Usa `Year_Final` y `Month_Final` para fechas.
                
                2. **VISUALIZACIÓN**:
                   - Usa `plotly.express` (px).
                   - Si piden "torta" o "distribución", usa `px.pie`.
                   - Si piden "comparación", usa `px.bar` con `text_auto=True`.
                   - IMPORTANTE: Usa `template='plotly_white'` para que se vea bien en fondo blanco.
                
                3. **RESPUESTA**:
                   - Calcula los números exactos y muéstralos con `st.metric`.
                   - Muestra la gráfica.
                   - IMPRIME DEBUG: `st.write(f"Filas encontradas: {{len(df_filtrado)}}")`.
                
                Genera SOLO código Python.
                """

                model = genai.GenerativeModel(sel_model)
                response = model.generate_content(prompt_sys)
                code = response.text.replace("```python", "").replace("```", "").strip()
                
                caja.empty()
                
                # --- EJECUCIÓN SEGURA (Fix Scope) ---
                # Pasamos 'normalize_text' y las librerías al entorno de ejecución
                exec_globals = {
                    "df": df, "pd": pd, "st": st, "px": px, "go": go,
                    "normalize_text": normalize_text, "unicodedata": unicodedata
                }
                
                exec(code, exec_globals)
                
                st.session_state.messages.append({"role": "assistant", "content": "✅ Análisis generado."})
                
                # --- PRUEBA DE FILTRO (VISOR DE EVIDENCIA) ---
                # Esto aparece siempre al final para que tú veas qué hizo la IA
                with st.expander("🔍 Auditoría Técnica (¿Qué filtró la IA?)"):
                    st.info("Si los números no cuadran, revisa si el código generado arriba usó el filtro correcto.")
                    st.code(code, language="python")

            except Exception as e:
                caja.error(f"Error de ejecución: {e}")
