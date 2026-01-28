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
# 1. CONFIGURACIÓN VISUAL (MODO ALTO CONTRASTE FORZADO)
# ==============================================================================
st.set_page_config(
    page_title="ONErpm Data Analyst",
    page_icon="🎹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS EXTREMO: FUERZA TEXTO NEGRO EN TODOS LADOS
st.markdown("""
<style>
    /* Fondo */
    .stApp { background-color: #FFFFFF !important; }
    
    /* Texto General */
    body, p, li, h1, h2, h3, h4, span, div { 
        color: #000000 !important; 
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* Cajas de Información/Alerta (Azules/Verdes/Rojas) */
    .stAlert {
        background-color: #E6F3FF !important; /* Fondo Azul Claro */
        border: 1px solid #004E92 !important;
    }
    .stAlert p, .stAlert div {
        color: #000000 !important; /* Texto NEGRO en alertas */
    }

    /* Tarjetas de Métricas */
    div[data-testid="stMetric"] {
        background-color: #F8F9FA !important;
        border: 1px solid #D1D5DB;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    div[data-testid="stMetricLabel"] { color: #333333 !important; font-weight: bold; }
    div[data-testid="stMetricValue"] { color: #000000 !important; }

    /* Burbujas del Chat */
    .stChatMessage {
        background-color: #F3F4F6 !important;
        border: 1px solid #E5E7EB;
        color: #000000 !important;
    }
    
    /* Tablas */
    div[data-testid="stDataFrame"] { border: 1px solid #000; }
    div[data-testid="stDataFrame"] * { color: #000000 !important; }
    
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #F8F9FA !important; border-right: 1px solid #ddd; }
</style>
""", unsafe_allow_html=True)

# Validación API
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("🚨 FALTA API KEY.")
    st.stop()
else:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# ==============================================================================
# 2. MOTOR ETL (NORMALIZACIÓN Y RESUMEN)
# ==============================================================================
URL_SHEET = "https://docs.google.com/spreadsheets/d/10y2YowTEgQYdWxs6c8D0fgJDDwGIT8_wyH0rQbERgG0/edit?gid=1919114384#gid=1919114384"

def remove_accents(input_str):
    if not isinstance(input_str, str): return str(input_str)
    nfkd = unicodedata.normalize('NFKD', input_str)
    return "".join([c for c in nfkd if not unicodedata.combining(c)])

@st.cache_data(ttl=3600, show_spinner="📡 Descargando GSheets...")
def fetch_data():
    conn = st.connection("gsheets", type=GSheetsConnection)
    return conn.read(spreadsheet=URL_SHEET, worksheet="DSP COPY")

@st.cache_data(ttl=3600, show_spinner="🧹 Analizando Estructura...")
def clean_data(df):
    try:
        # 1. Encabezados
        df.columns = df.columns.astype(str).str.replace(r'\n', ' ', regex=True).str.strip()
        
        # 2. Texto (Sin tildes, mayúsculas)
        cols_txt = ['DSP', 'Artist', 'Title', 'Playlist', 'Genre', 'Territory', 'Origin']
        for c in cols_txt:
            if c in df.columns:
                df[f"{c}_CLEAN"] = df[c].apply(lambda x: remove_accents(str(x)).upper().strip() if pd.notnull(x) else "UNKNOWN")
        
        # 3. Fechas
        for col in df.columns:
            if 'Inclusion' in col or 'Release' in col:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # 4. Lógica de Año/Mes (Robusta)
        col_fecha = next((c for c in df.columns if 'Inclusion' in c), None)
        
        if col_fecha:
            df['Year_Final'] = df[col_fecha].dt.year.fillna(0).astype(int)
            df['Month_Final'] = df[col_fecha].dt.month.fillna(0).astype(int)
            
            # Fallback manual
            if 'Year' in df.columns:
                df['Year_Manual'] = pd.to_numeric(df['Year'], errors='coerce').fillna(0).astype(int)
                df['Year_Final'] = df.apply(lambda x: x['Year_Manual'] if x['Year_Final'] == 0 else x['Year_Final'], axis=1)
            
            if 'Month' in df.columns:
                mapa_mes = {'ENERO':1, 'ENE':1, 'JANUARY':1, 'JAN':1, '1':1, '01':1, 'FEBRERO':2, 'FEB':2, '02':2, '2':2,
                            'MARZO':3, 'MAR':3, '03':3, '3':3, 'ABRIL':4, 'ABR':4, '04':4, '4':4,
                            'MAYO':5, 'MAY':5, '05':5, '5':5, 'JUNIO':6, 'JUN':6, '06':6, '6':6,
                            'JULIO':7, 'JUL':7, '07':7, '7':7, 'AGOSTO':8, 'AGO':8, '08':8, '8':8,
                            'SEPTIEMBRE':9, 'SEP':9, '09':9, '9':9, 'OCTUBRE':10, 'OCT':10, '10':10,
                            'NOVIEMBRE':11, 'NOV':11, '11':11, 'DICIEMBRE':12, 'DIC':12, '12':12}
                def quick_month(x):
                    s = remove_accents(str(x)).upper().strip()
                    return int(s) if s.isdigit() else mapa_mes.get(s, 0)
                df['Month_Manual'] = df['Month'].apply(quick_month)
                df['Month_Final'] = df.apply(lambda x: x['Month_Manual'] if x['Month_Final'] == 0 else x['Month_Final'], axis=1)
        else:
            df['Year_Final'] = pd.to_numeric(df['Year'], errors='coerce').fillna(0).astype(int) if 'Year' in df.columns else 0
            df['Month_Final'] = pd.to_numeric(df['Month'], errors='coerce').fillna(0).astype(int) if 'Month' in df.columns else 0

        df = df[df['DSP_CLEAN'] != 'UNKNOWN']
        return df
    except Exception as e:
        st.error(f"Error ETL: {e}")
        return pd.DataFrame()

# ==============================================================================
# 3. INTERFAZ Y GENERADOR DE "VERDAD"
# ==============================================================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/19/Spotify_logo_without_text.svg/168px-Spotify_logo_without_text.svg.png", width=50)
    st.title("Control Panel")
    
    # Selector Modelo
    try:
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        opts = sorted(models, key=lambda x: 'flash' in x, reverse=True)
        sel_model = st.selectbox("Modelo:", opts)
    except:
        sel_model = "models/gemini-1.5-flash"
    
    st.divider()
    
    # Carga
    try:
        raw = fetch_data()
        df = clean_data(raw)
        st.success(f"Cargado: {len(df)} filas")
    except:
        st.stop()
        
    if st.button("🧹 Reiniciar Chat"):
        st.session_state.messages = []
        st.rerun()

# --- GENERADOR DE TABLA DE VERDAD (Python Puro) ---
# Creamos un resumen de TODOS los años y TODOS los DSPs para pasárselo a la IA.
# Así sabe que 2024 existe y cuántos tiene.
if not df.empty:
    pivot_truth = df.groupby(['Year_Final', 'DSP_CLEAN']).size().reset_index(name='Count')
    # Convertimos esto a texto para el Prompt
    truth_text = pivot_truth.to_string(index=False)
    
    with st.expander("Ver Resumen de Verdad (Python)"):
        st.text(truth_text)

# ==============================================================================
# 4. CHAT CON CONTEXTO COMPLETO
# ==============================================================================
if not df.empty:
    st.title("🎹 ONErpm Data Analyst")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hola. Tengo el reporte completo de todos los años (2024, 2025, 2026, etc). Pregúntame lo que sea."}]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ej: Comparativa Spotify 2024 vs 2025"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            caja = st.empty()
            caja.info(f"🧠 Analizando con {sel_model}...")
            
            try:
                # INYECCIÓN DE VERDAD MASIVA
                # Le damos a la IA la tabla resumen completa. Ya no puede decir "Asumo 0".
                
                prompt_sys = f"""
                Actúa como Senior Data Analyst (Python Expert).
                
                TABLA DE VERDAD (DATOS PRE-CALCULADOS POR PYTHON):
                {truth_text}
                
                INSTRUCCIONES:
                1. El usuario pregunta: "{prompt}"
                2. Mira la TABLA DE VERDAD arriba. Si el usuario pide 2024, busca 2024 en esa tabla.
                   - Si la tabla dice 'SPOTIFY 2024 120', entonces hay 120. NO ASUMAS 0.
                
                REGLAS DE CÓDIGO (VISUALIZACIÓN):
                1. Usa `df` (el DataFrame completo está disponible).
                2. Filtra `DSP_CLEAN` usando `remove_accents(str(x)).upper()`.
                3. GRAFICOS: Usa `plotly.express`.
                   - IMPORTANTE: Configura `template='plotly_white'` para que se vea bien.
                   - Fuerza texto negro: `fig.update_layout(font=dict(color='black'))`.
                4. KPIS: Usa `st.metric`.
                
                Genera SOLO código Python.
                """

                model = genai.GenerativeModel(sel_model)
                response = model.generate_content(prompt_sys)
                code = response.text.replace("```python", "").replace("```", "").strip()
                
                caja.empty()
                local_vars = {"df": df, "pd": pd, "st": st, "px": px, "go": go, "remove_accents": remove_accents}
                exec(code, {}, local_vars)
                
                st.session_state.messages.append({"role": "assistant", "content": "✅ Análisis completado."})

            except Exception as e:
                caja.error(f"Error: {e}")
