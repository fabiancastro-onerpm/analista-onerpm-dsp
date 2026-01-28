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
# 1. CONFIGURACIÓN VISUAL (MODO "ALTO CONTRASTE" FORZADO)
# ==============================================================================
st.set_page_config(
    page_title="ONErpm Data Analyst",
    page_icon="🎹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS MAESTRO: Arregla Métricas, Chat y TABLAS
st.markdown("""
<style>
    /* 1. Fondo General - Gris muy claro */
    .stApp { 
        background-color: #F0F2F6; 
    }
    
    /* 2. Títulos - Azul Corporativo */
    h1, h2, h3 { 
        color: #004E92 !important; 
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
    }
    
    /* 3. Tarjetas de Métricas (KPIs) - Blancas con texto NEGRO */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF !important;
        border: 1px solid #D1D5DB;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    div[data-testid="stMetricLabel"] { color: #4B5563 !important; }
    div[data-testid="stMetricValue"] { color: #000000 !important; }

    /* 4. Chat Bubbles - Blancas con texto NEGRO */
    .stChatMessage {
        background-color: #FFFFFF !important;
        border: 1px solid #E5E7EB;
        color: #000000 !important;
    }
    .stMarkdown p, .stMarkdown li {
        color: #1F2937 !important; 
    }
    
    /* 5. ARREGLO DE TABLAS (Dataframes) */
    /* Fuerza el texto de las tablas a negro */
    div[data-testid="stDataFrame"] div[data-testid="stTable"] {
        color: #000000 !important;
        background-color: #FFFFFF !important;
    }
    /* Encabezados de tabla */
    div[class^="stDataFrame"] th {
        color: #004E92 !important;
        background-color: #E5E7EB !important;
    }
    /* Celdas de tabla */
    div[class^="stDataFrame"] td {
        color: #000000 !important;
    }
    
    /* 6. Sidebar */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E5E7EB;
    }
    /* Texto del Sidebar */
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] label {
        color: #000000 !important;
    }
</style>
""", unsafe_allow_html=True)

# Validación de Seguridad
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("🚨 CRÍTICO: No se encontró la API Key en los Secrets.")
    st.stop()
else:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# ==============================================================================
# 2. MOTOR DE DATOS (NORMALIZACIÓN PROFUNDA)
# ==============================================================================
URL_SHEET = "https://docs.google.com/spreadsheets/d/10y2YowTEgQYdWxs6c8D0fgJDDwGIT8_wyH0rQbERgG0/edit?gid=1919114384#gid=1919114384"

# Función Auxiliar: Quitar tildes (Claro Música -> CLARO MUSICA)
def remove_accents(input_str):
    if not isinstance(input_str, str):
        return str(input_str)
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

# FASE 1: DESCARGA
@st.cache_data(ttl=3600, show_spinner="📡 Conectando con Google Sheets...")
def fetch_raw_data():
    conn = st.connection("gsheets", type=GSheetsConnection)
    return conn.read(spreadsheet=URL_SHEET, worksheet="DSP COPY")

# FASE 2: LIMPIEZA
@st.cache_data(ttl=3600, show_spinner="🧹 Normalizando acentos y fechas...")
def process_data(df):
    try:
        # 1. Limpieza de encabezados
        df.columns = df.columns.astype(str).str.replace(r'\n', ' ', regex=True).str.strip()
        
        # 2. NORMALIZACIÓN DE TEXTO (FIX DE CLARO MÚSICA)
        # Convertimos a Mayúsculas Y quitamos tildes
        cols_txt = ['DSP', 'Artist', 'Title', 'Playlist', 'Genre', 'Territory', 'Origin']
        for c in cols_txt:
            if c in df.columns:
                df[f"{c}_CLEAN"] = df[c].apply(lambda x: remove_accents(str(x)).upper().strip() if pd.notnull(x) else "UNKNOWN")
        
        # 3. Conversión de Fechas
        for col in df.columns:
            if 'Inclusion' in col or 'Release' in col:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # 4. Lógica de Fechas (Año y Mes)
        col_fecha = next((c for c in df.columns if 'Inclusion' in c), None)
        
        if col_fecha:
            df['Year_Final'] = df[col_fecha].dt.year.fillna(0).astype(int)
            df['Month_Final'] = df[col_fecha].dt.month.fillna(0).astype(int)
            
            # Fallback manual
            if 'Year' in df.columns:
                df['Year_Manual'] = pd.to_numeric(df['Year'], errors='coerce').fillna(0).astype(int)
                df['Year_Final'] = df.apply(lambda x: x['Year_Manual'] if x['Year_Final'] == 0 else x['Year_Final'], axis=1)
            
            if 'Month' in df.columns:
                # Mapa de meses (incluyendo abreviaturas)
                mapa_mes = {'ENERO':1, 'ENE':1, 'JANUARY':1, 'JAN':1, '1':1, '01':1, 'FEBRERO':2, 'FEB':2, '02':2, '2':2,
                            'MARZO':3, 'MAR':3, '03':3, '3':3, 'ABRIL':4, 'ABR':4, '04':4, '4':4,
                            'MAYO':5, 'MAY':5, '05':5, '5':5, 'JUNIO':6, 'JUN':6, '06':6, '6':6,
                            'JULIO':7, 'JUL':7, '07':7, '7':7, 'AGOSTO':8, 'AGO':8, '08':8, '8':8,
                            'SEPTIEMBRE':9, 'SEP':9, '09':9, '9':9, 'OCTUBRE':10, 'OCT':10, '10':10,
                            'NOVIEMBRE':11, 'NOV':11, '11':11, 'DICIEMBRE':12, 'DIC':12, '12':12}
                
                def quick_month(x):
                    s = remove_accents(str(x)).upper().strip() # Quitamos tilde también al mes
                    return int(s) if s.isdigit() else mapa_mes.get(s, 0)
                
                df['Month_Manual'] = df['Month'].apply(quick_month)
                df['Month_Final'] = df.apply(lambda x: x['Month_Manual'] if x['Month_Final'] == 0 else x['Month_Final'], axis=1)
        else:
            df['Year_Final'] = pd.to_numeric(df['Year'], errors='coerce').fillna(0).astype(int) if 'Year' in df.columns else 0
            df['Month_Final'] = pd.to_numeric(df['Month'], errors='coerce').fillna(0).astype(int) if 'Month' in df.columns else 0

        # Filtro de Seguridad
        df = df[df['DSP_CLEAN'] != 'UNKNOWN']
        
        return df
    except Exception as e:
        st.error(f"Error procesando datos: {e}")
        return pd.DataFrame()

# ==============================================================================
# 3. BARRA LATERAL (CONTROL TOTAL)
# ==============================================================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/19/Spotify_logo_without_text.svg/168px-Spotify_logo_without_text.svg.png", width=50)
    st.title("Panel de Control")
    st.markdown("---")
    
    # A. Selector de Modelo
    st.subheader("🧠 Configuración AI")
    try:
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        model_options = sorted(models, key=lambda x: 'flash' in x, reverse=True)
        selected_model = st.selectbox("Modelo:", model_options, index=0)
    except:
        selected_model = "models/gemini-1.5-flash"
        st.warning("⚠️ Offline")
    
    st.markdown("---")
    
    # B. Carga de Datos
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    try:
        status_text.caption("📡 Conectando...")
        raw_df = fetch_raw_data()
        progress_bar.progress(50)
        
        status_text.caption("🧹 Limpiando...")
        df = process_data(raw_df)
        progress_bar.progress(100)
        
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
        
    except Exception as e:
        st.error(f"Error de Carga: {e}")
        st.stop()

    # C. AUDITORÍA DINÁMICA (SELECTOR DE DSP)
    if not df.empty:
        st.markdown("### ✅ Auditoría en Vivo")
        st.caption("Verifica aquí los datos REALES:")
        
        # Selector para que elijas qué auditar (CLARO MUSICA, SPOTIFY, ETC)
        dsp_options = sorted(df['DSP_CLEAN'].unique())
        # Intentamos poner SPOTIFY por defecto, si no el primero
        idx_def = dsp_options.index("SPOTIFY") if "SPOTIFY" in dsp_options else 0
        
        selected_audit_dsp = st.selectbox("DSP a Auditar:", dsp_options, index=idx_def)
        
        # Filtro dinámico
        audit_df = df[df['DSP_CLEAN'] == selected_audit_dsp]
        c25 = len(audit_df[(audit_df['Year_Final'] == 2025) & (audit_df['Month_Final'] == 1)])
        c26 = len(audit_df[(audit_df['Year_Final'] == 2026) & (audit_df['Month_Final'] == 1)])
        
        col1, col2 = st.columns(2)
        col1.metric("Ene '25", c25)
        col2.metric("Ene '26", c26)
        
    if st.button("🗑️ Reiniciar Chat"):
        st.session_state.messages = []
        st.rerun()

# ==============================================================================
# 4. CHAT ANALISTA
# ==============================================================================
if not df.empty:
    st.title("🎹 ONErpm Data Analyst")
    st.caption(f"Analizando {len(df)} registros. Modelo activo: {selected_model}")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": f"Datos listos. Auditoría actual para **{selected_audit_dsp}**: {c25} (2025) vs {c26} (2026). ¿Qué analizamos?"}]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input(f"Ej: Analizar crecimiento de {selected_audit_dsp}"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            caja = st.empty()
            caja.info(f"🧠 Analizando con {selected_model}...")
            
            try:
                # Contexto
                dsps_list = list(df['DSP_CLEAN'].unique())
                # Le pasamos la auditoría DEL DSP QUE EL USUARIO ELIGIÓ EN EL SIDEBAR
                truth_injection = f"AUDITORÍA REAL para {selected_audit_dsp}: Ene 2025 = {c25}, Ene 2026 = {c26}."
                
                prompt_sistema = f"""
                Actúa como Data Analyst experto.
                
                DATOS (DataFrame `df`):
                - Columnas: `DSP_CLEAN`, `Year_Final`, `Month_Final`, `Artist_CLEAN`.
                - {truth_injection} (Si el usuario pregunta por {selected_audit_dsp}, usa estos números).
                - DSPs Disponibles: {dsps_list}
                
                USUARIO: "{prompt}"
                
                INSTRUCCIONES PYTHON:
                1. Filtra `DSP_CLEAN` usando `remove_accents(str(x)).upper()`.
                   - Ejemplo: `df[df['DSP_CLEAN'] == 'CLARO MUSICA']` (SIN TILDES).
                2. Genera gráficos `plotly.express` (px) con `text_auto=True`.
                3. Usa `st.metric` para KPIs.
                4. DEBUG: `st.write(f"Filas filtradas: {{len(df_filtrado)}}")`.
                
                Genera SOLO código Python.
                """

                model = genai.GenerativeModel(selected_model)
                response = model.generate_content(prompt_sistema)
                code_clean = response.text.replace("```python", "").replace("```", "").strip()
                
                caja.empty()
                local_vars = {"df": df, "pd": pd, "st": st, "px": px, "go": go, "remove_accents": remove_accents}
                exec(code_clean, {}, local_vars)
                
                st.session_state.messages.append({"role": "assistant", "content": "✅ Visualización generada."})

            except Exception as e:
                caja.error(f"Error: {e}")
