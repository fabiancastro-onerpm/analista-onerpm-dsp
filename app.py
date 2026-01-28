import streamlit as st
import pandas as pd
import google.generativeai as genai
from streamlit_gsheets import GSheetsConnection
import plotly.express as px
import plotly.graph_objects as go
import time
import re

# ==============================================================================
# 1. CONFIGURACIÓN VISUAL
# ==============================================================================
st.set_page_config(
    page_title="ONErpm Data Analyst",
    page_icon="🎹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS
st.markdown("""
<style>
    .stApp { background-color: #F8F9FA; }
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        padding: 10px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stChatMessage {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

# Validación de API
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("🚨 CRÍTICO: Falta API Key en Secrets.")
    st.stop()
else:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# ==============================================================================
# 2. MOTOR DE DATOS OPTIMIZADO (ETL POR FASES)
# ==============================================================================
URL_SHEET = "https://docs.google.com/spreadsheets/d/10y2YowTEgQYdWxs6c8D0fgJDDwGIT8_wyH0rQbERgG0/edit?gid=1919114384#gid=1919114384"

# FASE 1: DESCARGA CRUDA (Cacheada por 1 hora)
@st.cache_data(ttl=3600, show_spinner="📡 Descargando datos de Google Sheets...")
def fetch_raw_data():
    conn = st.connection("gsheets", type=GSheetsConnection)
    # Leemos sin procesar nada para ser rápidos
    return conn.read(spreadsheet=URL_SHEET, worksheet="DSP COPY")

# FASE 2: LIMPIEZA Y PROCESAMIENTO (Cacheada por 1 hora)
@st.cache_data(ttl=3600, show_spinner="🧹 Limpiando y organizando datos...")
def process_data(df):
    try:
        # 1. Limpieza de columnas (Vectorizada = Rápida)
        df.columns = df.columns.astype(str).str.replace(r'\n', ' ', regex=True).str.strip()
        
        # 2. Normalización de Texto
        cols_txt = ['DSP', 'Artist', 'Title', 'Playlist', 'Genre', 'Territory', 'Origin']
        for c in cols_txt:
            if c in df.columns:
                df[f"{c}_CLEAN"] = df[c].astype(str).fillna("UNKNOWN").str.strip().str.upper()
        
        # 3. Fechas (Inclusion / Release)
        # Convertimos todo lo que parezca fecha de una vez
        for col in df.columns:
            if 'Inclusion' in col or 'Release' in col:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # 4. Cálculo de AÑO y MES (Vectorizado)
        # Priorizamos Inclusion Date, luego Year manual
        col_fecha = next((c for c in df.columns if 'Inclusion' in c), None)
        
        if col_fecha:
            # Extraer Año y Mes de la fecha real
            df['Year_Final'] = df[col_fecha].dt.year.fillna(0).astype(int)
            df['Month_Final'] = df[col_fecha].dt.month.fillna(0).astype(int)
            
            # Si la fecha dio 0 (era NaT), intentamos usar la columna manual
            if 'Year' in df.columns:
                df['Year_Manual'] = pd.to_numeric(df['Year'], errors='coerce').fillna(0).astype(int)
                df['Year_Final'] = df.apply(lambda x: x['Year_Manual'] if x['Year_Final'] == 0 else x['Year_Final'], axis=1)
                
            if 'Month' in df.columns:
                # Mapa de meses manuales
                mapa_mes = {'ENERO':1, 'ENE':1, 'JANUARY':1, 'JAN':1, '1':1, '01':1, 'FEBRERO':2, 'FEB':2, '02':2, '2':2} # (Abreviado para velocidad, la IA entiende el resto)
                def quick_month(x):
                    s = str(x).strip().upper()
                    return int(s) if s.isdigit() else mapa_mes.get(s, 0)
                
                df['Month_Manual'] = df['Month'].apply(quick_month)
                df['Month_Final'] = df.apply(lambda x: x['Month_Manual'] if x['Month_Final'] == 0 else x['Month_Final'], axis=1)
        
        else:
            # Si no hay Inclusion Date, usamos manuales directo
            df['Year_Final'] = pd.to_numeric(df['Year'], errors='coerce').fillna(0).astype(int) if 'Year' in df.columns else 0
            df['Month_Final'] = pd.to_numeric(df['Month'], errors='coerce').fillna(0).astype(int) if 'Month' in df.columns else 0

        # Filtro de basura (Filas vacías)
        df = df[df['DSP_CLEAN'] != 'UNKNOWN']
        
        return df
    except Exception as e:
        st.error(f"Error procesando datos: {e}")
        return pd.DataFrame()

# ==============================================================================
# 3. CARGA CON DIAGNÓSTICO
# ==============================================================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/19/Spotify_logo_without_text.svg/168px-Spotify_logo_without_text.svg.png", width=40)
    st.title("Panel de Control")
    
    # Estado de Carga
    status_text = st.empty()
    bar = st.progress(0)
    
    try:
        status_text.text("Conectando GSheets...")
        raw_df = fetch_raw_data()
        bar.progress(50)
        
        status_text.text("Procesando fechas...")
        df = process_data(raw_df)
        bar.progress(100)
        time.sleep(0.5)
        bar.empty()
        status_text.empty()
        
    except Exception as e:
        st.error(f"Se trabó en la carga: {e}")
        st.stop()

    # AUDITORÍA RÁPIDA
    if not df.empty:
        st.markdown("---")
        st.caption("Auditoría en Vivo:")
        
        spot_df = df[df['DSP_CLEAN'] == 'SPOTIFY']
        c25 = len(spot_df[(spot_df['Year_Final'] == 2025) & (spot_df['Month_Final'] == 1)])
        c26 = len(spot_df[(spot_df['Year_Final'] == 2026) & (spot_df['Month_Final'] == 1)])
        
        c1, c2 = st.columns(2)
        c1.metric("Spot Ene 25", c25)
        c2.metric("Spot Ene 26", c26)

# ==============================================================================
# 4. CHAT
# ==============================================================================
if not df.empty:
    st.title("🎹 ONErpm Data Analyst")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": f"Datos cargados: {c25} registros en Ene 2025 y {c26} en Ene 2026. ¿Qué analizamos?"}]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ej: Diferencia Enero 2025 vs 2026"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            caja = st.empty()
            caja.info("🧠 Pensando...")
            
            try:
                # Prompt con los datos correctos inyectados
                dsps = list(df['DSP_CLEAN'].unique())
                info_chivada = f"DATOS REALES: Spotify Ene 2025 tiene {c25} filas. Spotify Ene 2026 tiene {c26} filas."
                
                prompt_sys = f"""
                Actúa como Data Analyst experto.
                
                TUS DATOS (DataFrame `df`):
                - Columnas CLAVE: `DSP_CLEAN`, `Year_Final` (int), `Month_Final` (int).
                - {info_chivada} (Usa esto como verdad absoluta).
                - DSPs: {dsps}
                
                INSTRUCCIONES:
                1. Filtra usando `DSP_CLEAN == 'SPOTIFY'`, `Year_Final` y `Month_Final`.
                2. Genera código Python con `plotly.express`.
                3. Usa `st.metric` para mostrar la diferencia numérica.
                4. IMPRIME: `st.write(f"Filas encontradas: {{len(df_filtrado)}}")`.
                
                Genera SOLO código Python.
                """

                # Selección de modelo
                model = genai.GenerativeModel("models/gemini-1.5-flash") # Flash es más rápido y estable
                response = model.generate_content(prompt_sys)
                
                # Limpieza
                clean_res = response.text.replace("```python", "").replace("```", "").strip()
                
                caja.empty()
                local_env = {"df": df, "pd": pd, "st": st, "px": px, "go": go}
                exec(clean_res, {}, local_env)
                
                st.session_state.messages.append({"role": "assistant", "content": "✅ Hecho."})

            except Exception as e:
                caja.error(f"Error: {e}")
