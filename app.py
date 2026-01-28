import streamlit as st
import pandas as pd
import google.generativeai as genai
from streamlit_gsheets import GSheetsConnection
import plotly.express as px
import plotly.graph_objects as go
import time
import re

# ==============================================================================
# 1. CONFIGURACIÓN VISUAL (ESTILO "ENTERPRISE" LEGIBLE)
# ==============================================================================
st.set_page_config(
    page_title="ONErpm Data Analyst",
    page_icon="🎹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# INYECCIÓN CSS: Forzamos contraste alto (Letras oscuras, fondos claros)
st.markdown("""
<style>
    /* Fondo Principal - Gris muy suave para descansar la vista */
    .stApp { 
        background-color: #F0F2F6; 
    }
    
    /* Encabezados (H1, H2, H3) - Azul Corporativo ONErpm */
    h1, h2, h3 { 
        color: #004E92 !important; 
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
    }
    
    /* Tarjetas de Métricas (KPIs) - Blancas con texto negro */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF !important;
        border: 1px solid #D1D5DB;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    div[data-testid="stMetricLabel"] { 
        color: #4B5563 !important; /* Gris oscuro */
        font-size: 14px;
        font-weight: 600;
    }
    div[data-testid="stMetricValue"] { 
        color: #111827 !important; /* Negro casi puro */
        font-size: 24px;
        font-weight: 800;
    }

    /* Burbujas del Chat - Blancas con texto negro */
    .stChatMessage {
        background-color: #FFFFFF !important;
        border: 1px solid #E5E7EB;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    
    /* Texto general dentro del chat */
    .stMarkdown p, .stMarkdown li {
        color: #1F2937 !important; /* Gris oscuro para lectura óptima */
        font-size: 16px;
        line-height: 1.6;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E5E7EB;
    }
</style>
""", unsafe_allow_html=True)

# Validación de Seguridad
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("🚨 CRÍTICO: No se encontró la API Key en los Secrets (.streamlit/secrets.toml).")
    st.stop()
else:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# ==============================================================================
# 2. MOTOR DE DATOS (ETL ROBUSTO & CACHEADO)
# ==============================================================================
URL_SHEET = "https://docs.google.com/spreadsheets/d/10y2YowTEgQYdWxs6c8D0fgJDDwGIT8_wyH0rQbERgG0/edit?gid=1919114384#gid=1919114384"

# FASE 1: DESCARGA (Cache por 1 hora para velocidad)
@st.cache_data(ttl=3600, show_spinner="📡 Conectando con Google Sheets...")
def fetch_raw_data():
    conn = st.connection("gsheets", type=GSheetsConnection)
    # Leemos la pestaña exacta
    return conn.read(spreadsheet=URL_SHEET, worksheet="DSP COPY")

# FASE 2: LIMPIEZA INTELIGENTE
@st.cache_data(ttl=3600, show_spinner="🧹 Normalizando y estructurando datos...")
def process_data(df):
    try:
        # 1. Limpieza de encabezados (quita saltos de linea)
        df.columns = df.columns.astype(str).str.replace(r'\n', ' ', regex=True).str.strip()
        
        # 2. Normalización de Texto (Para filtros insensibles a mayúsculas)
        cols_txt = ['DSP', 'Artist', 'Title', 'Playlist', 'Genre', 'Territory', 'Origin']
        for c in cols_txt:
            if c in df.columns:
                df[f"{c}_CLEAN"] = df[c].astype(str).fillna("UNKNOWN").str.strip().str.upper()
        
        # 3. Conversión de Fechas
        for col in df.columns:
            if 'Inclusion' in col or 'Release' in col:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # 4. Lógica Maestra de Fechas (Año y Mes)
        # Intentamos obtener la fecha de inclusión real primero
        col_fecha = next((c for c in df.columns if 'Inclusion' in c), None)
        
        if col_fecha:
            df['Year_Final'] = df[col_fecha].dt.year.fillna(0).astype(int)
            df['Month_Final'] = df[col_fecha].dt.month.fillna(0).astype(int)
            
            # Si la fecha real falló (es 0), usamos las columnas manuales "Year" y "Month" como respaldo
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
                    s = str(x).strip().upper()
                    return int(s) if s.isdigit() else mapa_mes.get(s, 0)
                
                df['Month_Manual'] = df['Month'].apply(quick_month)
                df['Month_Final'] = df.apply(lambda x: x['Month_Manual'] if x['Month_Final'] == 0 else x['Month_Final'], axis=1)
        else:
            # Si no hay columna de inclusión, usamos manuales directos
            df['Year_Final'] = pd.to_numeric(df['Year'], errors='coerce').fillna(0).astype(int) if 'Year' in df.columns else 0
            df['Month_Final'] = pd.to_numeric(df['Month'], errors='coerce').fillna(0).astype(int) if 'Month' in df.columns else 0

        # Filtro de Seguridad: Eliminar filas vacías/basura
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
    
    # --- A. SELECTOR DE MODELO ---
    st.subheader("🧠 Configuración AI")
    try:
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        # Ponemos Flash primero (Más rápido, menos errores 429)
        model_options = sorted(models, key=lambda x: 'flash' in x, reverse=True)
        selected_model = st.selectbox("Modelo:", model_options, index=0, help="Flash: Rápido | Pro: Complejo")
    except:
        selected_model = "models/gemini-1.5-flash"
        st.warning("⚠️ Modo Offline (Default Flash)")
    
    st.markdown("---")
    
    # --- B. CARGA DE DATOS CON BARRA DE PROGRESO ---
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    try:
        status_text.caption("📡 1/2 Conectando...")
        raw_df = fetch_raw_data()
        progress_bar.progress(50)
        
        status_text.caption("🧹 2/2 Limpiando...")
        df = process_data(raw_df)
        progress_bar.progress(100)
        
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
        
    except Exception as e:
        st.error(f"Error de Carga: {e}")
        st.stop()

    # --- C. AUDITORÍA EN VIVO (LA VERDAD ABSOLUTA) ---
    if not df.empty:
        st.subheader("✅ Auditoría en Vivo")
        st.caption("Estos son los datos REALES en la base:")
        
        # Filtro de prueba (Spotify)
        spot_df = df[df['DSP_CLEAN'] == 'SPOTIFY']
        c25 = len(spot_df[(spot_df['Year_Final'] == 2025) & (spot_df['Month_Final'] == 1)])
        c26 = len(spot_df[(spot_df['Year_Final'] == 2026) & (spot_df['Month_Final'] == 1)])
        
        col1, col2 = st.columns(2)
        col1.metric("Ene 2025", c25)
        col2.metric("Ene 2026", c26)
        
        st.caption(f"Total Registros: {len(df)}")
        
    if st.button("🗑️ Reiniciar Chat"):
        st.session_state.messages = []
        st.rerun()

# ==============================================================================
# 4. CHAT ANALISTA (CON LÓGICA ANTI-ALUCINACIÓN)
# ==============================================================================
if not df.empty:
    st.title("🎹 ONErpm Data Analyst")
    st.caption("Herramienta de Inteligencia de Negocios y Visualización")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": f"¡Listo! Base de datos cargada. Según la auditoría, tenemos **{c25}** destaques de Spotify en Enero '25 y **{c26}** en Enero '26. ¿Qué deseas visualizar?"}]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ej: Comparativa visual Spotify Enero 2025 vs 2026"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            caja = st.empty()
            caja.info(f"🧠 Analizando solicitud con {selected_model}...")
            
            try:
                # Preparamos contexto
                dsps_list = list(df['DSP_CLEAN'].unique())
                # INYECCIÓN DE VERDAD: Le pasamos los datos auditados al prompt para que no invente
                truth_injection = f"AUDITORÍA PREVIA (NO LA CONTRADIGAS): Spotify Ene 2025 = {c25} filas. Spotify Ene 2026 = {c26} filas."
                
                prompt_sistema = f"""
                Actúa como Senior Data Analyst y Experto en Python.
                
                ESTRUCTURA DE DATOS (DataFrame `df`):
                - Columnas: `DSP_CLEAN`, `Year_Final`, `Month_Final`, `Artist_CLEAN`.
                - {truth_injection} (Usa estos números exactos).
                - DSPs Disponibles: {dsps_list}
                
                INSTRUCCIONES DEL USUARIO:
                "{prompt}"
                
                REGLAS DE CÓDIGO:
                1. **FILTRADO**: Usa `DSP_CLEAN == 'SPOTIFY'`, `Year_Final` y `Month_Final`.
                2. **VISUALIZACIÓN**: Usa `plotly.express` (px) para gráficos interactivos.
                   - Usa colores corporativos si es posible.
                   - Agrega `text_auto=True` a las barras.
                3. **KPIs**: Usa `st.metric` para mostrar las diferencias numéricas clave.
                4. **DEBUG**: Imprime siempre `st.write(f"Filas procesadas: {{len(df_filtrado)}}")`.
                
                Genera SOLO el bloque de código Python.
                """

                # Llamada a la API
                model = genai.GenerativeModel(selected_model)
                response = model.generate_content(prompt_sistema)
                
                # Limpieza de respuesta (Markdown)
                code_clean = response.text.replace("```python", "").replace("```", "").strip()
                
                caja.empty()
                
                # Ejecución Segura
                local_vars = {"df": df, "pd": pd, "st": st, "px": px, "go": go}
                exec(code_clean, {}, local_vars)
                
                st.session_state.messages.append({"role": "assistant", "content": "✅ Visualización generada con éxito."})

            except Exception as e:
                caja.error(f"Ocurrió un error: {e}")
                with st.expander("Ver detalle del error"):
                    st.write(e)
