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
# 1. ARQUITECTURA VISUAL (UX/UI PREMIUM)
# ==============================================================================
st.set_page_config(
    page_title="ONErpm Enterprise Analytics",
    page_icon="🎹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inyección de CSS para corregir contrastes y dar look corporativo
st.markdown("""
<style>
    /* Estructura Principal */
    .stApp { background-color: #FFFFFF !important; }
    
    /* Tipografía */
    h1, h2, h3, p, div, span, li { 
        color: #1A1A1A !important; 
        font-family: 'Segoe UI', sans-serif;
    }
    
    /* Tarjetas de Métricas (KPIs) */
    div[data-testid="stMetric"] {
        background-color: #F8F9FA !important;
        border: 1px solid #E9ECEF;
        border-left: 5px solid #007BFF; /* Acento Azul */
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    div[data-testid="stMetricLabel"] { color: #6C757D !important; font-weight: 600; }
    div[data-testid="stMetricValue"] { color: #212529 !important; font-weight: 800; font-size: 1.8rem !important; }
    
    /* Chat */
    .stChatMessage { 
        background-color: #F8F9FA !important; 
        border: 1px solid #DEE2E6; 
        border-radius: 12px;
    }
    
    /* Tablas */
    div[data-testid="stDataFrame"] { border: 1px solid #DEE2E6; }
    
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #F8F9FA !important; border-right: 1px solid #DEE2E6; }
    
    /* Botones */
    .stButton button { width: 100%; border-radius: 6px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Validación de Seguridad
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("🚨 CRÍTICO: Falta API Key en Secrets.")
    st.stop()
else:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# ==============================================================================
# 2. MOTOR DE INGENIERÍA DE DATOS (ETL & CLEANING)
# ==============================================================================
def normalize_text(text):
    """Función crítica para estandarizar texto (Quita tildes, Espacios, Mayúsculas)."""
    if not isinstance(text, str): return str(text)
    text = "".join(c for c in unicodedata.normalize('NFKD', text) if not unicodedata.combining(c))
    return text.upper().strip()

URL_SHEET = "https://docs.google.com/spreadsheets/d/10y2YowTEgQYdWxs6c8D0fgJDDwGIT8_wyH0rQbERgG0/edit?gid=1919114384#gid=1919114384"

@st.cache_data(ttl=3600, show_spinner="📡 Conectando al Servidor de Datos...")
def load_data():
    conn = st.connection("gsheets", type=GSheetsConnection)
    return conn.read(spreadsheet=URL_SHEET, worksheet="DSP COPY")

@st.cache_data(ttl=3600, show_spinner="⚙️ Procesando Big Data & Limpieza...")
def clean_dataframe(df):
    try:
        # A. Limpieza de Encabezados
        df.columns = [
            str(c).upper().replace('\n', ' ').replace('/', '_').replace('.', '').strip().replace(' ', '_') 
            for c in df.columns
        ]
        
        cleaned_cols_log = []

        # B. Limpieza de Texto (Vectorizada)
        # Identificamos columnas de texto y las normalizamos
        ignore_cols = ['YEAR', 'MONTH', 'WEEK', 'Q', 'INCLUSION_DATE', 'RELEASE_DATE']
        for col in df.columns:
            if col not in ignore_cols:
                clean_name = f"{col}_CLEAN"
                df[clean_name] = df[col].apply(lambda x: normalize_text(str(x)) if pd.notnull(x) else "UNKNOWN")
                cleaned_cols_log.append(clean_name)
        
        # C. Ingeniería de Fechas (Time Intelligence)
        col_inc = next((c for c in df.columns if 'INCLUSION' in c), None)
        col_year = next((c for c in df.columns if c == 'YEAR'), None)
        col_month = next((c for c in df.columns if c == 'MONTH'), None)

        df['Year_Final'] = 0
        df['Month_Final'] = 0
        
        # C1. Extracción desde Fecha Completa (Prioridad Alta)
        if col_inc:
            dt_inc = pd.to_datetime(df[col_inc], errors='coerce')
            df['Year_Final'] = dt_inc.dt.year.fillna(0).astype(int)
            df['Month_Final'] = dt_inc.dt.month.fillna(0).astype(int)
            
        # C2. Fallback a columnas manuales (Si existen)
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

        # D. Filtro de Integridad (Eliminar basura)
        col_dsp = next((c for c in cleaned_cols_log if 'DSP' in c), None)
        if col_dsp: df = df[df[col_dsp] != 'UNKNOWN']

        return df, cleaned_cols_log

    except Exception as e:
        st.error(f"Error Fatal en ETL: {e}")
        return pd.DataFrame(), []

# ==============================================================================
# 3. BARRA LATERAL: CONTROL DE MISIÓN
# ==============================================================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/19/Spotify_logo_without_text.svg/168px-Spotify_logo_without_text.svg.png", width=60)
    st.title("ONErpm Suite")
    st.caption("v.24.0 Enterprise Edition")
    
    st.subheader("🧠 Motor de IA")
    try:
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        # Preferencia por PRO
        opts = sorted(models, key=lambda x: 'pro' in x, reverse=True)
        sel_model = st.selectbox("Modelo Activo:", opts)
    except:
        sel_model = "models/gemini-1.5-pro"
        st.warning("⚠️ Modo Offline")
    
    st.divider()
    
    # Carga de Datos
    raw_df = load_data()
    df, cols_clean = clean_dataframe(raw_df)
    
    if not df.empty:
        # Generamos Tabla de Verdad (Snapshot para la IA)
        col_dsp = next((c for c in cols_clean if 'DSP' in c), None)
        if col_dsp:
            pivot = df.groupby(['Year_Final', 'Month_Final', col_dsp]).size().reset_index(name='Count')
            pivot = pivot[pivot['Count'] > 0]
            truth_table = pivot.to_string(index=False)
        else:
            truth_table = "Sin datos de DSP."
            
        st.success(f"🟢 DB Conectada: {len(df):,} registros")
        
        # Herramienta de Descarga (Usa XlsxWriter)
        st.subheader("💾 Exportar Data")
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='CleanData', index=False)
        
        st.download_button(
            label="Descargar Excel Limpio (.xlsx)",
            data=buffer,
            file_name="onerpm_clean_data.xlsx",
            mime="application/vnd.ms-excel"
        )

# ==============================================================================
# 4. ÁREA PRINCIPAL (PESTAÑAS DE TRABAJO)
# ==============================================================================
if not df.empty:
    
    # Creamos pestañas para organizar la complejidad
    tab_dashboard, tab_chat, tab_data = st.tabs(["📊 Dashboard Ejecutivo", "🤖 Analista IA (Chat)", "🔎 Inspector de Datos"])

    # --------------------------------------------------------------------------
    # PESTAÑA 1: DASHBOARD (ESTADÍSTICAS AUTOMÁTICAS SIN IA)
    # --------------------------------------------------------------------------
    with tab_dashboard:
        st.header("Resumen Ejecutivo")
        
        # Métricas Globales
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Destaques Histórico", f"{len(df):,}")
        
        last_year = df['Year_Final'].max()
        this_year_count = len(df[df['Year_Final'] == last_year])
        c2.metric(f"Destaques {last_year}", f"{this_year_count:,}")
        
        if col_dsp:
            top_dsp = df[col_dsp].mode()[0]
            c3.metric("DSP Líder", top_dsp)
            
        # Gráfico Rápido: Evolución Anual
        chart_data = df.groupby('Year_Final').size().reset_index(name='Destaques')
        fig = px.bar(chart_data, x='Year_Final', y='Destaques', title="Crecimiento Anual", template='plotly_white', text_auto=True)
        fig.update_layout(font=dict(color="black"))
        st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------------------------------
    # PESTAÑA 2: CHATBOT (CON SCIKIT-LEARN Y REINTENTOS)
    # --------------------------------------------------------------------------
    with tab_chat:
        st.header("Analista Virtual Avanzado")
        st.markdown("Capaz de realizar: **Proyecciones (Linear Regression)**, **Comparativas**, y **Análisis Profundo**.")
        
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "Hola. Tengo acceso a algoritmos de predicción y a toda tu base de datos. ¿Qué calculamos?"}]

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Ej: Proyección Spotify Q1 2026"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                caja = st.empty()
                caja.info(f"🧠 Ejecutando modelo matemático con {sel_model}...")
                
                # --- FUNCIÓN ROBUSTA DE LLAMADA ---
                def call_ai_robust(prompt_text):
                    retries = 4
                    for i in range(retries):
                        try:
                            model = genai.GenerativeModel(sel_model)
                            return model.generate_content(prompt_text)
                        except Exception as e:
                            if "429" in str(e) or "503" in str(e):
                                sleep_t = 2 ** i * 3 # Backoff exponencial (3s, 6s, 12s...)
                                caja.warning(f"🚦 Reintentando conexión con Google... ({i+1}/{retries})")
                                time.sleep(sleep_t)
                            else:
                                raise e
                    raise Exception("Servidor Google saturado. Intenta de nuevo.")

                code = None
                try:
                    # PROMPT DE ALTA INGENIERÍA
                    prompt_sys = f"""
                    Eres un Senior Data Scientist experto en Python, Pandas y Scikit-Learn.
                    
                    CONTEXTO TÉCNICO:
                    - `df` ya está cargado en memoria.
                    - Librerías Disponibles: `pandas as pd`, `numpy as np`, `LinearRegression` (sklearn), `plotly.express as px`.
                    - `normalize_text` función disponible.
                    
                    METADATA DE DATOS:
                    - Columnas Texto Limpias: {cols_clean}
                    - Fechas: `Year_Final`, `Month_Final`.
                    
                    RESUMEN DE DATOS REALE (CHIVATO):
                    {truth_table}
                    
                    SOLICITUD: "{prompt}"
                    
                    REGLAS DE CODIFICACIÓN (STRICT):
                    1. **PROYECCIONES**: 
                       - Si piden predicciones, agrupa datos históricos.
                       - Crea un modelo `model = LinearRegression()`.
                       - Entrena con años pasados, predice el futuro.
                       - Muestra la proyección con `st.metric` y gráfica de linea.
                    
                    2. **MANEJO DE DATOS**:
                       - ¡PROHIBIDO CREAR DATOS FALSOS! Usa `df`.
                       - ¡PROHIBIDO USAR COLUMNA 'Count'! Usa `len(df_filtrado)`.
                       - Filtra strings con `normalize_text()`.
                    
                    3. **VISUALIZACIÓN**:
                       - Usa `plotly.express` con `template='plotly_white'`.
                       - Asegura que el texto de las gráficas sea visible (negro).
                    
                    Genera SOLO el bloque de código Python.
                    """

                    response = call_ai_robust(prompt_sys)
                    code = response.text.replace("```python", "").replace("```", "").strip()
                    
                    caja.empty()
                    
                    # ENTORNO DE EJECUCIÓN CON TODAS LAS HERRAMIENTAS
                    exec_globals = {
                        "df": df, "pd": pd, "np": np, "st": st, "px": px, "go": go,
                        "LinearRegression": LinearRegression, # SKLEARN HABILITADO
                        "normalize_text": normalize_text, "unicodedata": unicodedata
                    }
                    
                    exec(code, exec_globals)
                    
                    st.session_state.messages.append({"role": "assistant", "content": "✅ Cálculo finalizado."})

                except Exception as e:
                    caja.error(f"Error: {e}")
                    if code:
                        with st.expander("Ver código (Debug)"):
                            st.code(code)

    # --------------------------------------------------------------------------
    # PESTAÑA 3: INSPECTOR DE DATOS (VERDAD ABSOLUTA)
    # --------------------------------------------------------------------------
    with tab_data:
        st.header("Base de Datos Maestra")
        st.markdown("Vista directa de los datos limpios que utiliza la IA.")
        st.dataframe(df, use_container_width=True)
        
        with st.expander("Ver Tipos de Datos y Columnas"):
            st.write(df.dtypes)
