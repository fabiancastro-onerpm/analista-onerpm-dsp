import streamlit as st
import pandas as pd
import google.generativeai as genai
from streamlit_gsheets import GSheetsConnection
import plotly.express as px
import plotly.graph_objects as go
import time
import re

# ==============================================================================
# 1. CONFIGURACIÓN VISUAL Y ESTILOS (UX/UI PREMIUM)
# ==============================================================================
st.set_page_config(
    page_title="ONErpm Data Analyst | Enterprise",
    page_icon="🎹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS para apariencia corporativa
st.markdown("""
<style>
    /* Fondo y fuentes */
    .stApp {
        background-color: #f8f9fa;
    }
    h1, h2, h3 {
        color: #004e92;
        font-family: 'Inter', sans-serif;
    }
    /* Tarjetas de Métricas */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    /* Chat bubbles */
    .stChatMessage {
        background-color: #ffffff;
        border: 1px solid #f0f0f0;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        padding: 15px;
    }
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# Validación de API Key
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("🚨 CRÍTICO: Falta la API Key en los Secrets.")
    st.stop()
else:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# ==============================================================================
# 2. MOTOR ETL (SIN BORRAR FILAS - MODO LITERAL)
# ==============================================================================
URL_SHEET = "https://docs.google.com/spreadsheets/d/10y2YowTEgQYdWxs6c8D0fgJDDwGIT8_wyH0rQbERgG0/edit?gid=1919114384#gid=1919114384"

@st.cache_data(ttl=600, show_spinner=False)
def cargar_datos_maestros():
    conn = st.connection("gsheets", type=GSheetsConnection)
    try:
        # 1. Lectura
        df = conn.read(spreadsheet=URL_SHEET, worksheet="DSP COPY")
        
        # 2. Limpieza de Encabezados
        df.columns = df.columns.astype(str).str.replace(r'\n', ' ', regex=True).str.strip()
        
        # 3. Normalización de Texto (Clave para filtros)
        cols_texto = ['DSP', 'Artist', 'Title', 'Playlist', 'Genre', 'Territory', 'Origin']
        for col in cols_texto:
            if col in df.columns:
                df[f"{col}_CLEAN"] = df[col].astype(str).fillna("UNKNOWN").str.strip().str.upper()
        
        # 4. Lógica de Fechas (Extracción Robusta)
        # Priorizamos columna Inclusion Date si existe, sino usamos Year/Month manuales
        col_fecha_real = None
        for col in df.columns:
            if 'Inclusion' in col or 'Release' in col:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                if col_fecha_real is None and 'Inclusion' in col:
                    col_fecha_real = col
        
        # Calculamos Año y Mes desde la fecha real si es posible
        if col_fecha_real:
            df['Calculated_Year'] = df[col_fecha_real].dt.year.fillna(0).astype(int)
            df['Calculated_Month'] = df[col_fecha_real].dt.month.fillna(0).astype(int)
        else:
            df['Calculated_Year'] = 0
            df['Calculated_Month'] = 0

        # --- LÓGICA HÍBRIDA AÑO/MES ---
        # Si la columna manual Year existe y tiene datos, la usamos. Si es 0, usamos la calculada.
        if 'Year' in df.columns:
            df['Year'] = pd.to_numeric(df['Year'], errors='coerce').fillna(0).astype(int)
            # Rellenar ceros con el año calculado
            df['Year'] = df.apply(lambda x: x['Calculated_Year'] if x['Year'] == 0 else x['Year'], axis=1)
        else:
            df['Year'] = df['Calculated_Year']

        # Limpieza de Mes (Texto a Número)
        if 'Month' in df.columns:
            meses_map = {
                'ENERO':1, 'ENE':1, 'JANUARY':1, 'JAN':1, '1':1, '01':1,
                'FEBRERO':2, 'FEB':2, 'FEBRUARY':2, '2':2, '02':2,
                'MARZO':3, 'MAR':3, 'MARCH':3, '3':3, '03':3,
                'ABRIL':4, 'ABR':4, 'APRIL':4, '4':4, '04':4,
                'MAYO':5, 'MAY':5, '5':5, '05':5,
                'JUNIO':6, 'JUN':6, 'JUNE':6, '6':6, '06':6,
                'JULIO':7, 'JUL':7, 'JULY':7, '7':7, '07':7,
                'AGOSTO':8, 'AGO':8, 'AUGUST':8, '8':8, '08':8,
                'SEPTIEMBRE':9, 'SEP':9, 'SEPTEMBER':9, '9':9, '09':9,
                'OCTUBRE':10, 'OCT':10, 'OCTOBER':10, '10':10,
                'NOVIEMBRE':11, 'NOV':11, 'NOVEMBER':11, '11':11,
                'DICIEMBRE':12, 'DIC':12, 'DECEMBER':12, '12':12
            }
            def clean_month(val):
                if isinstance(val, (int, float)): return int(val)
                s = str(val).strip().upper()
                if s.isdigit(): return int(s)
                return meses_map.get(s, 0)
            
            df['Month_CLEAN'] = df['Month'].apply(clean_month)
            # Si sigue siendo 0, intentar con el calculado
            df['Month_CLEAN'] = df.apply(lambda x: x['Calculated_Month'] if x['Month_CLEAN'] == 0 else x['Month_CLEAN'], axis=1)
        else:
             df['Month_CLEAN'] = df['Calculated_Month']

        # Filtro de Seguridad: Ignorar filas sin DSP (Data vacía)
        df = df[df['DSP_CLEAN'] != 'UNKNOWN']
        
        return df

    except Exception as e:
        st.error(f"Error en Carga de Datos: {e}")
        return None

with st.spinner('🔄 Sincronizando con DSP COPY...'):
    df = cargar_datos_maestros()

# ==============================================================================
# 3. BARRA LATERAL (AUDITOR AUTOMÁTICO)
# ==============================================================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/19/Spotify_logo_without_text.svg/168px-Spotify_logo_without_text.svg.png", width=40)
    st.title("Panel de Control")
    st.markdown("---")
    
    # Selector de Modelo
    try:
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        model_options = sorted(models, key=lambda x: 'flash' in x, reverse=True)
        selected_model = st.selectbox("Modelo IA:", model_options, index=0)
    except:
        selected_model = "models/gemini-1.5-flash"
        st.warning("⚠️ Modo Offline")

    if st.button("🗑️ Nueva Conversación"):
        st.session_state.messages = []
        st.rerun()

    # --- AUDITOR DE LA VERDAD ---
    if df is not None:
        st.markdown("---")
        st.subheader("✅ Auditoría en Vivo")
        st.caption("Conteo real en base de datos (Sin IA):")
        
        # Calculamos Spotify Enero 2025 y 2026 manualmente con Python
        df_spot = df[df['DSP_CLEAN'] == 'SPOTIFY']
        
        c25 = len(df_spot[(df_spot['Year'] == 2025) & (df_spot['Month_CLEAN'] == 1)])
        c26 = len(df_spot[(df_spot['Year'] == 2026) & (df_spot['Month_CLEAN'] == 1)])
        
        col_a, col_b = st.columns(2)
        col_a.metric("Ene '25", c25) # Debería ser 43
        col_b.metric("Ene '26", c26) # Debería ser 28
        
        st.markdown(f"**Total Filas:** {len(df)}")

# ==============================================================================
# 4. INTERFAZ PRINCIPAL
# ==============================================================================
if df is not None:
    tab_chat, tab_data, tab_logs = st.tabs(["💬 Chat Analista", "📊 Datos Maestros", "⚙️ Logs"])

    # --- PESTAÑA 1: CHAT ---
    with tab_chat:
        st.subheader("Asistente Virtual ONErpm")
        st.caption(f"Conectado a la hoja: DSP COPY")
        
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": f"Hola. He verificado los datos: Tenemos **{c25}** destaques de Spotify en Ene 2025 y **{c26}** en Ene 2026. ¿Cómo quieres visualizar esto?"}]

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                if isinstance(msg["content"], str):
                    st.markdown(msg["content"])

        if prompt := st.chat_input("Ej: Gráfica de barras comparativa"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                status = st.status("🧠 Procesando...", expanded=True)
                
                try:
                    dsps = list(df['DSP_CLEAN'].unique())
                    
                    prompt_ia = f"""
                    Actúa como Senior Data Analyst para ONErpm.
                    
                    DATOS (DataFrame `df`):
                    - 1 Fila = 1 Destaque (Placement). NO borres duplicados.
                    - Columnas CLAVE: `DSP_CLEAN`, `Year` (int), `Month_CLEAN` (int).
                    - DSPs Válidos: {dsps}
                    
                    REGLAS DE NEGOCIO (STRICT):
                    1. FILTRO:
                       - `DSP_CLEAN == 'SPOTIFY'` (Mayúsculas).
                       - `Year` y `Month_CLEAN`.
                       - ¡NO ALUCINES DATOS! Usa `len(df_filtrado)` para contar.
                    
                    2. VISUALIZACIÓN:
                       - Usa `plotly.express` (px) para gráficas interactivas.
                       - Muestra métricas KPI con `st.metric`.
                       - Gráfico de barras comparativo es ideal para diferencias.
                    
                    3. DEBUG:
                       - Imprime el conteo real encontrado: `st.write(f"Encontré {{len(df_filtrado)}} filas...")`
                    
                    Genera SOLO código Python.
                    """

                    def call_ai():
                        model = genai.GenerativeModel(selected_model)
                        return model.generate_content(prompt_ia)
                    
                    # Manejo de error 429
                    try:
                        response = call_ai()
                    except Exception as e:
                        if "429" in str(e):
                            status.warning("Tráfico alto. Esperando 15s...")
                            time.sleep(15)
                            response = call_ai()
                        else:
                            raise e

                    match = re.search(r"```python(.*?)```", response.text, re.DOTALL)
                    code = match.group(1).strip() if match else response.text.replace("```python", "").replace("```", "").strip()
                    
                    status.write("Generando visualizaciones...")
                    
                    local_vars = {"df": df, "pd": pd, "st": st, "px": px, "go": go}
                    exec(code, {}, local_vars)
                    
                    status.update(label="✅ Análisis Completado", state="complete", expanded=False)
                    st.session_state.messages.append({"role": "assistant", "content": "✅ Análisis generado."})

                except Exception as e:
                    status.update(label="❌ Error", state="error")
                    st.error(f"Error técnico: {e}")

    # --- PESTAÑA 2: INSPECTOR (LA VERDAD) ---
    with tab_data:
        st.header("Datos Maestros")
        st.markdown("Filtra aquí para verificar que los datos coinciden con tu Excel.")
        
        col1, col2 = st.columns(2)
        with col1:
            filtro_anio = st.multiselect("Año", sorted(df['Year'].unique()), default=[2025, 2026])
        with col2:
            filtro_dsp = st.multiselect("DSP", sorted(df['DSP_CLEAN'].unique()), default=['SPOTIFY'])
            
        df_filtered = df[df['Year'].isin(filtro_anio)]
        if filtro_dsp:
            df_filtered = df_filtered[df_filtered['DSP_CLEAN'].isin(filtro_dsp)]
            
        st.dataframe(df_filtered, use_container_width=True)
        
        if not df_filtered.empty:
            st.markdown("#### Resumen Mensual (Prueba de Verdad)")
            resumen = df_filtered.groupby(['Year', 'Month_CLEAN']).size().reset_index(name='Cantidad Destaques')
            st.dataframe(resumen, use_container_width=True)

    # --- PESTAÑA 3: LOGS ---
    with tab_logs:
        st.json(list(df.columns))
        st.write(df.dtypes.astype(str))

else:
    st.error("Error cargando la aplicación.")
