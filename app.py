import streamlit as st
import pandas as pd
import google.generativeai as genai
from streamlit_gsheets import GSheetsConnection
import plotly.express as px
import plotly.graph_objects as go
import time
import re

# ==============================================================================
# 1. DISEÑO UX/UI PREMIUM
# ==============================================================================
st.set_page_config(
    page_title="ONErpm Data Analyst",
    page_icon="🎹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paleta de Colores ONErpm y Estilos
st.markdown("""
<style>
    /* Estilo Global */
    .stApp { background-color: #F8F9FA; }
    
    /* Encabezados */
    h1, h2, h3 { color: #004E92; font-family: 'Helvetica Neue', sans-serif; font-weight: 700; }
    
    /* Métricas */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    /* Chat Bubbles */
    .stChatMessage {
        background-color: #FFFFFF;
        border: 1px solid #F3F4F6;
        border-radius: 16px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        padding: 20px;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E5E7EB;
    }
</style>
""", unsafe_allow_html=True)

# API Key Check
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("🚨 Error de Configuración: Falta la API Key en `.streamlit/secrets.toml`")
    st.stop()
else:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# ==============================================================================
# 2. MOTOR DE DATOS (ETL & LIMPIEZA)
# ==============================================================================
# URL del archivo real
URL_SHEET = "https://docs.google.com/spreadsheets/d/10y2YowTEgQYdWxs6c8D0fgJDDwGIT8_wyH0rQbERgG0/edit?gid=1919114384#gid=1919114384"

@st.cache_data(ttl=600, show_spinner=False)
def get_data():
    conn = st.connection("gsheets", type=GSheetsConnection)
    try:
        # Carga directa de la pestaña
        df = conn.read(spreadsheet=URL_SHEET, worksheet="DSP COPY")
        
        # --- LIMPIEZA DE ENCABEZADOS ---
        # "Inclusion Date \nMM/DD..." -> "Inclusion Date MM/DD..."
        df.columns = df.columns.astype(str).str.replace(r'\n', ' ', regex=True).str.strip()
        
        # --- COLUMNAS NORMALIZADAS (Para la IA) ---
        # Creamos copias en MAYÚSCULAS para que el filtro sea insensible a mayúsculas/minúsculas
        cols_txt = ['DSP', 'Artist', 'Title', 'Playlist', 'Genre', 'Territory', 'Origin']
        for c in cols_txt:
            if c in df.columns:
                df[f"{c}_CLEAN"] = df[c].astype(str).fillna("UNKNOWN").str.strip().str.upper()
        
        # --- LIMPIEZA DE FECHAS (CRÍTICO) ---
        # 1. Año
        if 'Year' in df.columns:
            df['Year'] = pd.to_numeric(df['Year'], errors='coerce').fillna(0).astype(int)
            
        # 2. Mes (Mapeo Inteligente)
        if 'Month' in df.columns:
            mapa_mes = {
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
            def fix_month(x):
                if isinstance(x, (int, float)): return int(x)
                s = str(x).strip().upper()
                if s.isdigit(): return int(s)
                return mapa_mes.get(s, 0)
            
            df['Month_CLEAN'] = df['Month'].apply(fix_month)
        else:
            df['Month_CLEAN'] = 0

        # --- VALIDACIÓN FINAL ---
        # Solo mantenemos filas que tengan un DSP válido (eliminamos filas vacías del Excel)
        df = df[df['DSP_CLEAN'] != 'UNKNOWN']
        
        return df

    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        return None

with st.spinner('🔄 Sincronizando datos con ONErpm DSP COPY...'):
    df = get_data()

# ==============================================================================
# 3. BARRA LATERAL (AUDITORÍA)
# ==============================================================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/19/Spotify_logo_without_text.svg/168px-Spotify_logo_without_text.svg.png", width=50)
    st.title("Panel de Control")
    st.markdown("---")

    # Configuración IA
    try:
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        # Preferimos Flash para rapidez, Pro para lógica
        model_options = sorted(models, key=lambda x: 'flash' in x, reverse=True)
        sel_model = st.selectbox("Modelo IA:", model_options, index=0)
    except:
        sel_model = "models/gemini-1.5-flash"
        st.warning("⚠️ Modo Offline")

    if st.button("🧹 Limpiar Chat"):
        st.session_state.messages = []
        st.rerun()
    
    # --- AUDITOR DE LA VERDAD (Python Puro) ---
    if df is not None:
        st.markdown("### 🛡️ Auditoría en Vivo")
        st.caption("Datos reales en base de datos (Sin alucinaciones):")
        
        # Filtro Hardcoded para Spotify
        spot_df = df[df['DSP_CLEAN'] == 'SPOTIFY']
        
        # Contamos filas reales
        c25 = len(spot_df[(spot_df['Year'] == 2025) & (spot_df['Month_CLEAN'] == 1)])
        c26 = len(spot_df[(spot_df['Year'] == 2026) & (spot_df['Month_CLEAN'] == 1)])
        
        c1, c2 = st.columns(2)
        c1.metric("Spotify Ene 25", c25)
        c2.metric("Spotify Ene 26", c26)
        
        st.markdown(f"**Total Filas:** {len(df)}")

# ==============================================================================
# 4. CHAT PRINCIPAL
# ==============================================================================
if df is not None:
    # Pestañas para organizar
    tab1, tab2 = st.tabs(["💬 Asistente Virtual", "📊 Explorador de Datos"])

    with tab1:
        st.subheader("Analista de Destaques")
        st.caption("Haz preguntas sobre comparativas, tendencias y conteos.")

        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": f"Hola. He validado los datos. Para Spotify Enero, tengo **{c25}** registros en 2025 y **{c26}** en 2026. ¿Qué quieres analizar?"}]

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Ej: Diferencia porcentual entre 2025 y 2026"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                status = st.status("🧠 Procesando lógica...", expanded=True)
                
                try:
                    # PREPARAMOS DATOS "CHIVADOS" PARA LA IA
                    # Calculamos algunas métricas clave en Python para pasárselas en el prompt
                    # Esto evita que la IA tenga que calcular desde cero y se equivoque.
                    resumen_txt = f"AUDITORÍA PREVIA: Spotify Ene 2025 = {c25}, Spotify Ene 2026 = {c26}."
                    
                    dsps_list = list(df['DSP_CLEAN'].unique())
                    
                    prompt_sistema = f"""
                    Actúa como Senior Data Analyst (Python Expert).
                    
                    CONTEXTO DE DATOS:
                    - DataFrame `df` cargado.
                    - Columnas CLAVE: `DSP_CLEAN`, `Year` (int), `Month_CLEAN` (int).
                    - {resumen_txt} (USA ESTOS NÚMEROS COMO REFERENCIA).
                    
                    INSTRUCCIONES:
                    1. Escribe código Python para responder al usuario: "{prompt}"
                    2. **FILTRADO:**
                       - Usa `df['DSP_CLEAN'] == 'SPOTIFY'` (Mayúsculas).
                       - Usa `Year` y `Month_CLEAN`.
                       - ¡NO ALUCINES! Confía en los filtros simples.
                    
                    3. **VISUALIZACIÓN:**
                       - Usa `plotly.express` (px) para gráficas.
                       - Muestra `st.metric` para KPIs.
                    
                    4. **EVIDENCIA (IMPORTANTE):**
                       - Al final, muestra las primeras 5 filas del dataframe filtrado usando:
                       - `with st.expander("📂 Ver Evidencia (Filas usadas)"): st.dataframe(df_filtrado)`
                    
                    Genera SOLO código Python.
                    """

                    # Llamada a IA con Reintento
                    def call_api():
                        model = genai.GenerativeModel(sel_model)
                        return model.generate_content(prompt_sistema)

                    try:
                        response = call_api()
                    except Exception as e:
                        if "429" in str(e):
                            status.warning("Tráfico alto. Reintentando en 15s...")
                            time.sleep(15)
                            response = call_api()
                        else:
                            raise e

                    # Limpieza de código
                    match = re.search(r"```python(.*?)```", response.text, re.DOTALL)
                    code = match.group(1).strip() if match else response.text.replace("```python", "").replace("```", "").strip()
                    
                    status.write("Generando gráficas...")
                    
                    # Ejecución
                    local_env = {"df": df, "pd": pd, "st": st, "px": px, "go": go}
                    exec(code, {}, local_env)
                    
                    status.update(label="✅ Análisis Completado", state="complete", expanded=False)
                    st.session_state.messages.append({"role": "assistant", "content": "✅ Análisis generado."})

                except Exception as e:
                    status.update(label="❌ Error", state="error")
                    st.error(f"Error Técnico: {e}")

    # --- PESTAÑA 2: EXPLORADOR ---
    with tab2:
        st.header("Explorador de Datos Maestros")
        
        c1, c2 = st.columns(2)
        with c1:
            f_year = st.multiselect("Año", sorted(df['Year'].unique()), default=[2025, 2026])
        with c2:
            f_dsp = st.multiselect("DSP", sorted(df['DSP_CLEAN'].unique()), default=['SPOTIFY'])
            
        # Filtro Dinámico
        df_show = df[df['Year'].isin(f_year)]
        if f_dsp:
            df_show = df_show[df_show['DSP_CLEAN'].isin(f_dsp)]
            
        st.markdown(f"**Resultados:** {len(df_show)} filas")
        st.dataframe(df_show, use_container_width=True)
        
        if not df_show.empty:
            st.markdown("#### Resumen por Mes")
            pivot = df_show.groupby(['Year', 'Month_CLEAN']).size().reset_index(name='Cantidad')
            st.dataframe(pivot, use_container_width=True)

else:
    st.info("Cargando sistema...")
