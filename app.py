import streamlit as st
import pandas as pd
import google.generativeai as genai
from streamlit_gsheets import GSheetsConnection
import matplotlib.pyplot as plt
import seaborn as sns
import re
import time

# -----------------------------------------------------------------------------
# 1. CONFIGURACIÓN DE PÁGINA Y ESTILO
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Data Analyst ONErpm", page_icon="🎹", layout="wide")

st.title("🎹 ONErpm Data Analyst (Modo Debug Total)")
st.markdown("---")

# -----------------------------------------------------------------------------
# 2. CONEXIÓN Y SELECCIÓN DE MODELO
# -----------------------------------------------------------------------------
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except Exception:
    st.error("❌ CRÍTICO: No se encontró la API Key en los Secrets.")
    st.stop()

@st.cache_resource
def get_robust_model():
    """Intenta obtener el modelo más estable y económico"""
    try:
        # Forzamos Flash 1.5 porque es rápido y consume menos cuota
        return 'models/gemini-1.5-flash' 
    except:
        return 'models/gemini-pro'

MODEL_NAME = get_robust_model()

# -----------------------------------------------------------------------------
# 3. CARGA Y "LAVADO" DE DATOS (ETL)
# -----------------------------------------------------------------------------
url_sheet = "https://docs.google.com/spreadsheets/d/10y2YowTEgQYdWxs6c8D0fgJDDwGIT8_wyH0rQbERgG0/edit?gid=1919114384#gid=1919114384"

@st.cache_data(ttl=600)
def load_data_expert():
    conn = st.connection("gsheets", type=GSheetsConnection)
    try:
        df = conn.read(spreadsheet=url_sheet, worksheet="DSP COPY")
        
        # --- LIMPIEZA MAESTRA (NORMALIZACIÓN) ---
        # 1. Limpiar nombres de columnas
        df.columns = df.columns.str.strip()
        
        # 2. Crear columnas "NORMALIZADAS" (Mayúsculas + Sin Espacios) para filtrado infalible
        # La IA usará estas columnas, no las originales sucias.
        cols_texto = ['DSP', 'Artist', 'Title', 'Playlist', 'Genre', 'Territory']
        for col in cols_texto:
            if col in df.columns:
                # Rellenar vacíos, convertir a string, quitar espacios, poner mayúsculas
                df[f"{col}_NORM"] = df[col].fillna("UNKNOWN").astype(str).str.strip().str.upper()

        # 3. Blindar Fechas y Números
        if 'Year' in df.columns:
            df['Year'] = pd.to_numeric(df['Year'], errors='coerce').fillna(0).astype(int)
        
        if 'Month' in df.columns:
            # Mapeo manual por si vienen en texto en español/inglés
            meses_map = {'enero':1, 'january':1, 'jan':1, 'febrero':2, 'february':2, 'feb':2} # Se puede extender
            # Si es texto, intentamos mapear. Si es numero, lo dejamos.
            df['Month'] = df['Month'].apply(lambda x: meses_map.get(str(x).lower(), x) if isinstance(x, str) and not x.isnumeric() else x)
            df['Month'] = pd.to_numeric(df['Month'], errors='coerce').fillna(0).astype(int)

        if 'Release Date' in df.columns:
            df['Release Date'] = pd.to_datetime(df['Release Date'], errors='coerce')

        return df
    except Exception as e:
        st.error(f"Error en ETL (Carga de datos): {e}")
        return None

with st.spinner('Realizando limpieza profunda de datos...'):
    df = load_data_expert()

# -----------------------------------------------------------------------------
# 4. BARRA LATERAL DE LA VERDAD (DEBUG DATA)
# -----------------------------------------------------------------------------
if df is not None:
    with st.sidebar:
        st.header("🔍 Panel de Control de Datos")
        st.info("Estos son los datos que Python ve ANTES de la IA.")
        
        st.write(f"**Total Destaques:** {len(df)}")
        
        # Auditoría de Años
        if 'Year' in df.columns:
            counts_year = df['Year'].value_counts().sort_index()
            st.write("**Conteo por Año:**")
            st.dataframe(counts_year)
            
        # Auditoría de DSPs
        if 'DSP_NORM' in df.columns:
            st.write("**DSPs Detectados:**")
            st.code(df['DSP_NORM'].unique())

# -----------------------------------------------------------------------------
# 5. MOTOR DE INTELIGENCIA (CHAT)
# -----------------------------------------------------------------------------

def extract_python_code(text):
    """Extrae quirúrgicamente solo el código Python"""
    pattern = r"```python(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match: return match.group(1).strip()
    return text.replace("```python", "").replace("```", "").strip()

if df is not None:
    # Inicializar historial
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({"role": "assistant", "content": "Soy tu Data Analyst. Uso datos normalizados para máxima precisión. ¿Qué analizamos?"})

    # Mostrar mensajes previos
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input Usuario
    if prompt := st.chat_input("Ej: Diferencia Spotify Enero 2025 vs 2026"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            caja = st.empty()
            caja.info("🧠 Generando lógica de análisis...")

            try:
                # --- PROMPT DE INGENIERÍA DE DATOS ---
                # Le damos las columnas NORMALIZADAS para que filtre por ahí
                columnas_disponibles = list(df.columns)
                unique_dsps = list(df['DSP_NORM'].unique()) if 'DSP_NORM' in df.columns else []
                
                prompt_maestro = f"""
                Actúa como Data Scientist Senior en Python.
                
                OBJETIVO: Responder: "{prompt}"
                
                DATOS DISPONIBLES (DataFrame `df`):
                - Columnas: {columnas_disponibles}
                - DSPs Disponibles (Usar columna 'DSP_NORM'): {unique_dsps}
                
                REGLAS DE ORO (PARA EVITAR ERRORES):
                1. **FILTRADO INFALIBLE**: 
                   - NO uses la columna 'DSP'. USA SIEMPRE `df['DSP_NORM']`.
                   - Al filtrar texto, usa MAYÚSCULAS. Ej: `df[df['DSP_NORM'] == 'SPOTIFY']`.
                
                2. **FILTRADO DE FECHAS**:
                   - Usa `Year` (int) y `Month` (int).
                   - Ej para Enero 2025: `df[(df['Year'] == 2025) & (df['Month'] == 1)]`
                
                3. **VERIFICACIÓN (DEBUG)**:
                   - Antes de mostrar el resultado final, IMPRIME cuántas filas encontraste.
                   - `st.write(f"Debug: Encontré {{len(df_filtrado)}} registros para... ")`
                   - Si len es 0, usa `st.warning("No encontré datos con estos filtros.")` y detente.
                
                4. **VISUALIZACIÓN**:
                   - Usa `st.metric(label="...", value="...")` para números clave.
                   - Gráficos: `fig, ax = plt.subplots()`, usa `sns.barplot`, finaliza con `st.pyplot(fig)`.
                
                Genera SOLO código Python.
                """
                
                # Llamada a la API con control de Errores (Retry)
                code = "" # Inicializamos variable para evitar NameError
                try:
                    model = genai.GenerativeModel(MODEL_NAME)
                    response = model.generate_content(prompt_maestro)
                    code = extract_python_code(response.text)
                except Exception as api_error:
                    if "429" in str(api_error):
                        st.error("🚦 Tráfico alto en la IA (Error 429). Espera 30 segundos y prueba de nuevo.")
                        st.stop()
                    else:
                        raise api_error

                caja.empty()
                
                # Ejecución del Código
                local_vars = {"df": df, "pd": pd, "st": st, "plt": plt, "sns": sns}
                exec(code, {}, local_vars)
                
                st.session_state.messages.append({"role": "assistant", "content": "✅ Análisis finalizado."})

            except Exception as e:
                caja.error(f"Error de Ejecución: {e}")
                with st.expander("Ver código que falló (Debug)"):
                    st.code(code if code else "No se generó código por error de API")
