import streamlit as st
import pandas as pd
import google.generativeai as genai
from streamlit_gsheets import GSheetsConnection
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURACIÓN VISUAL ---
st.set_page_config(page_title="Analista ONErpm AI", page_icon="🎹", layout="centered")
st.title("🎹 Chat con Datos ONErpm")
st.caption("Modo: Análisis de Destaques (Placements) - Lógica Completa")
st.markdown("---")

# --- 1. CONEXIÓN API ---
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except Exception:
    st.error("⚠️ Error: No se detectó la API Key en los Secrets.")
    st.stop()

# --- 2. AUTO-DETECTAR MODELO ---
@st.cache_resource
def get_best_model():
    try:
        available_models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                available_models.append(m.name)
        
        # Preferencia de modelos
        preferences = ['models/gemini-1.5-flash', 'models/gemini-1.5-pro', 'models/gemini-pro']
        for pref in preferences:
            if pref in available_models:
                return pref
        if available_models: return available_models[0]
        return None
    except Exception:
        return None

valid_model_name = get_best_model()
if not valid_model_name:
    st.error("❌ Error Crítico: Habilita 'Generative Language API' en Google Cloud.")
    st.stop()

# --- 3. CARGA DE DATOS ---
url_sheet = "https://docs.google.com/spreadsheets/d/10y2YowTEgQYdWxs6c8D0fgJDDwGIT8_wyH0rQbERgG0/edit?gid=1919114384#gid=1919114384"

@st.cache_data(ttl=600)
def load_data():
    conn = st.connection("gsheets", type=GSheetsConnection)
    try:
        df = conn.read(spreadsheet=url_sheet, worksheet="DSP COPY")
        
        # Procesamiento inteligente de fechas
        if 'Release Date' in df.columns:
            df['Release Date'] = pd.to_datetime(df['Release Date'], errors='coerce')
        if 'Inclusion Date' in df.columns:
            df['Inclusion Date'] = pd.to_datetime(df['Inclusion Date'], errors='coerce')
            
        # Aseguramos numéricos para filtros
        for col in ['Year', 'Month', 'Q', 'Week']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        return df
    except Exception as e:
        st.error(f"Error Sheets: {e}")
        return None

with st.spinner('Cargando base de Destaques...'):
    df = load_data()

# --- 4. CHAT VISUAL ---
if df is not None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({"role": "assistant", "content": "Hola. Entiendo que **cada fila es un destaque**. Analizo Release Date, Año, Q, etc. ¿Qué necesitas saber?"})

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if isinstance(message["content"], str):
                st.markdown(message["content"])

    if prompt := st.chat_input("Ej: Diferencia de Spotify Enero 2025 vs 2026"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            caja_loading = st.empty()
            caja_loading.markdown(f"🧠 *Analizando cada fila como un destaque...*")

            try:
                info_columnas = df.dtypes.to_markdown()
                head_data = df.head(3).to_markdown(index=False)

                # --- PROMPT MAESTRO CORREGIDO ---
                prompt_maestro = f"""
                Actúa como Analista de Datos Senior para ONErpm.
                Tienes un DataFrame `df`.
                Metadata: {info_columnas}
                Muestra: {head_data}
                
                DICCIONARIO DE DATOS Y LÓGICA DE NEGOCIO (LEER ATENTAMENTE):
                1. **UNIDAD DE MEDIDA:** CADA FILA del DataFrame representa UN DESTAQUE (Placement) logrado en una playlist.
                   - Si hay 10 filas, hubo 10 destaques.
                
                2. **INTERPRETACIÓN DE FECHAS:**
                   - Columnas `Year`, `Month`, `Q`, `Week`, `Inclusion Date`: Indican CUÁNDO ocurrió el destaque (Fecha de reporte/ingreso a playlist).
                   - Columna `Release Date`: Indica cuándo se lanzó la canción.
                
                3. **CÓMO RESPONDER:**
                   - Si el usuario pregunta "¿Cuántos destaques en Enero 2025?", FILTRA usando `Year` y `Month` (o `Inclusion Date`).
                   - Si el usuario pregunta por "Lanzamientos de 2024 con destaques en 2025", usa `Release Date` Y `Year`.
                   - NO ignores ninguna columna, úsalas con lógica.
                
                4. **FORMATO:**
                   - Genera SOLO código Python ejecutable.
                   - Usa `st.write`, `st.metric`, `st.dataframe`.
                   - Para gráficos: usa `fig, ax = plt.subplots()` y `st.pyplot(fig)`.
                   - Si piden porcentajes, calcula primero los valores absolutos y muéstralos.
                
                Usuario pregunta: "{prompt}"
                """

                model = genai.GenerativeModel(valid_model_name)
                response = model.generate_content(prompt_maestro)
                codigo = response.text.replace("```python", "").replace("```", "").replace("plt.show()", "").strip()
                
                caja_loading.empty()
                local_vars = {"df": df, "pd": pd, "st": st, "plt": plt, "sns": sns}
                exec(codigo, {}, local_vars)
                st.session_state.messages.append({"role": "assistant", "content": "✅ Análisis completado."})

            except Exception as e:
                caja_loading.error(f"Error técnico: {str(e)}")
