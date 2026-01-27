import streamlit as st
import pandas as pd
import google.generativeai as genai
from streamlit_gsheets import GSheetsConnection
import matplotlib.pyplot as plt
import seaborn as sns
import re

# -----------------------------------------------------------------------------
# 1. CONFIGURACIÓN
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Analista ONErpm (Modo Selector)", page_icon="🎹", layout="wide")

# --- CONEXIÓN API ---
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except Exception:
    st.error("❌ FALTA API KEY: Configúrala en los Secrets.")
    st.stop()

# -----------------------------------------------------------------------------
# 2. SELECTOR DE MODELO (LA SOLUCIÓN AL ERROR 404)
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ Configuración del Cerebro")

@st.cache_resource
def list_available_models():
    """Pregunta a Google qué modelos REALMENTE tienes disponibles."""
    try:
        models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                models.append(m.name)
        return models
    except Exception as e:
        return []

available_models = list_available_models()

if not available_models:
    st.sidebar.error("⚠️ No se encontraron modelos. Habilita 'Generative Language API' en Google Cloud.")
    st.stop()
else:
    # Deja que el usuario ELIJA el modelo de la lista real
    selected_model = st.sidebar.selectbox(
        "Selecciona el Modelo:", 
        available_models, 
        index=0
    )
    st.sidebar.success(f"✅ Usando: {selected_model}")

# -----------------------------------------------------------------------------
# 3. CARGA Y LIMPIEZA DE DATOS
# -----------------------------------------------------------------------------
url_sheet = "https://docs.google.com/spreadsheets/d/10y2YowTEgQYdWxs6c8D0fgJDDwGIT8_wyH0rQbERgG0/edit?gid=1919114384#gid=1919114384"

@st.cache_data(ttl=600)
def load_data_expert():
    conn = st.connection("gsheets", type=GSheetsConnection)
    try:
        df = conn.read(spreadsheet=url_sheet, worksheet="DSP COPY")
        
        # --- LIMPIEZA MAESTRA ---
        df.columns = df.columns.str.strip()
        
        # Columnas Normalizadas para búsqueda
        cols_texto = ['DSP', 'Artist', 'Title', 'Playlist', 'Genre', 'Territory']
        for col in cols_texto:
            if col in df.columns:
                df[f"{col}_NORM"] = df[col].fillna("UNKNOWN").astype(str).str.strip().str.upper()

        # Números y Fechas
        if 'Year' in df.columns:
            df['Year'] = pd.to_numeric(df['Year'], errors='coerce').fillna(0).astype(int)
        
        if 'Month' in df.columns:
            df['Month'] = pd.to_numeric(df['Month'], errors='coerce').fillna(0).astype(int)

        if 'Release Date' in df.columns:
            df['Release Date'] = pd.to_datetime(df['Release Date'], errors='coerce')

        return df
    except Exception as e:
        st.error(f"Error Carga: {e}")
        return None

df = load_data_expert()

# -----------------------------------------------------------------------------
# 4. DIAGNÓSTICO VISUAL DE DATOS
# -----------------------------------------------------------------------------
if df is not None:
    st.title("🎹 Analista ONErpm")
    
    # Check rápido en sidebar
    with st.sidebar:
        st.markdown("---")
        st.write(f"📊 **Total Filas:** {len(df)}")
        if 'Year' in df.columns:
            st.write("**Años:**", sorted(df['Year'].unique()))
        if 'DSP_NORM' in df.columns:
            st.write("**DSPs:**", df['DSP_NORM'].unique())

# -----------------------------------------------------------------------------
# 5. CHAT
# -----------------------------------------------------------------------------
def extract_python_code(text):
    pattern = r"```python(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match: return match.group(1).strip()
    return text.replace("```python", "").replace("```", "").strip()

if df is not None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({"role": "assistant", "content": "Hola. Selecciona un modelo a la izquierda y pregúntame."})

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ej: Diferencia Spotify Enero 2025 vs 2026"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            caja = st.empty()
            caja.info("🧠 Analizando...")
            
            code_generated = "" # Variable para debug

            try:
                # Contexto
                unique_dsps = list(df['DSP_NORM'].unique()) if 'DSP_NORM' in df.columns else []
                
                prompt_maestro = f"""
                Actúa como Data Scientist Senior en Python.
                Usuario pregunta: "{prompt}"
                
                DATOS DISPONIBLES (DataFrame `df`):
                - DSPs (NORMALIZADOS): {unique_dsps}
                - Columnas: {list(df.columns)}
                
                REGLAS DE ORO:
                1. Usa `df['DSP_NORM'] == 'VALOR_EN_MAYUSCULAS'` para filtrar DSP.
                2. Usa `Year` y `Month` (enteros) para fechas.
                3. IMPRIME DEBUG: `st.write(f"Filas encontradas: {{len(df_filtrado)}}")`
                4. Genera SOLO código Python.
                """
                
                # Usamos el modelo seleccionado por TI en la barra lateral
                model = genai.GenerativeModel(selected_model)
                response = model.generate_content(prompt_maestro)
                
                code_generated = extract_python_code(response.text)
                
                caja.empty()
                local_vars = {"df": df, "pd": pd, "st": st, "plt": plt, "sns": sns}
                exec(code_generated, {}, local_vars)
                
                st.session_state.messages.append({"role": "assistant", "content": "✅ Listo."})

            except Exception as e:
                caja.error(f"Error: {e}")
                with st.expander("Ver código (Debug)"):
                    st.code(code_generated)
