import streamlit as st
import pandas as pd
import google.generativeai as genai
from streamlit_gsheets import GSheetsConnection

# ---------------------------------------------------------
# CONFIGURACIÓN DE LA PÁGINA
# ---------------------------------------------------------
st.set_page_config(page_title="Analista ONErpm AI", page_icon="🎹", layout="centered")
st.title("🎹 Chat con Datos ONErpm")
st.markdown("Conectado a: **DSP Global - Pestaña: DSP COPY**")

# ---------------------------------------------------------
# 1. CONEXIÓN CON GEMINI (CEREBRO)
# ---------------------------------------------------------
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except Exception:
    st.error("⚠️ Falta configurar la GOOGLE_API_KEY en los Secrets de Streamlit.")
    st.stop()

# ---------------------------------------------------------
# 2. CONEXIÓN CON GOOGLE SHEETS (DATOS)
# ---------------------------------------------------------
url_sheet = "https://docs.google.com/spreadsheets/d/10y2YowTEgQYdWxs6c8D0fgJDDwGIT8_wyH0rQbERgG0/edit?gid=1919114384#gid=1919114384"

@st.cache_data(ttl=600)
def load_data():
    conn = st.connection("gsheets", type=GSheetsConnection)
    try:
        df = conn.read(spreadsheet=url_sheet, worksheet="DSP COPY")
        if 'Release Date' in df.columns:
            df['Release Date'] = pd.to_datetime(df['Release Date'], errors='coerce')
        return df
    except Exception as e:
        st.error(f"Error leyendo Sheets: {str(e)}")
        return None

with st.spinner('Cargando datos...'):
    df = load_data()

# ---------------------------------------------------------
# 3. INTERFAZ DE CHAT Y LÓGICA BLINDADA
# ---------------------------------------------------------
if df is not None:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ej: ¿Diferencia % de Spotify en Enero 2025 vs 2026?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            caja_mensaje = st.empty()
            caja_mensaje.markdown("⏳ *Analizando datos...*")

            try:
                # --- PREPARAR DATOS ---
                info_columnas = df.dtypes.to_markdown()
                muestra_datos = df.head(3).to_markdown(index=False)

                prompt_maestro = f"""
                Actúa como analista de datos Python (Pandas).
                DataFrame `df` disponible.
                Metadata: {info_columnas}
                Muestra: {muestra_datos}
                Usuario: "{prompt}"
                
                Genera SOLO código Python.
                Guarda el resultado en variable `resultado`.
                NO print(). NO ```python.
                """

                # --- EL "PLAN B" AUTOMÁTICO ---
                try:
                    # Intento 1: Usar Flash (Rápido)
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    response = model.generate_content(prompt_maestro)
                except Exception:
                    # Intento 2: Si Flash falla, usar Pro (Estándar)
                    model = genai.GenerativeModel('gemini-pro')
                    response = model.generate_content(prompt_maestro)
                
                codigo_generado = response.text.replace("```python", "").replace("```", "").strip()

                # --- EJECUCIÓN ---
                variables_locales = {"df": df, "pd": pd}
                exec(codigo_generado, {}, variables_locales)
                
                respuesta_final = variables_locales.get("resultado", "No pude calcular una respuesta.")

                caja_mensaje.markdown(respuesta_final)
                st.session_state.messages.append({"role": "assistant", "content": respuesta_final})

            except Exception as e:
                caja_mensaje.error(f"Error técnico: {e}")
