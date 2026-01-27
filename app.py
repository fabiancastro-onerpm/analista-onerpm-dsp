import streamlit as st
import pandas as pd
import google.generativeai as genai
from streamlit_gsheets import GSheetsConnection
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURACIÓN VISUAL ---
st.set_page_config(page_title="Analista ONErpm AI", page_icon="🎹", layout="centered")
st.title("🎹 Chat con Datos ONErpm")
st.markdown("---")

# --- 1. CONEXIÓN API ---
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except Exception:
    st.error("⚠️ Error: No se detectó la API Key en los Secrets.")
    st.stop()

# --- 2. CARGA DE DATOS ---
url_sheet = "https://docs.google.com/spreadsheets/d/10y2YowTEgQYdWxs6c8D0fgJDDwGIT8_wyH0rQbERgG0/edit?gid=1919114384#gid=1919114384"

@st.cache_data(ttl=600)
def load_data():
    conn = st.connection("gsheets", type=GSheetsConnection)
    try:
        df = conn.read(spreadsheet=url_sheet, worksheet="DSP COPY")
        # Limpieza automática de fechas
        if 'Release Date' in df.columns:
            df['Release Date'] = pd.to_datetime(df['Release Date'], errors='coerce')
        return df
    except Exception as e:
        st.error(f"Error conectando a Sheets: {e}")
        return None

with st.spinner('Conectando con la nube...'):
    df = load_data()

# --- 3. LÓGICA DEL CHAT VISUAL ---
if df is not None:
    # Mensaje de bienvenida
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({"role": "assistant", "content": "Hola 👋. Pregúntame lo que quieras. Puedo generar **tablas** y **gráficas**."})

    # Mostrar historial
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # Si el contenido es texto, lo mostramos
            if isinstance(message["content"], str):
                st.markdown(message["content"])
            # (Nota: Las gráficas pasadas no se guardan en historial simple para ahorrar memoria, 
            # pero las nuevas se generarán al momento)

    # Input del usuario
    if prompt := st.chat_input("Ej: Haz una gráfica de torta comparando Spotify 2025 vs 2026"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            caja_loading = st.empty()
            caja_loading.markdown("🎨 *Diseñando respuesta visual...*")

            try:
                # --- CEREBRO: INSTRUCCIONES PARA STREAMLIT ---
                info_columnas = df.dtypes.to_markdown()
                head_data = df.head(3).to_markdown(index=False)

                prompt_maestro = f"""
                Actúa como un Científico de Datos experto usando Streamlit.
                Tienes un DataFrame `df`.
                Metadata: {info_columnas}
                Muestra: {head_data}
                
                Usuario pide: "{prompt}"
                
                TU TAREA:
                Genera código Python que se ejecutará dentro de una app Streamlit.
                
                REGLAS OBLIGATORIAS:
                1. PARA TEXTO: Usa `st.write("Texto")` o `st.success("Dato")`. NO uses print().
                2. PARA TABLAS: Usa `st.dataframe(df_resultado)`.
                3. PARA GRÁFICAS:
                   - Usa `fig, ax = plt.subplots()`
                   - Usa seaborn (`sns`) o matplotlib.
                   - AL FINAL DE LA GRÁFICA: usa `st.pyplot(fig)`.
                   - NO uses plt.show().
                4. Si calculas un porcentaje, muéstralo claro con `st.metric()`.
                5. Importa lo necesario dentro del código si hace falta.
                
                Dame SOLO el código, sin ```python al inicio.
                """

                # Intentamos con Flash, si falla vamos a Pro
                try:
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    response = model.generate_content(prompt_maestro)
                except:
                    model = genai.GenerativeModel('gemini-pro')
                    response = model.generate_content(prompt_maestro)

                codigo = response.text.replace("```python", "").replace("```", "").replace("plt.show()", "#plt.show() anulado").strip()
                
                # Limpiamos el mensaje de carga
                caja_loading.empty()
                
                # --- EJECUCIÓN VISUAL ---
                # Pasamos las librerías necesarias al entorno de ejecución
                local_vars = {
                    "df": df, 
                    "pd": pd, 
                    "st": st, 
                    "plt": plt, 
                    "sns": sns
                }
                exec(codigo, {}, local_vars)
                
                # Guardamos solo el texto del prompt en historial para referencia
                st.session_state.messages.append({"role": "assistant", "content": "✅ Análisis visual generado arriba."})

            except Exception as e:
                caja_loading.error(f"Hubo un error técnico: {str(e)}")
                with st.expander("Ver detalle del error"):
                    st.write(e)
