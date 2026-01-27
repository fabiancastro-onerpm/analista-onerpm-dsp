import streamlit as st
import pandas as pd
import google.generativeai as genai
from streamlit_gsheets import GSheetsConnection
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURACIÓN VISUAL ---
st.set_page_config(page_title="Analista ONErpm AI", page_icon="🎹", layout="centered")
st.title("🎹 Chat con Datos ONErpm")
st.caption("Modo: Auto-detección de modelo + Gráficos")
st.markdown("---")

# --- 1. CONEXIÓN API ---
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except Exception:
    st.error("⚠️ Error: No se detectó la API Key en los Secrets.")
    st.stop()

# --- 2. FUNCIÓN PARA ENCONTRAR EL MODELO QUE SÍ FUNCIONA (NUEVO) ---
@st.cache_resource
def get_best_model():
    """Pregunta a Google qué modelos tienes habilitados y elige el mejor."""
    try:
        available_models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                available_models.append(m.name)
        
        # Lista de preferencia (del más rápido al más estándar)
        preferences = ['models/gemini-1.5-flash', 'models/gemini-1.5-pro', 'models/gemini-pro']
        
        for pref in preferences:
            if pref in available_models:
                return pref
        
        # Si no encuentra favoritos, usa el primero que sirva
        if available_models:
            return available_models[0]
            
        return None
    except Exception as e:
        return None

# Seleccionamos el modelo automáticamente
valid_model_name = get_best_model()

if not valid_model_name:
    st.error("""
    ❌ **ERROR CRÍTICO: Tu API Key no tiene modelos habilitados.**
    
    SOLUCIÓN OBLIGATORIA:
    1. Ve a Google Cloud Console (https://console.cloud.google.com/apis/library/generativelanguage.googleapis.com)
    2. Selecciona tu proyecto.
    3. Dale al botón azul **ENABLE** (Habilitar).
    """)
    st.stop()
else:
    # Mostramos pequeñito qué modelo estamos usando para que sepas que funcionó
    st.toast(f"✅ Conectado a: {valid_model_name}")

# --- 3. CARGA DE DATOS ---
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
        st.error(f"Error conectando a Sheets: {e}")
        return None

with st.spinner('Conectando con la nube...'):
    df = load_data()

# --- 4. CHAT VISUAL ---
if df is not None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({"role": "assistant", "content": "Hola 👋. Pregúntame y generaré tablas o gráficas."})

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if isinstance(message["content"], str):
                st.markdown(message["content"])

    if prompt := st.chat_input("Ej: Gráfica de barras de Spotify por año"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            caja_loading = st.empty()
            caja_loading.markdown(f"🤖 *Pensando con {valid_model_name.split('/')[-1]}...*")

            try:
                # Prompt mejorado para visualización
                info_columnas = df.dtypes.to_markdown()
                head_data = df.head(3).to_markdown(index=False)

                prompt_maestro = f"""
                Actúa como experto en Data Science con Python/Streamlit.
                DataFrame `df` cargado.
                Metadata: {info_columnas}
                Muestra: {head_data}
                Usuario: "{prompt}"
                
                REGLAS:
                1. Genera SOLO código Python ejecutable.
                2. Usa `st.write()` para texto y `st.dataframe()` para tablas.
                3. Para GRÁFICOS: Usa `fig, ax = plt.subplots()`, usa `sns` (seaborn), y finaliza con `st.pyplot(fig)`.
                4. NO uses print() ni plt.show().
                5. Si hay error de datos, usa st.error().
                """

                # Usamos el modelo que ENCONTRAMOS que sí funciona
                model = genai.GenerativeModel(valid_model_name)
                response = model.generate_content(prompt_maestro)

                codigo = response.text.replace("```python", "").replace("```", "").replace("plt.show()", "").strip()
                
                caja_loading.empty()
                
                # Ejecución
                local_vars = {"df": df, "pd": pd, "st": st, "plt": plt, "sns": sns}
                exec(codigo, {}, local_vars)
                
                st.session_state.messages.append({"role": "assistant", "content": "✅ Gráfico/Dato generado arriba."})

            except Exception as e:
                caja_loading.error(f"Error: {str(e)}")
