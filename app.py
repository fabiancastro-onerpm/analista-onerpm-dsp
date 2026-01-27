import streamlit as st
import pandas as pd
import google.generativeai as genai
from streamlit_gsheets import GSheetsConnection

st.set_page_config(page_title="Analista ONErpm AI", page_icon="🎹")
st.title("🎹 Chat con Datos ONErpm (Modo Diagnóstico)")

# --- 1. CONFIGURACIÓN DE LLAVES ---
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
except Exception:
    st.error("⚠️ Falta la API Key en los Secrets.")
    st.stop()

# --- 2. DIAGNÓSTICO DE MODELOS (NUEVO) ---
# Esto busca qué modelos tienes realmente disponibles para evitar errores 404
@st.cache_resource
def get_working_model():
    try:
        # Preguntamos a Google qué modelos ve tu llave
        available_models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                available_models.append(m.name)
        
        # Intentamos elegir el mejor disponible
        if 'models/gemini-1.5-flash' in available_models:
            return 'gemini-1.5-flash'
        elif 'models/gemini-pro' in available_models:
            return 'gemini-pro'
        elif len(available_models) > 0:
            # Si no están los favoritos, usamos el primero que encontremos
            return available_models[0].replace('models/', '') 
        else:
            return None
    except Exception as e:
        return None

model_name = get_working_model()

if model_name:
    st.success(f"✅ Conectado exitosamente usando el modelo: **{model_name}**")
else:
    st.error("""
    ❌ **ERROR CRÍTICO: Tu API Key no tiene acceso a ningún modelo.**
    
    Solución:
    1. Ve a Google Cloud Console.
    2. Busca "Generative Language API".
    3. Dale al botón "ENABLE" (Habilitar).
    """)
    st.stop()

# --- 3. CONEXIÓN DATOS ---
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
        st.error(f"Error Sheets: {e}")
        return None

df = load_data()

# --- 4. CHAT ---
if df is not None:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Pregunta algo..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            caja = st.empty()
            caja.markdown("⏳ Analizando...")
            
            try:
                model = genai.GenerativeModel(model_name) # Usamos el modelo que encontramos
                
                info = df.dtypes.to_markdown()
                head = df.head(3).to_markdown(index=False)
                
                prompt_full = f"""
                Actúa como experto en Pandas Python.
                Data: {info}
                Muestra: {head}
                User: {prompt}
                Genera SOLO código Python. Guarda resultado en variable `resultado`.
                """
                
                response = model.generate_content(prompt_full)
                code = response.text.replace("```python", "").replace("```", "").strip()
                
                local_vars = {"df": df, "pd": pd}
                exec(code, {}, local_vars)
                res = local_vars.get("resultado", "Sin respuesta.")
                
                caja.markdown(res)
                st.session_state.messages.append({"role": "assistant", "content": res})
                
            except Exception as e:
                caja.error(f"Error: {e}")
